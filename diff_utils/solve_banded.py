from dataclasses import dataclass
import os
from typing import Dict, Tuple

import numpy as np
import torch

from diff_utils._ext import _cpu_ext, _tensor_has_storage


@dataclass(frozen=True)
class _RHSLayout:
    mode: str
    shape: torch.Size


_SCIPY_LINALG = None
_SCIPY_FAILED = False
_VALIDATED_PATTERNS = set()
_SCIPY_STRUCT_CACHE = {}
_SCIPY_VALUE_CACHE = {}


def _force_cpp() -> bool:
    return os.getenv("BANDED_FORCE_CPP", "0") == "1"


def _scipy_linalg():
    global _SCIPY_LINALG, _SCIPY_FAILED
    if _force_cpp():
        return None
    if _SCIPY_LINALG is None and not _SCIPY_FAILED:
        try:
            import scipy.linalg as la

            _SCIPY_LINALG = la
        except Exception:
            _SCIPY_FAILED = True
            _SCIPY_LINALG = None
    return _SCIPY_LINALG


def _validate_bandwidth(kl: int, ku: int, n: int) -> None:
    if kl < 0 or ku < 0:
        raise ValueError(f"kl and ku must be non-negative, got kl={kl}, ku={ku}")
    if kl + ku + 1 > n:
        raise ValueError(
            f"kl+ku+1 must be <= N, got kl={kl}, ku={ku}, N={n}, kl+ku+1={kl + ku + 1}"
        )


def _expected_row_nnz(i: int, n: int, kl: int, ku: int) -> int:
    lo = max(0, i - kl)
    hi = min(n - 1, i + ku)
    return hi - lo + 1


def _expected_nnz(n: int, kl: int, ku: int) -> int:
    return n * (kl + ku + 1) - (kl * (kl + 1)) // 2 - (ku * (ku + 1)) // 2


def _csr_row_indices(crow: torch.Tensor) -> torch.Tensor:
    n = crow.numel() - 1
    counts = crow[1:] - crow[:-1]
    return torch.repeat_interleave(torch.arange(n, device=crow.device), counts)


def _validate_banded_csr_pattern(
    crow: torch.Tensor,
    col: torch.Tensor,
    n: int,
    kl: int,
    ku: int,
) -> None:
    if crow.dim() != 1 or col.dim() != 1:
        raise ValueError("Expected unbatched CSR indices for a (N, N) sparse CSR tensor")
    if crow.dtype != torch.int64 or col.dtype != torch.int64:
        raise TypeError("CSR index tensors must be int64")
    if crow.numel() != n + 1:
        raise ValueError(f"crow_indices length must be N+1={n + 1}, got {crow.numel()}")
    if int(crow[0].item()) != 0:
        raise ValueError("crow_indices[0] must be 0")
    if int(crow[-1].item()) != col.numel():
        raise ValueError("crow_indices[-1] must equal number of col_indices")
    if int(crow[-1].item()) != _expected_nnz(n, kl, ku):
        raise ValueError("CSR nnz does not match the analytical banded pattern")

    for i in range(n):
        rs = int(crow[i].item())
        re = int(crow[i + 1].item())
        expected = _expected_row_nnz(i, n, kl, ku)
        if re - rs != expected:
            raise ValueError(f"Row {i} nnz mismatch: expected {expected}, got {re - rs}")
        expected_cols = torch.arange(
            max(0, i - kl),
            min(n - 1, i + ku) + 1,
            device=col.device,
            dtype=col.dtype,
        )
        if not torch.equal(col[rs:re], expected_cols):
            raise ValueError(f"Row {i} columns do not match canonical banded CSR ordering")


def _validate_banded_csr_pattern_cached(
    crow: torch.Tensor,
    col: torch.Tensor,
    n: int,
    kl: int,
    ku: int,
) -> None:
    key = (
        int(crow.data_ptr()),
        int(col.data_ptr()),
        int(crow.numel()),
        int(col.numel()),
        int(n),
        int(kl),
        int(ku),
    )
    if key in _VALIDATED_PATTERNS:
        return
    _validate_banded_csr_pattern(crow, col, n, kl, ku)
    _VALIDATED_PATTERNS.add(key)


def _rhs_to_canonical(b: torch.Tensor, n: int) -> Tuple[torch.Tensor, _RHSLayout]:
    if b.dim() == 1:
        if b.shape[0] != n:
            raise ValueError(f"Expected b shape (N,), got {tuple(b.shape)}")
        return b.reshape(1, n, 1).contiguous(), _RHSLayout("vec", b.shape)

    if b.dim() == 2:
        if b.shape[0] == n and b.shape[1] != n:
            return b.unsqueeze(0).contiguous(), _RHSLayout("nrhs", b.shape)
        if b.shape[1] == n and b.shape[0] != n:
            return b.unsqueeze(-1).contiguous(), _RHSLayout("batch_vec", b.shape)
        if b.shape[0] == n and b.shape[1] == n:
            return b.unsqueeze(0).contiguous(), _RHSLayout("nrhs", b.shape)
        raise ValueError(
            f"Ambiguous/invalid 2D b shape {tuple(b.shape)} for N={n}; expected (N, rhs) or (batch, N)"
        )

    if b.dim() == 3:
        if b.shape[1] != n:
            raise ValueError(f"Expected b shape (batch, N, rhs), got {tuple(b.shape)}")
        return b.contiguous(), _RHSLayout("batch_nrhs", b.shape)

    raise ValueError(
        f"Unsupported b rank {b.dim()}. Expected rank in {{1,2,3}} with N={n} on the solve axis"
    )


def _rhs_from_canonical(x: torch.Tensor, layout: _RHSLayout) -> torch.Tensor:
    if layout.mode == "vec":
        return x.reshape(layout.shape)
    if layout.mode == "nrhs":
        return x.squeeze(0).reshape(layout.shape)
    if layout.mode == "batch_vec":
        return x.squeeze(-1).reshape(layout.shape)
    if layout.mode == "batch_nrhs":
        return x.reshape(layout.shape)
    raise RuntimeError(f"Unknown rhs layout mode {layout.mode}")


def _values_to_band(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    n: int,
    kl: int,
    ku: int,
) -> torch.Tensor:
    ext = _cpu_ext()
    if (
        ext is not None
        and _tensor_has_storage(crow)
        and _tensor_has_storage(col)
        and _tensor_has_storage(values)
    ):
        return ext.csr_values_to_band(crow, col, values, kl, ku)
    bw = kl + ku + 1
    row = _csr_row_indices(crow)
    offsets = col - row
    band = torch.zeros((bw, n), dtype=values.dtype, device=values.device)
    band[(offsets + kl).to(torch.int64), row.to(torch.int64)] = values
    return band.contiguous()


def _csr_lu_factorize(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    n: int,
    kl: int,
    ku: int,
) -> torch.Tensor:
    ext = _cpu_ext()
    if (
        ext is not None
        and _tensor_has_storage(crow)
        and _tensor_has_storage(col)
        and _tensor_has_storage(values)
    ):
        return ext.csr_lu_factorize(crow, col, values, kl, ku)
    band = _values_to_band(crow, col, values, n, kl, ku)
    return _lu_factorize(band, kl, ku)


def _solve_scipy_banded(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    b3: torch.Tensor,
    kl: int,
    ku: int,
) -> torch.Tensor | None:
    if _force_cpp():
        return None
    la = _scipy_linalg()
    if la is None:
        return None
    n = int(crow.numel() - 1)
    if n == 1 and kl == 0 and ku == 0:
        den = values[0]
        return (b3 / den).contiguous()
    if not (_tensor_has_storage(values) and _tensor_has_storage(b3)):
        return None
    struct_key = (
        int(crow.data_ptr()),
        int(col.data_ptr()),
        int(crow.numel()),
        int(col.numel()),
        int(n),
        int(kl),
        int(ku),
        str(values.dtype),
    )
    ab = _scipy_ab_matrix(crow, col, values, kl, ku, n, struct_key)
    if ab is None:
        return None

    bmat = b3.permute(1, 0, 2).reshape(n, -1).contiguous()
    try:
        xmat = la.solve_banded((kl, ku), ab, bmat.detach().resolve_conj().numpy())
    except Exception:
        return None
    x = torch.from_numpy(np.ascontiguousarray(xmat))
    if x.dtype != b3.dtype or x.device != b3.device:
        x = x.to(dtype=b3.dtype, device=b3.device)
    return x.reshape(n, b3.shape[0], b3.shape[2]).permute(1, 0, 2).contiguous()


def _solve_adjoint_scipy_banded(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    g3: torch.Tensor,
    kl: int,
    ku: int,
) -> torch.Tensor | None:
    if _force_cpp():
        return None
    crow_h, col_h, values_h = _csr_adjoint(crow, col, values, int(crow.numel() - 1))
    return _solve_scipy_banded(crow_h, col_h, values_h, g3, ku, kl)


def _scipy_ab_matrix(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    kl: int,
    ku: int,
    n: int,
    struct_key: Tuple[int, int, int, int, int, int, int, str] | None = None,
) -> np.ndarray | None:
    if not _tensor_has_storage(values):
        return None
    if struct_key is None:
        struct_key = (
            int(crow.data_ptr()),
            int(col.data_ptr()),
            int(crow.numel()),
            int(col.numel()),
            int(n),
            int(kl),
            int(ku),
            str(values.dtype),
        )

    struct = _SCIPY_STRUCT_CACHE.get(struct_key)
    if struct is None:
        row = _csr_row_indices(crow).to(torch.int64)
        offsets = col - row
        ab_row = (ku - offsets).cpu().numpy()
        ab_col = col.cpu().numpy()
        struct = (ab_row, ab_col)
        if len(_SCIPY_STRUCT_CACHE) > 64:
            _SCIPY_STRUCT_CACHE.clear()
        _SCIPY_STRUCT_CACHE[struct_key] = struct
    else:
        ab_row, ab_col = struct

    use_cache = (os.getenv("BANDED_CACHE_VALUES", "0") == "1") and (not torch.is_grad_enabled())
    value_key = None
    ab = None
    if use_cache:
        value_key = (
            struct_key,
            int(values.data_ptr()),
            int(values._version),
            bool(values.is_conj()),
        )
        ab = _SCIPY_VALUE_CACHE.get(value_key)
    if ab is None:
        vals_np = values.detach().resolve_conj().numpy()
        ab = np.zeros((kl + ku + 1, n), dtype=vals_np.dtype)
        ab[ab_row, ab_col] = vals_np
        if use_cache:
            if len(_SCIPY_VALUE_CACHE) > 64:
                _SCIPY_VALUE_CACHE.clear()
            _SCIPY_VALUE_CACHE[value_key] = ab
    return ab


def _solve_scipy_direct_rhs(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    b: torch.Tensor,
    kl: int,
    ku: int,
) -> torch.Tensor | None:
    if _force_cpp():
        return None
    la = _scipy_linalg()
    if la is None:
        return None
    if not (_tensor_has_storage(values) and _tensor_has_storage(b)):
        return None
    n = int(crow.numel() - 1)
    if b.dim() == 1:
        if b.shape[0] != n:
            return None
    elif b.dim() == 2:
        if b.shape[0] != n:
            return None
    else:
        return None
    ab = _scipy_ab_matrix(crow, col, values, kl, ku, n)
    if ab is None:
        return None
    try:
        x_np = la.solve_banded((kl, ku), ab, b.detach().resolve_conj().numpy())
    except Exception:
        return None
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    if x.dtype != b.dtype or x.device != b.device:
        x = x.to(dtype=b.dtype, device=b.device)
    return x


def _lu_factorize(band: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    if band.device.type != "cpu":
        raise ValueError("solve_banded currently supports CPU only")
    ext = _cpu_ext()
    if ext is not None and _tensor_has_storage(band):
        return ext.lu_factorize(band, kl, ku)
    n = band.shape[1]
    lu = band.clone().contiguous()
    for k in range(n):
        pivot = lu[kl, k]
        i_hi = min(n - 1, k + kl)
        j_hi = min(n - 1, k + ku)
        for i in range(k + 1, i_hi + 1):
            li = kl + (k - i)
            factor = lu[li, i] / pivot
            lu[li, i] = factor
            for j in range(k + 1, j_hi + 1):
                dij = kl + (j - i)
                if 0 <= dij < kl + ku + 1:
                    lu[dij, i] = lu[dij, i] - factor * lu[kl + (j - k), k]
    return lu


def _lu_solve(lu: torch.Tensor, b: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    if lu.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("solve_banded currently supports CPU only")
    ext = _cpu_ext()
    if ext is not None and _tensor_has_storage(lu) and _tensor_has_storage(b):
        return ext.lu_solve(lu, b, kl, ku)
    n = lu.shape[1]
    y = torch.empty_like(b)
    for i in range(n):
        acc = b[:, i, :]
        tmax = min(kl, i)
        for t in range(1, tmax + 1):
            acc = acc - lu[kl - t, i] * y[:, i - t, :]
        y[:, i, :] = acc
    x = torch.empty_like(b)
    for i in range(n - 1, -1, -1):
        acc = y[:, i, :]
        tmax = min(ku, n - 1 - i)
        for t in range(1, tmax + 1):
            acc = acc - lu[kl + t, i] * x[:, i + t, :]
        x[:, i, :] = acc / lu[kl, i]
    return x.contiguous()


def _lu_solve_adjoint(
    lu: torch.Tensor,
    g: torch.Tensor,
    kl: int,
    ku: int,
    complex_case: bool,
) -> torch.Tensor:
    if lu.device.type != "cpu" or g.device.type != "cpu":
        raise ValueError("solve_banded currently supports CPU only")
    ext = _cpu_ext()
    if ext is not None and _tensor_has_storage(lu) and _tensor_has_storage(g):
        return ext.lu_solve_adjoint(lu, g, kl, ku, complex_case)
    n = lu.shape[1]
    y = torch.empty_like(g)
    for i in range(n):
        acc = g[:, i, :]
        tmax = min(ku, i)
        for t in range(1, tmax + 1):
            coeff = lu[kl + t, i - t]
            if complex_case:
                coeff = coeff.conj()
            acc = acc - coeff * y[:, i - t, :]
        diag = lu[kl, i]
        if complex_case:
            diag = diag.conj()
        y[:, i, :] = acc / diag
    out = torch.empty_like(g)
    for i in range(n - 1, -1, -1):
        acc = y[:, i, :]
        tmax = min(kl, n - 1 - i)
        for t in range(1, tmax + 1):
            coeff = lu[kl - t, i + t]
            if complex_case:
                coeff = coeff.conj()
            acc = acc - coeff * out[:, i + t, :]
        out[:, i, :] = acc
    return out.contiguous()


def _csr_adjoint(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row = _csr_row_indices(crow)
    counts = torch.bincount(col, minlength=n)
    crow_t = torch.empty(n + 1, dtype=torch.int64, device=crow.device)
    crow_t[0] = 0
    crow_t[1:] = torch.cumsum(counts, dim=0)
    key = col * n + row
    perm = torch.argsort(key)
    col_t = row[perm].to(torch.int64)
    values_t = values[perm]
    if torch.is_complex(values_t):
        values_t = values_t.conj()
    return crow_t, col_t, values_t


def make_banded_csr(diags: Dict[int, torch.Tensor], n: int) -> torch.Tensor:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not diags:
        raise ValueError("diags must be non-empty")

    offsets = sorted(diags.keys())
    dmin, dmax = offsets[0], offsets[-1]
    expected = list(range(dmin, dmax + 1))
    if offsets != expected:
        raise ValueError(f"diagonal keys must be contiguous offsets {expected}, got {offsets}")

    kl = -dmin
    ku = dmax
    _validate_bandwidth(kl, ku, n)

    first = diags[offsets[0]]
    device = first.device
    dtype = first.dtype

    for d in offsets:
        diag = diags[d]
        if diag.dim() != 1:
            raise ValueError(f"diagonal {d} must be 1D")
        required = n - abs(d)
        if diag.numel() != required:
            raise ValueError(
                f"diagonal {d} has length {diag.numel()}, expected {required} for n={n}"
            )
        if diag.device != device:
            raise ValueError("all diagonal tensors must be on same device")
        if diag.dtype != dtype:
            raise ValueError("all diagonal tensors must have same dtype")

    crow = torch.zeros(n + 1, dtype=torch.int64, device=device)
    row_counts = [_expected_row_nnz(i, n, kl, ku) for i in range(n)]
    crow[1:] = torch.cumsum(torch.tensor(row_counts, dtype=torch.int64, device=device), dim=0)

    col_parts = []
    val_parts = []
    for i in range(n):
        lo = max(0, i - kl)
        hi = min(n - 1, i + ku)
        col_parts.append(torch.arange(lo, hi + 1, dtype=torch.int64, device=device))
        vals = []
        for j in range(lo, hi + 1):
            d = j - i
            idx = i if d >= 0 else j
            vals.append(diags[d][idx])
        val_parts.append(torch.stack(vals))

    col = torch.cat(col_parts).to(torch.int64).contiguous()
    values = torch.cat(val_parts).to(dtype=dtype).contiguous()
    return torch.sparse_csr_tensor(crow, col, values, size=(n, n), dtype=dtype, device=device)


class _SolveBandedValuesFn(torch.autograd.Function):
    @staticmethod
    def forward(
        crow: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        b: torch.Tensor,
        kl: int,
        ku: int,
    ):
        n = crow.numel() - 1
        _validate_bandwidth(kl, ku, n)
        _validate_banded_csr_pattern_cached(crow, col, n, kl, ku)
        b3, layout = _rhs_to_canonical(b, n)
        if b3.dtype != values.dtype:
            raise TypeError(f"A and b must share dtype, got {values.dtype} and {b3.dtype}")
        if b3.device != values.device:
            raise ValueError("A and b must be on the same device")
        x3 = _solve_scipy_banded(crow, col, values, b3, kl, ku)
        if x3 is None:
            lu = _csr_lu_factorize(crow, col, values, n, kl, ku)
            x3 = _lu_solve(lu, b3, kl, ku)
        else:
            lu = torch.empty(0, dtype=values.dtype, device=values.device)
        return _rhs_from_canonical(x3, layout), lu

    @staticmethod
    def setup_context(ctx, inputs, output):
        crow, col, values, b, kl, ku = inputs
        x, lu = output
        n = crow.numel() - 1
        layout = _rhs_to_canonical(b, n)[1]
        b3, _ = _rhs_to_canonical(b, n)
        x3, _ = _rhs_to_canonical(x, n)
        ctx.kl = kl
        ctx.ku = ku
        ctx.n = n
        ctx.layout_mode = layout.mode
        ctx.b_shape = tuple(layout.shape)
        ctx.mark_non_differentiable(lu)
        ctx.save_for_backward(crow, col, values, b3, x3, lu)

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor, grad_lu: torch.Tensor):
        del grad_lu
        if grad_x is None:
            return None, None, None, None, None, None

        crow, col, values, b3, x3, lu = ctx.saved_tensors
        kl = ctx.kl
        ku = ctx.ku
        n = ctx.n
        layout = _RHSLayout(ctx.layout_mode, torch.Size(ctx.b_shape))
        grad3, _ = _rhs_to_canonical(grad_x, n)
        needs_values = ctx.needs_input_grad[2]
        needs_b = ctx.needs_input_grad[3]
        if not needs_values and not needs_b:
            return None, None, None, None, None, None

        x_for_formula = x3
        if torch.is_grad_enabled():
            crow_h, col_h, values_h = _csr_adjoint(crow, col, values, n)
            b_bar, _ = _SolveBandedValuesFn.apply(crow_h, col_h, values_h, grad_x, ku, kl)
            b_bar3, _ = _rhs_to_canonical(b_bar, n)
            if needs_values:
                x_for_formula, _ = _SolveBandedValuesFn.apply(crow, col, values, b3, kl, ku)
        else:
            b_bar3 = _solve_adjoint_scipy_banded(crow, col, values, grad3, kl, ku)
            if b_bar3 is None:
                if lu.numel() == 0:
                    lu = _csr_lu_factorize(crow, col, values, n, kl, ku)
                b_bar3 = _lu_solve_adjoint(
                    lu,
                    grad3,
                    kl,
                    ku,
                    complex_case=torch.is_complex(values),
                )

        grad_b = _rhs_from_canonical(b_bar3, layout) if needs_b else None
        grad_values = None
        if needs_values:
            rows = _csr_row_indices(crow).to(torch.int64)
            bb = b_bar3[:, rows, :]
            xx = x_for_formula[:, col.to(torch.int64), :]
            if torch.is_complex(values):
                contrib = bb * xx.conj()
            else:
                contrib = bb * xx
            grad_values = -contrib.sum(dim=(0, 2)).contiguous()

        return None, None, grad_values, grad_b, None, None

    @staticmethod
    def vmap(
        info,
        in_dims,
        crow: torch.Tensor,
        col: torch.Tensor,
        values: torch.Tensor,
        b: torch.Tensor,
        kl: int,
        ku: int,
    ):
        crow_bdim, col_bdim, values_bdim, b_bdim, kl_bdim, ku_bdim = in_dims
        if kl_bdim is not None or ku_bdim is not None:
            raise ValueError("kl and ku cannot be vmapped")
        if crow_bdim is not None or col_bdim is not None:
            raise ValueError("CSR structure batch-vmap is unsupported; vmap over values and/or b")

        if values_bdim is None and b_bdim is None:
            out = _SolveBandedValuesFn.apply(crow, col, values, b, kl, ku)
            return out, (None, None)

        batch = None
        if values_bdim is not None:
            batch = values.size(values_bdim)
        if b_bdim is not None:
            b_batch = b.size(b_bdim)
            if batch is None:
                batch = b_batch
            elif batch != b_batch:
                raise ValueError(
                    f"Mismatched vmap batch sizes for values and b: {batch} vs {b_batch}"
                )

        xs = []
        lus = []
        for i in range(batch):
            vi = values.select(values_bdim, i) if values_bdim is not None else values
            bi = b.select(b_bdim, i) if b_bdim is not None else b
            xi, lui = _SolveBandedValuesFn.apply(crow, col, vi, bi, kl, ku)
            xs.append(xi)
            lus.append(lui)

        return (torch.stack(xs, dim=0), torch.stack(lus, dim=0)), (0, 0)


def _extract_csr_components(
    A: torch.Tensor,
    kl: int,
    ku: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if A.layout != torch.sparse_csr:
        raise TypeError("A must be a sparse CSR tensor")
    if A.dim() != 2:
        raise ValueError("A must be rank-2 (N, N)")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {tuple(A.shape)}")
    if A.device.type != "cpu":
        raise ValueError("solve_banded currently supports CPU only")

    n = A.shape[0]
    _validate_bandwidth(kl, ku, n)
    crow = A.crow_indices()
    col = A.col_indices()
    values = A.values()
    return crow, col, values


def solve_banded_csr_values(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    b: torch.Tensor,
    kl: int,
    ku: int,
) -> torch.Tensor:
    if (not torch.is_grad_enabled()) and (not values.requires_grad) and (not b.requires_grad):
        x_direct = _solve_scipy_direct_rhs(crow, col, values, b, kl, ku)
        if x_direct is not None:
            return x_direct
        n = crow.numel() - 1
        b3, layout = _rhs_to_canonical(b, n)
        x3 = _solve_scipy_banded(crow, col, values, b3, kl, ku)
        if x3 is None:
            lu = _csr_lu_factorize(crow, col, values, n, kl, ku)
            x3 = _lu_solve(lu, b3, kl, ku)
        return _rhs_from_canonical(x3, layout)
    x, _ = _SolveBandedValuesFn.apply(crow, col, values, b, kl, ku)
    return x


def solve_banded(A: torch.Tensor, b: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    crow, col, values = _extract_csr_components(A, kl, ku)
    return solve_banded_csr_values(crow, col, values, b, kl, ku)


__all__ = ["make_banded_csr", "solve_banded", "solve_banded_csr_values"]
