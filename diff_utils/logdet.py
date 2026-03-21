
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

try:
    from scipy.linalg.lapack import dgbtrf, dgbtrs
except Exception as exc:
    dgbtrf = None
    dgbtrs = None
    _LAPACK_IMPORT_ERROR = exc
else:
    _LAPACK_IMPORT_ERROR = None

try:
    from scipy.linalg.lapack import zgbtrf, zgbtrs
except Exception:
    zgbtrf = None
    zgbtrs = None

@dataclass(frozen=True)
class _BandLayout:
    n: int
    kl: int
    ku: int
    ldab: int
    main_row: int

@lru_cache(maxsize=128)
def _layout(n: int, kl: int, ku: int) -> _BandLayout:
    return _BandLayout(n=n, kl=kl, ku=ku, ldab=2 * kl + ku + 1, main_row=kl + ku)

def _require_lapack() -> None:
    if dgbtrf is None or dgbtrs is None:
        raise RuntimeError(
            "SciPy LAPACK wrappers are required for banded_logdet"
        ) from _LAPACK_IMPORT_ERROR

def _require_complex_lapack() -> None:
    if zgbtrf is None or zgbtrs is None:
        raise RuntimeError(
            "SciPy complex LAPACK wrappers (zgbtrf/zgbtrs) are required for complex banded_logdet"
        )

def _check_input(A_band: torch.Tensor, kl: int, ku: int) -> _BandLayout:
    if not isinstance(kl, int) or not isinstance(ku, int):
        raise TypeError("kl and ku must be ints")
    if kl < 0 or ku < 0:
        raise ValueError(f"kl and ku must be non-negative, got kl={kl}, ku={ku}")
    if A_band.device.type != "cpu":
        raise ValueError("A_band must be on CPU")
    if A_band.dtype not in (torch.float64, torch.complex128):
        raise TypeError(f"A_band must be float64 or complex128, got {A_band.dtype}")
    if A_band.dim() != 2:
        raise ValueError(f"A_band must be rank-2, got shape {tuple(A_band.shape)}")
    n = int(A_band.shape[1])
    if n <= 0:
        raise ValueError("n must be positive")
    lay = _layout(n, kl, ku)
    if int(A_band.shape[0]) != lay.ldab:
        raise ValueError(f"A_band first dim must be 2*kl+ku+1={lay.ldab}, got {A_band.shape[0]}")
    return lay

def _as_fortran_copy(A_band: torch.Tensor, kl: int) -> np.ndarray:
    ab = np.array(A_band.detach().contiguous().numpy(), dtype=np.float64, order="F", copy=True)
    if kl > 0:
        ab[:kl, :] = 0.0
    return ab

def _as_fortran_copy_complex(A_band: torch.Tensor, kl: int) -> np.ndarray:
    ab = np.array(
        A_band.detach().resolve_conj().contiguous().numpy(),
        dtype=np.complex128,
        order="F",
        copy=True,
    )
    if kl > 0:
        ab[:kl, :] = 0.0
    return ab

def _swap_count(ipiv: np.ndarray) -> int:
    n = int(ipiv.shape[0])
    if n == 0:
        return 0
    if int(ipiv.min()) >= 1 and int(ipiv.max()) <= n:
        return int(np.count_nonzero(ipiv != (np.arange(n, dtype=ipiv.dtype) + 1)))
    return int(np.count_nonzero(ipiv != np.arange(n, dtype=ipiv.dtype)))

def _forward_impl(A_band: torch.Tensor, kl: int, ku: int):
    lay = _check_input(A_band, kl, ku)
    ab = _as_fortran_copy(A_band, kl)
    lu, ipiv, info = dgbtrf(ab, kl, ku, overwrite_ab=1)
    if info < 0:
        raise RuntimeError(f"dgbtrf invalid argument {-info}")
    u_diag = lu[lay.main_row, :]
    if info > 0 or np.any(u_diag == 0.0):
        sign = 0.0
        logabsdet = -np.inf
        singular = True
    else:
        swaps = _swap_count(ipiv)
        perm_sign = -1.0 if (swaps & 1) else 1.0
        diag_sign = float(np.prod(np.sign(u_diag)))
        sign = perm_sign * diag_sign
        logabsdet = float(np.log(np.abs(u_diag)).sum(dtype=np.float64))
        singular = False
    return lay, lu, ipiv, singular, sign, logabsdet

def _forward_impl_complex(A_band: torch.Tensor, kl: int, ku: int):
    lay = _check_input(A_band, kl, ku)
    ab = _as_fortran_copy_complex(A_band, kl)
    lu, ipiv, info = zgbtrf(ab, kl, ku, overwrite_ab=1)
    if info < 0:
        raise RuntimeError(f"zgbtrf invalid argument {-info}")
    u_diag = lu[lay.main_row, :]
    if info > 0 or np.any(u_diag == 0.0):
        sign = complex(0.0)
        logabsdet = -np.inf
        singular = True
    else:
        swaps = _swap_count(ipiv)
        perm_sign = -1.0 if (swaps & 1) else 1.0
        abs_diag = np.abs(u_diag)
        phases = u_diag / abs_diag
        sign = complex(perm_sign * np.prod(phases))
        logabsdet = float(np.log(abs_diag).sum(dtype=np.float64))
        singular = False
    return lay, lu, ipiv, singular, sign, logabsdet

def _chunk_size_for_inverse_t(n: int, target_bytes: int = 64 * 1024 * 1024) -> int:
    if n <= 0:
        return 1
    per_rhs_bytes = 8 * n
    chunk = target_bytes // per_rhs_bytes
    if chunk < 1:
        chunk = 1
    if chunk > n:
        chunk = n
    return int(chunk)

class BandedLogDet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_band: torch.Tensor, kl: int, ku: int):
        is_complex = A_band.is_complex()

        if is_complex:
            _require_complex_lapack()
            lay, lu, ipiv, singular, sign, logabsdet = _forward_impl_complex(A_band, kl, ku)
            sign_t = torch.tensor(sign, dtype=torch.complex128)
        else:
            _require_lapack()
            lay, lu, ipiv, singular, sign, logabsdet = _forward_impl(A_band, kl, ku)
            sign_t = A_band.new_tensor(sign)

        logabsdet_t = torch.tensor(logabsdet, dtype=torch.float64)
        ctx.kl = kl
        ctx.ku = ku
        ctx.n = lay.n
        ctx.ldab = lay.ldab
        ctx.main_row = lay.main_row
        ctx.lu = lu
        ctx.ipiv = ipiv
        ctx.singular = singular
        ctx.is_complex = is_complex
        ctx.mark_non_differentiable(sign_t)
        return sign_t, logabsdet_t

    @staticmethod
    def backward(ctx, grad_sign: torch.Tensor | None, grad_logabsdet: torch.Tensor | None):
        del grad_sign
        if grad_logabsdet is None:
            return None, None, None

        dtype = torch.complex128 if ctx.is_complex else torch.float64
        np_dtype = np.complex128 if ctx.is_complex else np.float64
        grad_scale = float(grad_logabsdet.detach().item())

        if grad_scale == 0.0 or ctx.singular:
            return torch.zeros((ctx.ldab, ctx.n), dtype=dtype), None, None

        n = ctx.n
        kl = ctx.kl
        ku = ctx.ku
        main_row = ctx.main_row
        lu = ctx.lu
        ipiv = ctx.ipiv

        gbtrs = zgbtrs if ctx.is_complex else dgbtrs
        trans_flag = 2 if ctx.is_complex else 1  # 2 = conjugate transpose for zgbtrs

        grad_ab = np.zeros((ctx.ldab, n), dtype=np_dtype)
        chunk = _chunk_size_for_inverse_t(n)
        for j0 in range(0, n, chunk):
            j1 = min(n, j0 + chunk)
            nrhs = j1 - j0
            rhs = np.zeros((n, nrhs), dtype=np_dtype, order="F")
            cols = np.arange(j0, j1, dtype=np.int64)
            rhs[cols, np.arange(nrhs, dtype=np.int64)] = 1.0
            sol, info = gbtrs(lu, kl, ku, rhs, ipiv, trans=trans_flag, overwrite_b=1)
            if info < 0:
                raise RuntimeError(f"gbtrs invalid argument {-info}")
            if info > 0:
                raise RuntimeError(f"gbtrs failed with info={info}")
            for local in range(nrhs):
                j = j0 + local
                i_lo = max(0, j - ku)
                i_hi = min(n - 1, j + kl)
                rows = np.arange(i_lo, i_hi + 1, dtype=np.int64)
                band_rows = main_row + rows - j
                grad_ab[band_rows, j] = sol[rows, local]

        grad_ab *= grad_scale
        grad_A = torch.from_numpy(np.ascontiguousarray(grad_ab))
        return grad_A, None, None

def banded_logdet(A_band: torch.Tensor, kl: int, ku: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return BandedLogDet.apply(A_band, int(kl), int(ku))

def dense_to_lapack_band(A: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    if A.device.type != "cpu":
        raise ValueError("A must be on CPU")
    if A.dtype not in (torch.float64, torch.complex128):
        raise TypeError("A must be float64 or complex128")
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square rank-2")
    n = int(A.shape[0])
    lay = _layout(n, kl, ku)
    band = torch.zeros((lay.ldab, n), dtype=A.dtype)
    for d in range(-kl, ku + 1):
        if abs(d) >= n:
            continue
        row = lay.main_row - d
        diag = A.diagonal(offset=d)
        if d >= 0:
            band[row, d:] = diag
        else:
            band[row, : n + d] = diag
    return band

def lapack_band_to_dense(A_band: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    lay = _check_input(A_band, kl, ku)
    dense = torch.zeros((lay.n, lay.n), dtype=A_band.dtype)
    for d in range(-kl, ku + 1):
        if abs(d) >= lay.n:
            continue
        row = lay.main_row - d
        vals = A_band[row]
        if d >= 0:
            i = torch.arange(0, lay.n - d, dtype=torch.int64)
            j = i + d
            dense[i, j] = vals[d:]
        else:
            i = torch.arange(-d, lay.n, dtype=torch.int64)
            j = i + d
            dense[i, j] = vals[: lay.n + d]
    return dense

__all__ = [
    "BandedLogDet",
    "banded_logdet",
    "dense_to_lapack_band",
    "lapack_band_to_dense",
]
