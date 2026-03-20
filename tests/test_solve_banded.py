from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from banded import make_banded_csr, solve_banded, solve_banded_csr_values

scipy_linalg = pytest.importorskip("scipy.linalg")


def _rand_diags(
    n: int,
    kl: int,
    ku: int,
    *,
    dtype: torch.dtype,
    seed: int,
) -> Dict[int, torch.Tensor]:
    """Create diagonally dominant banded diagonals so A is nonsingular."""
    g = torch.Generator().manual_seed(seed)
    diags: Dict[int, torch.Tensor] = {}
    abs_sum = torch.zeros(n, dtype=_real_dtype(dtype))

    for d in range(-kl, ku + 1):
        if d == 0:
            continue
        length = n - abs(d)
        if torch.is_complex(torch.empty((), dtype=dtype)):
            re = torch.randn(length, dtype=_real_dtype(dtype), generator=g) * 0.05
            im = torch.randn(length, dtype=_real_dtype(dtype), generator=g) * 0.05
            vals = re + 1j * im
            vals = vals.to(dtype)
        else:
            vals = (torch.randn(length, dtype=dtype, generator=g) * 0.05).to(dtype)

        diags[d] = vals
        rows = torch.arange(max(0, -d), min(n, n - d))
        abs_sum[rows] += vals.abs().to(abs_sum.dtype)

    margin = torch.full((n,), 2.0, dtype=abs_sum.dtype)
    main = abs_sum + margin
    if torch.is_complex(torch.empty((), dtype=dtype)):
        main = main.to(dtype) + 0.1j * torch.ones(n, dtype=dtype)
    else:
        main = main.to(dtype)

    diags[0] = main
    return diags


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return {
        torch.float32: torch.float32,
        torch.float64: torch.float64,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }[dtype]


def _scipy_ab_from_diags(diags: Dict[int, torch.Tensor], n: int, kl: int, ku: int) -> np.ndarray:
    """Map diagonal dictionary to SciPy solve_banded storage."""
    ab = np.zeros((kl + ku + 1, n), dtype=np.asarray(diags[0].cpu()).dtype)
    for d, v in diags.items():
        arr = np.asarray(v.detach().cpu())
        row = ku - d
        if d >= 0:
            ab[row, d:] = arr
        else:
            ab[row, : n + d] = arr
    return ab


def _assert_residual_small(
    A: torch.Tensor, x: torch.Tensor, b: torch.Tensor, tol: float = 1e-8
) -> None:
    """Check ||Ax-b|| is small, handling vector/multi-RHS and batched shapes."""
    dense_A = A.to_dense()
    if b.dim() == 1:
        r = dense_A @ x - b
    elif b.dim() == 2 and b.shape[0] == A.shape[0]:
        r = dense_A @ x - b
    elif b.dim() == 2:
        r = torch.einsum("ij,bj->bi", dense_A, x) - b
    else:
        r = torch.einsum("ij,bjr->bir", dense_A, x) - b
    assert torch.linalg.norm(r).item() < tol


def _csr_rows(crow: torch.Tensor) -> torch.Tensor:
    counts = crow[1:] - crow[:-1]
    return torch.repeat_interleave(torch.arange(crow.numel() - 1), counts)


def _to_b3(t: torch.Tensor, n: int) -> torch.Tensor:
    if t.dim() == 1:
        return t.reshape(1, n, 1)
    if t.dim() == 2:
        if t.shape[0] == n:
            return t.unsqueeze(0)
        if t.shape[1] == n:
            return t.unsqueeze(-1)
    if t.dim() == 3 and t.shape[1] == n:
        return t
    raise ValueError(f"Unsupported shape {tuple(t.shape)} for N={n}")


def _from_b3(t3: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if like.dim() == 1:
        return t3.reshape_as(like)
    if like.dim() == 2:
        if like.shape[0] == t3.shape[1]:
            return t3.squeeze(0).reshape_as(like)
        return t3.squeeze(-1).reshape_as(like)
    return t3.reshape_as(like)


def test_correctness_tridiagonal_against_scipy_float64() -> None:
    """Verifies solve_banded matches SciPy for a real tridiagonal system."""
    n = 64
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=0)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, dtype=torch.float64)

    x = solve_banded(A, b, 1, 1)
    ab = _scipy_ab_from_diags(diags, n, 1, 1)
    x_ref = scipy_linalg.solve_banded((1, 1), ab, b.numpy())
    np.testing.assert_allclose(x.detach().numpy(), x_ref, rtol=1e-10, atol=1e-10)


def test_correctness_general_banded_against_scipy_float64() -> None:
    """Verifies solve_banded matches SciPy for a real general banded system (kl=2, ku=3)."""
    n = 80
    kl, ku = 2, 3
    diags = _rand_diags(n, kl, ku, dtype=torch.float64, seed=1)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, dtype=torch.float64)

    x = solve_banded(A, b, kl, ku)
    ab = _scipy_ab_from_diags(diags, n, kl, ku)
    x_ref = scipy_linalg.solve_banded((kl, ku), ab, b.numpy())
    np.testing.assert_allclose(x.detach().numpy(), x_ref, rtol=1e-10, atol=1e-10)


def test_correctness_complex128_against_scipy() -> None:
    """Verifies complex solve matches SciPy for a complex128 tridiagonal system."""
    n = 48
    diags = _rand_diags(n, 1, 1, dtype=torch.complex128, seed=2)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, dtype=torch.float64) + 1j * torch.randn(n, dtype=torch.float64)
    b = b.to(torch.complex128)

    x = solve_banded(A, b, 1, 1)
    ab = _scipy_ab_from_diags(diags, n, 1, 1)
    x_ref = scipy_linalg.solve_banded((1, 1), ab, b.numpy())
    np.testing.assert_allclose(x.detach().numpy(), x_ref, rtol=1e-10, atol=1e-10)


def test_correctness_multiple_rhs_shape() -> None:
    """Verifies linearity and shape preservation for multiple RHS with shape (N, rhs)."""
    n, rhs = 40, 4
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=3)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, rhs, dtype=torch.float64)

    x = solve_banded(A, b, 1, 1)
    assert x.shape == b.shape
    _assert_residual_small(A, x, b)


def test_correctness_batch_shape() -> None:
    """Verifies independent solves across a batch of RHS vectors with shape (batch, N)."""
    n, batch = 30, 5
    diags = _rand_diags(n, 2, 1, dtype=torch.float64, seed=4)
    A = make_banded_csr(diags, n)
    b = torch.randn(batch, n, dtype=torch.float64)

    x = solve_banded(A, b, 2, 1)
    assert x.shape == b.shape
    _assert_residual_small(A, x, b)


def test_correctness_batch_multiple_rhs_shape() -> None:
    """Verifies batched multi-RHS solves with shape (batch, N, rhs)."""
    n, batch, rhs = 28, 3, 2
    diags = _rand_diags(n, 1, 2, dtype=torch.float64, seed=5)
    A = make_banded_csr(diags, n)
    b = torch.randn(batch, n, rhs, dtype=torch.float64)

    x = solve_banded(A, b, 1, 2)
    assert x.shape == b.shape
    _assert_residual_small(A, x, b)


@pytest.mark.parametrize("kl,ku,seed", [(1, 1, 6), (2, 0, 31), (0, 2, 32), (2, 3, 33)])
def test_gradcheck_values_real_multiple_bandwidths(kl: int, ku: int, seed: int) -> None:
    n = 12
    diags = _rand_diags(n, kl, ku, dtype=torch.float64, seed=seed)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    values = A.values().detach().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64)

    def fn(v: torch.Tensor) -> torch.Tensor:
        return solve_banded_csr_values(crow, col, v, b, kl, ku)

    assert gradcheck(fn, (values,), eps=1e-6, atol=1e-4)


def test_gradcheck_values_tridiagonal_complex() -> None:
    """Verifies Wirtinger VJP wrt CSR values for a complex tridiagonal system."""
    n = 8
    diags = _rand_diags(n, 1, 1, dtype=torch.complex128, seed=7)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    values = A.values().detach().requires_grad_(True)
    b = (torch.randn(n, dtype=torch.float64) + 1j * torch.randn(n, dtype=torch.float64)).to(
        torch.complex128
    )

    def fn(v: torch.Tensor) -> torch.Tensor:
        return solve_banded_csr_values(crow, col, v, b, 1, 1)

    assert gradcheck(fn, (values,), eps=1e-6, atol=1e-4)


def test_gradcheck_b_real() -> None:
    """Verifies VJP wrt b for a real system equals finite-difference Jacobian."""
    n = 9
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=8)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    def fn(rhs: torch.Tensor) -> torch.Tensor:
        return solve_banded(A, rhs, 1, 1)

    assert gradcheck(fn, (b,), eps=1e-6, atol=1e-4)


def test_gradcheck_b_complex() -> None:
    """Verifies VJP wrt b for a complex system using Wirtinger finite-difference checks."""
    n = 9
    diags = _rand_diags(n, 1, 1, dtype=torch.complex128, seed=9)
    A = make_banded_csr(diags, n)
    b = (torch.randn(n, dtype=torch.float64) + 1j * torch.randn(n, dtype=torch.float64)).to(
        torch.complex128
    )
    b = b.requires_grad_(True)

    def fn(rhs: torch.Tensor) -> torch.Tensor:
        return solve_banded(A, rhs, 1, 1)

    assert gradcheck(fn, (b,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("kl,ku,seed", [(1, 1, 10), (2, 3, 34), (3, 1, 35)])
def test_gradcheck_b_real_multiple_bandwidths(kl: int, ku: int, seed: int) -> None:
    n = 14
    diags = _rand_diags(n, kl, ku, dtype=torch.float64, seed=seed)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    def fn(rhs: torch.Tensor) -> torch.Tensor:
        return solve_banded(A, rhs, kl, ku)

    assert gradcheck(fn, (b,), eps=1e-6, atol=1e-4)


def test_gradcheck_batched_case() -> None:
    """Verifies gradients for batched RHS solves share the same analytical VJP as unbatched solves."""
    n, batch = 8, 3
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=11)
    A = make_banded_csr(diags, n)
    b = torch.randn(batch, n, dtype=torch.float64, requires_grad=True)

    def fn(rhs: torch.Tensor) -> torch.Tensor:
        return solve_banded(A, rhs, 1, 1)

    assert gradcheck(fn, (b,), eps=1e-6, atol=1e-4)


def test_gradcheck_double_backward_create_graph() -> None:
    """Verifies create_graph=True enables finite, non-null second derivatives."""
    n = 7
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=12)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    values = A.values().detach().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    out = solve_banded_csr_values(crow, col, values, b, 1, 1)
    loss = (out.square()).sum()
    g_values, g_b = torch.autograd.grad(loss, (values, b), create_graph=True)

    second_obj = (g_values.square()).sum() + (g_b.square()).sum()
    h_values, h_b = torch.autograd.grad(second_obj, (values, b))
    assert torch.isfinite(h_values).all()
    assert torch.isfinite(h_b).all()


@pytest.mark.parametrize("kl,ku,seed", [(0, 2, 41), (1, 1, 42), (2, 3, 43)])
def test_backward_vjp_formula_real_multiple_bandwidths(kl: int, ku: int, seed: int) -> None:
    """Verifies real VJP formulas b_bar=A^{-T}x_bar and values_bar=-(b_bar[row]*x[col]) for multiple bandwidths."""
    n, batch, rhs = 18, 3, 2
    diags = _rand_diags(n, kl, ku, dtype=torch.float64, seed=seed)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    rows = _csr_rows(crow).to(torch.int64)

    values = A.values().detach().requires_grad_(True)
    b = torch.randn(batch, n, rhs, dtype=torch.float64, requires_grad=True)
    x = solve_banded_csr_values(crow, col, values, b, kl, ku)
    x_bar = torch.randn_like(x)
    loss = (x * x_bar).sum()
    grad_values, grad_b = torch.autograd.grad(loss, (values, b))

    with torch.no_grad():
        x3 = _to_b3(x.detach(), n)
        xbar3 = _to_b3(x_bar.detach(), n)
        dense = torch.sparse_csr_tensor(crow, col, values.detach(), (n, n)).to_dense()
        bbar_flat = torch.linalg.solve(
            dense.transpose(0, 1),
            xbar3.permute(1, 0, 2).reshape(n, -1),
        )
        bbar3 = bbar_flat.reshape(n, batch, rhs).permute(1, 0, 2).contiguous()
        grad_values_ref = -(bbar3[:, rows, :] * x3[:, col, :]).sum(dim=(0, 2))
        grad_b_ref = _from_b3(bbar3, b)

    torch.testing.assert_close(grad_b, grad_b_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(grad_values, grad_values_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kl,ku,seed", [(1, 1, 44), (2, 1, 45)])
def test_backward_vjp_formula_complex_multiple_bandwidths(kl: int, ku: int, seed: int) -> None:
    """Verifies complex Wirtinger VJP formulas b_bar=A^{-H}x_bar and values_bar=-conj(b_bar[row])*conj(x[col])."""
    n, batch, rhs = 16, 2, 3
    diags = _rand_diags(n, kl, ku, dtype=torch.complex128, seed=seed)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    rows = _csr_rows(crow).to(torch.int64)

    values = A.values().detach().requires_grad_(True)
    b = (
        torch.randn(batch, n, rhs, dtype=torch.float64)
        + 1j * torch.randn(batch, n, rhs, dtype=torch.float64)
    ).to(torch.complex128)
    b = b.requires_grad_(True)
    x = solve_banded_csr_values(crow, col, values, b, kl, ku)
    x_bar = (
        torch.randn_like(x.real, dtype=torch.float64)
        + 1j * torch.randn_like(x.real, dtype=torch.float64)
    ).to(torch.complex128)
    loss = torch.real((x * x_bar.conj()).sum())
    grad_values, grad_b = torch.autograd.grad(loss, (values, b))

    with torch.no_grad():
        x3 = _to_b3(x.detach(), n)
        xbar3 = _to_b3(x_bar.detach(), n)
        dense = torch.sparse_csr_tensor(crow, col, values.detach(), (n, n)).to_dense()
        bbar_flat = torch.linalg.solve(
            dense.conj().transpose(0, 1),
            xbar3.permute(1, 0, 2).reshape(n, -1),
        )
        bbar3 = bbar_flat.reshape(n, batch, rhs).permute(1, 0, 2).contiguous()
        grad_values_ref = -(bbar3[:, rows, :] * x3[:, col, :].conj()).sum(dim=(0, 2))
        grad_b_ref = _from_b3(bbar3, b)

    torch.testing.assert_close(grad_b, grad_b_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(grad_values, grad_values_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kl,ku,seed", [(1, 1, 46), (2, 3, 47)])
def test_backward_matches_dense_autograd_values_and_b(kl: int, ku: int, seed: int) -> None:
    """Verifies gradients wrt sparse CSR values and RHS match dense torch.linalg.solve autograd."""
    n, rhs = 14, 3
    diags = _rand_diags(n, kl, ku, dtype=torch.float64, seed=seed)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    rows = _csr_rows(crow).to(torch.int64)

    values = A.values().detach().requires_grad_(True)
    b = torch.randn(n, rhs, dtype=torch.float64, requires_grad=True)
    w = torch.randn(n, rhs, dtype=torch.float64)

    x_sparse = solve_banded_csr_values(crow, col, values, b, kl, ku)
    loss_sparse = (x_sparse * w).sum()
    g_values_sparse, g_b_sparse = torch.autograd.grad(loss_sparse, (values, b))

    dense = torch.sparse_csr_tensor(crow, col, values, (n, n)).to_dense()
    x_dense = torch.linalg.solve(dense, b)
    loss_dense = (x_dense * w).sum()
    g_dense_A, g_b_dense = torch.autograd.grad(loss_dense, (dense, b))
    g_values_dense = g_dense_A[rows, col]

    torch.testing.assert_close(g_b_sparse, g_b_dense, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g_values_sparse, g_values_dense, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kl,ku,seed", [(1, 1, 48), (2, 3, 49)])
def test_double_backward_create_graph_multiple_bandwidths(kl: int, ku: int, seed: int) -> None:
    """Verifies second-order derivatives are finite for values and b across multiple bandwidths."""
    n = 7
    diags = _rand_diags(n, kl, ku, dtype=torch.float64, seed=seed)
    A = make_banded_csr(diags, n)
    crow, col = A.crow_indices(), A.col_indices()
    values = A.values().detach().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)
    out = solve_banded_csr_values(crow, col, values, b, kl, ku)
    loss = (out.square()).sum()
    g_values, g_b = torch.autograd.grad(loss, (values, b), create_graph=True)
    second_obj = (g_values.square()).sum() + (g_b.square()).sum()
    h_values, h_b = torch.autograd.grad(second_obj, (values, b))
    assert torch.isfinite(h_values).all()
    assert torch.isfinite(h_b).all()


def test_edge_case_n1_scalar_system() -> None:
    """Verifies the solver reduces to scalar division when N=1."""
    A = make_banded_csr({0: torch.tensor([2.5], dtype=torch.float64)}, 1)
    b = torch.tensor([5.0], dtype=torch.float64)
    x = solve_banded(A, b, 0, 0)
    torch.testing.assert_close(x, torch.tensor([2.0], dtype=torch.float64))


def test_edge_case_triangular_matches_solve_triangular() -> None:
    """Verifies one-sided bandwidth systems match torch.linalg.solve_triangular exactly."""
    n = 16

    diags_l = _rand_diags(n, 2, 0, dtype=torch.float64, seed=13)
    A_l = make_banded_csr(diags_l, n)
    b = torch.randn(n, dtype=torch.float64)
    x_l = solve_banded(A_l, b, 2, 0)
    x_l_ref = torch.linalg.solve_triangular(A_l.to_dense(), b.unsqueeze(-1), upper=False).squeeze(
        -1
    )
    torch.testing.assert_close(x_l, x_l_ref, rtol=1e-10, atol=1e-10)

    diags_u = _rand_diags(n, 0, 2, dtype=torch.float64, seed=14)
    A_u = make_banded_csr(diags_u, n)
    x_u = solve_banded(A_u, b, 0, 2)
    x_u_ref = torch.linalg.solve_triangular(A_u.to_dense(), b.unsqueeze(-1), upper=True).squeeze(-1)
    torch.testing.assert_close(x_u, x_u_ref, rtol=1e-10, atol=1e-10)


def test_edge_case_singular_system_tolerated() -> None:
    n = 5
    d0 = torch.tensor([2.0, 2.0, 0.0, 2.0, 2.0], dtype=torch.float64)
    A = make_banded_csr({0: d0}, n)
    b = torch.randn(n, dtype=torch.float64)
    _ = solve_banded(A, b, 0, 0)


def test_edge_case_diagonal_dominant_stability() -> None:
    """Verifies diagonally dominant systems have small residual and stable solve behavior."""
    n = 128
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=15)
    A = make_banded_csr(diags, n)
    b = torch.randn(n, dtype=torch.float64)

    x = solve_banded(A, b, 1, 1)
    _assert_residual_small(A, x, b, tol=1e-9)


def test_edge_case_bandwidth_exceeds_matrix_size_error() -> None:
    """Verifies invalid bandwidth (kl+ku+1>N) is rejected."""
    n = 4
    d0 = torch.ones(n, dtype=torch.float64)
    A = make_banded_csr({0: d0}, n)
    b = torch.ones(n, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"kl\+ku\+1"):
        solve_banded(A, b, 3, 1)


@pytest.mark.performance
def test_performance_vs_dense_reports_speedup() -> None:
    """Benchmarks tridiagonal solve against dense solve and reports empirical speedup."""
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    ns = [100, 500, 2000, 5000]
    speedups = {}

    for n in ns:
        d0 = torch.full((n,), 4.0, dtype=torch.float64)
        d1 = torch.full((n - 1,), -1.0, dtype=torch.float64)
        A = make_banded_csr({-1: d1, 0: d0, 1: d1}, n)
        b = torch.randn(n, dtype=torch.float64)

        dense = A.to_dense()

        _ = solve_banded(A, b, 1, 1)
        _ = torch.linalg.solve(dense, b)

        reps = 5 if n <= 500 else 2

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = solve_banded(A, b, 1, 1)
        t_banded = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = torch.linalg.solve(dense, b)
        t_dense = (time.perf_counter() - t0) / reps

        speedups[n] = t_dense / t_banded

    print("speedups_dense_over_banded=", speedups)
    assert speedups[5000] > 1.0


@pytest.mark.performance
def test_performance_empirical_linear_scaling() -> None:
    """Verifies approximately O(N) scaling for tridiagonal solves by timing across N."""
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    ns = [100, 500, 2000, 5000]
    per_n = []

    for n in ns:
        d0 = torch.full((n,), 4.0, dtype=torch.float64)
        d1 = torch.full((n - 1,), -1.0, dtype=torch.float64)
        A = make_banded_csr({-1: d1, 0: d0, 1: d1}, n)
        b = torch.randn(n, dtype=torch.float64)

        _ = solve_banded(A, b, 1, 1)
        reps = 10 if n <= 500 else 3

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = solve_banded(A, b, 1, 1)
        t = (time.perf_counter() - t0) / reps
        per_n.append(t / n)

    ratio = max(per_n) / min(per_n)
    print("time_over_n=", dict(zip(ns, per_n)))
    assert ratio < 8.0


def test_vmap_over_batch_dimension_of_b() -> None:
    """Verifies vmapping RHS vectors is equivalent to explicit looped solves with fixed A."""
    n, batch = 24, 6
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=16)
    A = make_banded_csr(diags, n)
    b = torch.randn(batch, n, dtype=torch.float64)

    out = torch.vmap(lambda rhs: solve_banded(A, rhs, 1, 1))(b)
    ref = torch.stack([solve_banded(A, b[i], 1, 1) for i in range(batch)], dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_vmap_over_batch_dimension_of_A_values() -> None:
    """Verifies vmapping over CSR values (batched A) matches looped solves with fixed b."""
    n, batch = 24, 4
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=17)
    A0 = make_banded_csr(diags, n)
    crow, col = A0.crow_indices(), A0.col_indices()

    values_batch = torch.stack([A0.values() + 0.05 * i for i in range(batch)], dim=0)
    b = torch.randn(n, dtype=torch.float64)

    out = torch.vmap(lambda v: solve_banded_csr_values(crow, col, v, b, 1, 1))(values_batch)
    ref = torch.stack(
        [solve_banded_csr_values(crow, col, values_batch[i], b, 1, 1) for i in range(batch)],
        dim=0,
    )
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_nested_vmap() -> None:
    """Verifies nested vmap composition agrees with nested explicit loops."""
    n, batch_a, batch_b = 16, 3, 5
    diags = _rand_diags(n, 1, 1, dtype=torch.float64, seed=18)
    A0 = make_banded_csr(diags, n)
    crow, col = A0.crow_indices(), A0.col_indices()

    values_batch = torch.stack([A0.values() + 0.02 * i for i in range(batch_a)], dim=0)
    b_batch = torch.randn(batch_b, n, dtype=torch.float64)

    out = torch.vmap(
        lambda v: torch.vmap(lambda rhs: solve_banded_csr_values(crow, col, v, rhs, 1, 1))(b_batch)
    )(values_batch)

    ref = torch.stack(
        [
            torch.stack(
                [
                    solve_banded_csr_values(crow, col, values_batch[i], b_batch[j], 1, 1)
                    for j in range(batch_b)
                ],
                dim=0,
            )
            for i in range(batch_a)
        ],
        dim=0,
    )

    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)
