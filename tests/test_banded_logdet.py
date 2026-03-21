import os
import time
import tracemalloc

import numpy as np
import pytest
import torch
from scipy.linalg.lapack import dgbtrf
from torch.autograd import gradcheck

from diff_utils import banded_logdet, dense_to_lapack_band, lapack_band_to_dense


def _random_banded_dense(n: int, kl: int, ku: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        lo = max(0, i - kl)
        hi = min(n - 1, i + ku)
        for j in range(lo, hi + 1):
            if i != j:
                A[i, j] = 0.08 * torch.randn((), generator=g, dtype=torch.float64)
    row_abs = A.abs().sum(dim=1)
    A[torch.arange(n), torch.arange(n)] = row_abs + 1.5
    return A


def _raw_lapack_logdet(A_band: torch.Tensor, kl: int, ku: int) -> tuple[float, float]:
    ab = np.array(A_band.detach().numpy(), dtype=np.float64, order="F", copy=True)
    if kl > 0:
        ab[:kl, :] = 0.0
    lu, ipiv, info = dgbtrf(ab, kl, ku, overwrite_ab=1)
    if info < 0:
        raise RuntimeError(f"dgbtrf invalid argument {-info}")
    main = kl + ku
    udiag = lu[main, :]
    if info > 0 or np.any(udiag == 0.0):
        return 0.0, -np.inf
    swaps = int(np.count_nonzero(ipiv != np.arange(ipiv.size, dtype=ipiv.dtype)))
    perm_sign = -1.0 if (swaps & 1) else 1.0
    return perm_sign * float(np.prod(np.sign(udiag))), float(np.log(np.abs(udiag)).sum())


def _assert_slogdet_match(A_band: torch.Tensor, kl: int, ku: int, rtol: float = 1e-12):
    sign, logabs = banded_logdet(A_band, kl, ku)
    dense = lapack_band_to_dense(A_band, kl, ku)
    sign_ref, logabs_ref = torch.linalg.slogdet(dense)
    assert float(sign.item()) == float(sign_ref.item())
    if torch.isfinite(logabs_ref):
        assert abs(float(logabs.item()) - float(logabs_ref.item())) <= rtol * max(
            1.0, abs(float(logabs_ref.item()))
        )
    else:
        assert float(logabs.item()) == float(logabs_ref.item())


@pytest.mark.parametrize("n", [5, 10, 20, 50])
@pytest.mark.parametrize("kl,ku", [(1, 1), (2, 2), (4, 4), (2, 3), (3, 1)])
def test_dense_comparison_random_banded(n: int, kl: int, ku: int) -> None:
    A_dense = _random_banded_dense(n, kl, ku, seed=11 + 7 * n + 3 * kl + ku)
    A_band = dense_to_lapack_band(A_dense, kl, ku)
    _assert_slogdet_match(A_band, kl, ku)


@pytest.mark.parametrize("n", [5, 8, 12])
@pytest.mark.parametrize("kl,ku", [(1, 1), (2, 2), (4, 4), (1, 3), (3, 1)])
def test_gradcheck_band_storage(n: int, kl: int, ku: int) -> None:
    A_dense = _random_banded_dense(n, kl, ku, seed=100 + 13 * n + kl + ku)
    A_band = dense_to_lapack_band(A_dense, kl, ku).requires_grad_(True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return banded_logdet(x, kl, ku)[1]

    assert gradcheck(fn, (A_band,), eps=1e-6, atol=1e-5, rtol=1e-5)


def test_near_singular_forward_and_backward_stability() -> None:
    n, kl, ku = 24, 2, 2
    A = _random_banded_dense(n, kl, ku, seed=203)
    A[7, 7] = 1e-15
    A_band = dense_to_lapack_band(A, kl, ku).requires_grad_(True)
    sign, logabs = banded_logdet(A_band, kl, ku)
    assert sign.item() in (-1.0, 1.0, 0.0)
    assert torch.isfinite(logabs)
    grad = torch.autograd.grad(logabs, A_band)[0]
    assert torch.isfinite(grad).all()


def test_known_identity_determinant() -> None:
    n, kl, ku = 12, 4, 4
    A = torch.eye(n, dtype=torch.float64)
    A_band = dense_to_lapack_band(A, kl, ku)
    sign, logabs = banded_logdet(A_band, kl, ku)
    assert sign.item() == 1.0
    assert logabs.item() == 0.0


def test_known_diagonal_determinant() -> None:
    diag = torch.tensor([2.0, -3.0, 0.5, -4.0, 1.5], dtype=torch.float64)
    A = torch.diag(diag)
    A_band = dense_to_lapack_band(A, 0, 0)
    sign, logabs = banded_logdet(A_band, 0, 0)
    expected_sign = float(torch.sign(diag).prod().item())
    expected_log = float(torch.log(diag.abs()).sum().item())
    assert sign.item() == expected_sign
    assert abs(logabs.item() - expected_log) <= 1e-12


def test_negative_determinant_sign() -> None:
    A = torch.diag(torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64))
    A_band = dense_to_lapack_band(A, 0, 0)
    sign, logabs = banded_logdet(A_band, 0, 0)
    assert sign.item() == -1.0
    assert abs(logabs.item() - float(np.log(24.0))) <= 1e-12


def test_permutation_stress_sign_matches_dense() -> None:
    n = 40
    A = torch.zeros((n, n), dtype=torch.float64)
    A[torch.arange(n), torch.arange(n)] = 1e-8
    A[torch.arange(1, n), torch.arange(0, n - 1)] = 2.0
    A[torch.arange(0, n - 1), torch.arange(1, n)] = 0.5
    A_band = dense_to_lapack_band(A, 1, 1)
    _assert_slogdet_match(A_band, 1, 1)


def test_zero_matrix_returns_zero_sign_negative_infinity_and_zero_grad() -> None:
    n, kl, ku = 16, 2, 3
    A_band = torch.zeros((2 * kl + ku + 1, n), dtype=torch.float64, requires_grad=True)
    sign, logabs = banded_logdet(A_band, kl, ku)
    assert sign.item() == 0.0
    assert logabs.item() == -float("inf")
    grad = torch.autograd.grad(logabs, A_band, allow_unused=False)[0]
    assert torch.all(grad == 0)


def test_huge_condition_number_forward_matches_dense() -> None:
    n = 20
    diag = torch.logspace(-8, 8, n, dtype=torch.float64)
    A = torch.diag(diag)
    A_band = dense_to_lapack_band(A, 0, 0)
    _assert_slogdet_match(A_band, 0, 0, rtol=1e-10)


def test_n1_degenerate_case() -> None:
    A = torch.tensor([[3.5]], dtype=torch.float64)
    A_band = dense_to_lapack_band(A, 1, 1)
    sign, logabs = banded_logdet(A_band, 1, 1)
    assert sign.item() == 1.0
    assert abs(logabs.item() - np.log(3.5)) <= 1e-12


def test_bandwidth_exceeds_matrix_size_is_handled() -> None:
    A = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float64)
    A_band = dense_to_lapack_band(A, 4, 4)
    _assert_slogdet_match(A_band, 4, 4)


@pytest.mark.performance
def test_forward_benchmark_wrapper_overhead() -> None:
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    torch.set_num_threads(1)
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    kl = ku = 4
    overhead = {}

    for n in sizes:
        A = _random_banded_dense(n, kl, ku, seed=500 + n)
        A_band = dense_to_lapack_band(A, kl, ku)
        reps = 200 if n <= 1000 else (60 if n <= 10000 else 10)

        for _ in range(3):
            _ = banded_logdet(A_band, kl, ku)
            _ = _raw_lapack_logdet(A_band, kl, ku)

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = banded_logdet(A_band, kl, ku)
        t_wrap = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = _raw_lapack_logdet(A_band, kl, ku)
        t_raw = (time.perf_counter() - t0) / reps

        overhead[n] = t_wrap / t_raw

    print("forward_overhead_vs_raw=", overhead)
    assert overhead[1000] < 3.0


@pytest.mark.performance
def test_backward_benchmark_and_crossover() -> None:
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    torch.set_num_threads(1)
    sizes = [100, 500, 1000, 5000]
    kl = ku = 4
    ratio = {}

    for n in sizes:
        A = _random_banded_dense(n, kl, ku, seed=700 + n)
        A_band = dense_to_lapack_band(A, kl, ku).requires_grad_(True)
        reps = 50 if n <= 1000 else 8

        t0 = time.perf_counter()
        for _ in range(reps):
            _, logabs = banded_logdet(A_band, kl, ku)
        t_fwd = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            A_band.grad = None
            _, logabs = banded_logdet(A_band, kl, ku)
            logabs.backward()
        t_bwd = (time.perf_counter() - t0) / reps
        ratio[n] = t_bwd / t_fwd

    print("backward_over_forward_ratio=", ratio)
    assert ratio[5000] > ratio[500]


@pytest.mark.performance
def test_memory_benchmark_backward_peak() -> None:
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    n = 10000
    kl = ku = 4
    A = _random_banded_dense(n, kl, ku, seed=901)
    A_band = dense_to_lapack_band(A, kl, ku).requires_grad_(True)

    tracemalloc.start()
    _, logabs = banded_logdet(A_band, kl, ku)
    logabs.backward()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("python_peak_memory_bytes=", peak)
    assert peak > 0


@pytest.mark.performance
def test_allocation_profiling_forward_backward() -> None:
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    n = 500
    kl = ku = 4
    A = _random_banded_dense(n, kl, ku, seed=1001)
    A_band = dense_to_lapack_band(A, kl, ku).requires_grad_(True)

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    _, logabs = banded_logdet(A_band, kl, ku)
    logabs.backward()
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap1.compare_to(snap0, "lineno")
    alloc_count = sum(s.count_diff for s in stats if s.count_diff > 0)
    print("python_allocation_count_delta=", alloc_count)
    assert alloc_count >= 0


@pytest.mark.performance
def test_repeated_forward_calls_inner_loop_overhead() -> None:
    if os.getenv("RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERF_TESTS=1 to run performance tests")

    torch.set_num_threads(1)
    n = 500
    kl = ku = 4
    A = _random_banded_dense(n, kl, ku, seed=1201)
    A_band = dense_to_lapack_band(A, kl, ku)

    reps = 10000
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = banded_logdet(A_band, kl, ku)
    elapsed = time.perf_counter() - t0
    per_call_us = elapsed / reps * 1e6
    print("forward_per_call_us=", per_call_us)
    assert per_call_us < 250.0


def _random_complex_banded_dense(n: int, kl: int, ku: int, seed: int) -> torch.Tensor:
    from _test_helpers import random_banded_dense

    return random_banded_dense(n, kl, ku, seed, dtype=torch.complex128)


def _assert_complex_slogdet_match(A_band: torch.Tensor, kl: int, ku: int, rtol: float = 1e-12):
    sign, logabs = banded_logdet(A_band, kl, ku)
    dense = lapack_band_to_dense(A_band, kl, ku)
    sign_ref, logabs_ref = torch.linalg.slogdet(dense)
    if logabs_ref.item() == float("-inf"):
        assert logabs.item() == float("-inf")
    else:
        assert abs(logabs.item() - logabs_ref.item()) <= rtol * max(1.0, abs(logabs_ref.item()))
    if sign_ref.abs().item() > 0:
        assert abs(sign.item() - sign_ref.item()) < 1e-10


@pytest.mark.parametrize("n", [5, 10, 20, 50])
@pytest.mark.parametrize("kl,ku", [(1, 1), (2, 2), (2, 3)])
def test_complex_forward_matches_dense_slogdet(n: int, kl: int, ku: int) -> None:
    A_dense = _random_complex_banded_dense(n, kl, ku, seed=2000 + 7 * n + 3 * kl + ku)
    A_band = dense_to_lapack_band(A_dense, kl, ku)
    _assert_complex_slogdet_match(A_band, kl, ku)


@pytest.mark.parametrize("n", [5, 8])
@pytest.mark.parametrize("kl,ku", [(1, 1), (2, 2), (1, 3)])
def test_complex_gradcheck(n: int, kl: int, ku: int) -> None:
    A_dense = _random_complex_banded_dense(n, kl, ku, seed=3000 + 13 * n + kl + ku)
    A_band = dense_to_lapack_band(A_dense, kl, ku).requires_grad_(True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return banded_logdet(x, kl, ku)[1]

    assert gradcheck(fn, (A_band,), eps=1e-6, atol=1e-5, rtol=1e-5)


def test_complex_hermitian_pd_gives_real_positive_det() -> None:
    n = 8
    g = torch.Generator().manual_seed(4000)
    R = torch.randn(n, n, dtype=torch.float64, generator=g) * 0.1
    I = torch.randn(n, n, dtype=torch.float64, generator=g) * 0.1
    Z = (R + 1j * I).to(torch.complex128)
    A = Z @ Z.conj().T + 3.0 * torch.eye(n, dtype=torch.complex128)
    kl = ku = n - 1
    A_band = dense_to_lapack_band(A, kl, ku)
    sign, logabs = banded_logdet(A_band, kl, ku)
    assert sign.imag.abs().item() < 1e-10
    assert sign.real.item() > 0

    sign_ref, logabs_ref = torch.linalg.slogdet(A)
    assert abs(logabs.item() - logabs_ref.item()) < 1e-8


def test_complex_singular_gives_zero_sign() -> None:
    n, kl, ku = 8, 2, 2
    A_band = torch.zeros((2 * kl + ku + 1, n), dtype=torch.complex128, requires_grad=True)
    sign, logabs = banded_logdet(A_band, kl, ku)
    assert sign.item() == 0.0
    assert logabs.item() == float("-inf")


def test_complex_lapack_importable() -> None:
    from scipy.linalg.lapack import zgbtrf, zgbtrs

    assert zgbtrf is not None
    assert zgbtrs is not None


def test_ext_caching() -> None:
    from diff_utils._ext import _cpu_ext

    ext1 = _cpu_ext()
    ext2 = _cpu_ext()
    assert ext1 is ext2
