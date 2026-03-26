import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.solve_tridiag import (
    solve_tridiag,
    solve_tridiag_batch,
    tridiag_inverse_iteration,
    tridiag_inverse_iteration_batch,
)


def test_solve_tridiag_identity():
    N = 100
    d = torch.ones(N, dtype=torch.float64)
    e = torch.zeros(N - 1, dtype=torch.float64)
    b = torch.randn(N, dtype=torch.float64)
    x = solve_tridiag(e, d, e, b)
    assert torch.allclose(x, b, atol=1e-14)


def test_solve_tridiag_small():
    d = torch.tensor([2.0, 3.0, 2.0], dtype=torch.float64)
    dl = torch.tensor([1.0, 1.0], dtype=torch.float64)
    du = torch.tensor([1.0, 1.0], dtype=torch.float64)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    x = solve_tridiag(dl, d, du, b)

    Ax = torch.zeros(3, dtype=torch.float64)
    Ax[0] = d[0] * x[0] + du[0] * x[1]
    Ax[1] = dl[0] * x[0] + d[1] * x[1] + du[1] * x[2]
    Ax[2] = dl[1] * x[1] + d[2] * x[2]
    assert torch.allclose(Ax, b, atol=1e-14)


def test_solve_tridiag_large_dd():
    N = 10000
    d = torch.randn(N, dtype=torch.float64) + 5.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.3
    b = torch.randn(N, dtype=torch.float64)
    x = solve_tridiag(e, d, e, b)

    Ax = d * x
    Ax[:-1] += e * x[1:]
    Ax[1:] += e * x[:-1]
    assert torch.allclose(Ax, b, atol=1e-10)


def test_solve_tridiag_matches_scipy():
    from scipy.linalg import solve_banded as scipy_solve
    import numpy as np

    N = 500
    d = torch.randn(N, dtype=torch.float64) + 3.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.5
    b = torch.randn(N, dtype=torch.float64)

    x_ours = solve_tridiag(e, d, e, b)

    ab = np.zeros((3, N))
    ab[0, 1:] = e.numpy()
    ab[1, :] = d.numpy()
    ab[2, :-1] = e.numpy()
    x_scipy = scipy_solve((1, 1), ab, b.numpy())

    assert torch.allclose(x_ours, torch.from_numpy(x_scipy), atol=1e-12)


def test_solve_tridiag_asymmetric():
    N = 50
    dl = torch.randn(N - 1, dtype=torch.float64) * 0.5
    d = torch.randn(N, dtype=torch.float64) + 4.0
    du = torch.randn(N - 1, dtype=torch.float64) * 0.5
    b = torch.randn(N, dtype=torch.float64)
    x = solve_tridiag(dl, d, du, b)

    Ax = d * x
    Ax[:-1] += du * x[1:]
    Ax[1:] += dl * x[:-1]
    assert torch.allclose(Ax, b, atol=1e-10)


def test_solve_tridiag_batch_matches_single():
    N = 200
    M = 16
    e = torch.randn(N - 1, dtype=torch.float64) * 0.3
    d_batch = torch.randn(M, N, dtype=torch.float64) + 5.0
    b_batch = torch.randn(M, N, dtype=torch.float64)

    x_batch = solve_tridiag_batch(e, d_batch, e, b_batch)

    for m in range(M):
        x_single = solve_tridiag(e, d_batch[m], e, b_batch[m])
        assert torch.allclose(x_batch[m], x_single, atol=1e-13), f"mode {m}"


def test_solve_tridiag_batch_complex_matches_single():
    N = 128
    M = 12
    e = (torch.randn(N - 1, dtype=torch.float64) * 0.1).to(torch.complex128)
    e = e + 1j * (torch.randn(N - 1, dtype=torch.float64) * 0.1)
    d_batch = (torch.randn(M, N, dtype=torch.float64) + 5.0).to(torch.complex128)
    d_batch = d_batch + 1j * (torch.randn(M, N, dtype=torch.float64) * 0.05)
    b_batch = torch.randn(M, N, dtype=torch.float64).to(torch.complex128)
    b_batch = b_batch + 1j * torch.randn(M, N, dtype=torch.float64)

    x_batch = solve_tridiag_batch(e, d_batch, e, b_batch)

    for m in range(M):
        x_single = solve_tridiag(e, d_batch[m], e, b_batch[m])
        assert torch.allclose(x_batch[m], x_single, atol=1e-12), f"mode {m}"


def test_solve_tridiag_batch_large():
    N = 7501
    M = 192
    e = torch.randn(N - 1, dtype=torch.float64) * 0.1
    d_batch = torch.randn(M, N, dtype=torch.float64) + 10.0
    b_batch = torch.randn(M, N, dtype=torch.float64)

    import time

    t0 = time.perf_counter()
    x_batch = solve_tridiag_batch(e, d_batch, e, b_batch)
    t1 = time.perf_counter()

    for m in [0, M - 1]:
        Ax = d_batch[m] * x_batch[m]
        Ax[:-1] += e * x_batch[m, 1:]
        Ax[1:] += e * x_batch[m, :-1]
        assert torch.allclose(Ax, b_batch[m], atol=1e-9)

    assert t1 - t0 < 1.0, f"batch solve too slow: {t1 - t0:.3f}s"


def test_inverse_iteration_known_eigenvalue():
    N = 50
    d = torch.full((N,), 2.0, dtype=torch.float64)
    e = torch.full((N - 1,), -1.0, dtype=torch.float64)
    import math

    lam = 2.0 - 2.0 * math.cos(math.pi / (N + 1))
    d_shifted = d - lam

    phi = tridiag_inverse_iteration(d_shifted, e)

    expected = torch.tensor(
        [math.sin((k + 1) * math.pi / (N + 1)) for k in range(N)], dtype=torch.float64
    )
    expected = expected / torch.sqrt(torch.dot(expected, expected))

    if torch.dot(phi, expected) < 0:
        phi = -phi
    assert torch.allclose(phi, expected, atol=1e-6)


def test_inverse_iteration_batch_matches_single():
    N = 100
    M = 8
    d = torch.full((N,), 2.0, dtype=torch.float64)
    e = torch.full((N - 1,), -1.0, dtype=torch.float64)

    import math

    shifts = torch.tensor(
        [2.0 - 2.0 * math.cos((m + 1) * math.pi / (N + 1)) for m in range(M)], dtype=torch.float64
    )
    d_batch = d.unsqueeze(0).expand(M, -1) - shifts.unsqueeze(1)

    phi_batch = tridiag_inverse_iteration_batch(d_batch, e)

    for m in range(M):
        phi_single = tridiag_inverse_iteration(d_batch[m], e)
        if torch.dot(phi_batch[m], phi_single) < 0:
            phi_single = -phi_single
        assert torch.allclose(phi_batch[m], phi_single, atol=1e-10), f"mode {m}"


def test_inverse_iteration_batch_arctic_scale():
    N = 7501
    M = 192
    d = torch.full((N,), 2.0, dtype=torch.float64)
    e = torch.full((N - 1,), -1.0, dtype=torch.float64)

    shifts = torch.linspace(0.01, 0.5, M, dtype=torch.float64)
    d_batch = d.unsqueeze(0).expand(M, -1) - shifts.unsqueeze(1)

    import time

    t0 = time.perf_counter()
    phi_batch = tridiag_inverse_iteration_batch(d_batch, e)
    t1 = time.perf_counter()

    assert phi_batch.shape == (M, N)
    assert torch.isfinite(phi_batch).all()
    assert t1 - t0 < 2.0, f"batch inverse iteration too slow: {t1 - t0:.3f}s"
