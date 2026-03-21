import sys
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.tridiag_eigh import tridiag_eigh


def _dense_from_tridiag(d, e):
    N = d.shape[0]
    A = torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)
    return A


def test_forward_parity_with_dense():
    torch.manual_seed(42)
    N = 20
    d = torch.randn(N, dtype=torch.float64)
    e = torch.randn(N - 1, dtype=torch.float64) * 0.3

    sigma, Q = tridiag_eigh(d, e)
    A = _dense_from_tridiag(d, e)
    sigma_ref, Q_ref = torch.linalg.eigh(A)

    assert torch.allclose(sigma, sigma_ref, atol=1e-12)


def test_orthonormality():
    torch.manual_seed(100)
    N = 50
    d = torch.randn(N, dtype=torch.float64) + 3.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.2

    sigma, Q = tridiag_eigh(d, e)
    eye = Q.T @ Q
    assert torch.allclose(eye, torch.eye(N, dtype=torch.float64), atol=1e-12)


def test_reconstruction():
    torch.manual_seed(200)
    N = 30
    d = torch.randn(N, dtype=torch.float64)
    e = torch.randn(N - 1, dtype=torch.float64) * 0.5

    sigma, Q = tridiag_eigh(d, e)
    A_recon = Q @ torch.diag(sigma) @ Q.T
    A_orig = _dense_from_tridiag(d, e)
    assert torch.allclose(A_recon, A_orig, atol=1e-11)


def test_gradcheck_d():
    torch.manual_seed(300)
    N = 8
    d = torch.randn(N, dtype=torch.float64, requires_grad=True)
    e = torch.randn(N - 1, dtype=torch.float64) * 0.3

    def fn(d_in):
        sigma, Q = tridiag_eigh(d_in, e, eps=1e-6)
        return sigma.sum() + (Q**2).sum()

    assert gradcheck(fn, (d,), eps=1e-7, atol=1e-4)


def test_gradcheck_e():
    torch.manual_seed(400)
    N = 8
    d = torch.randn(N, dtype=torch.float64) + 3.0
    e = torch.randn(N - 1, dtype=torch.float64, requires_grad=True) * 0.3

    def fn(e_in):
        sigma, Q = tridiag_eigh(d, e_in, eps=1e-6)
        return sigma.sum() + (Q**2).sum()

    assert gradcheck(fn, (e,), eps=1e-7, atol=1e-4)


def test_clustered_spectrum():
    N = 10
    d = torch.tensor([1.0, 1.0 + 1e-8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float64)
    e = torch.zeros(N - 1, dtype=torch.float64)

    sigma, Q = tridiag_eigh(d, e, eps=1e-6)
    assert (sigma[1] - sigma[0]).abs() < 1e-6
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(Q).all()


def test_gradcheck_clustered():
    N = 6
    d = torch.tensor([1.0, 1.0 + 1e-8, 3.0, 5.0, 7.0, 9.0], dtype=torch.float64, requires_grad=True)
    e = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01], dtype=torch.float64)

    def fn(d_in):
        sigma, Q = tridiag_eigh(d_in, e, eps=1e-4)
        return sigma.sum() + (Q**2).sum()

    assert gradcheck(fn, (d,), eps=1e-6, atol=1e-3)


def test_large_tridiag():
    torch.manual_seed(500)
    N = 200
    d = torch.randn(N, dtype=torch.float64) + 5.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.1

    sigma, Q = tridiag_eigh(d, e)
    assert sigma.shape == (N,)
    assert Q.shape == (N, N)
    assert torch.isfinite(sigma).all()
    assert torch.allclose(Q.T @ Q, torch.eye(N, dtype=torch.float64), atol=1e-10)
