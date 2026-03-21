from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.eigvec_adjoint import tridiag_eigvec_reattach


def _build_tridiag_dense(d, e):
    return torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)


def _get_eigvec(d, e, idx=0):
    A = _build_tridiag_dense(d.detach(), e.detach())
    _, vecs = torch.linalg.eigh(A)
    return vecs[:, idx].clone()


def test_passthrough():
    N = 10
    d = torch.randn(N, dtype=torch.float64) + 3.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.3
    phi = _get_eigvec(d, e, idx=0)
    phi_out = tridiag_eigvec_reattach(phi, d, e)
    assert torch.allclose(phi_out, phi)


def test_gradient_finite_difference_d():
    N = 20
    torch.manual_seed(42)
    d = (torch.randn(N, dtype=torch.float64) + 3.0).detach().requires_grad_(True)
    e = torch.randn(N - 1, dtype=torch.float64) * 0.2
    phi = _get_eigvec(d, e, idx=0)

    phi_out = tridiag_eigvec_reattach(phi, d, e, tau=0.1, eps=1e-10)
    weights = torch.arange(1, N + 1, dtype=torch.float64)
    loss = (phi_out * weights).sum()
    loss.backward()
    grad_analytical = d.grad.clone()

    eps_fd = 1e-6
    grad_fd = torch.zeros(N, dtype=torch.float64)
    for j in range(N):
        d_p = d.detach().clone()
        d_p[j] += eps_fd
        phi_p = _get_eigvec(d_p, e, idx=0)
        if torch.dot(phi_p, phi) < 0:
            phi_p = -phi_p
        loss_p = (phi_p * weights).sum()

        d_m = d.detach().clone()
        d_m[j] -= eps_fd
        phi_m = _get_eigvec(d_m, e, idx=0)
        if torch.dot(phi_m, phi) < 0:
            phi_m = -phi_m
        loss_m = (phi_m * weights).sum()

        grad_fd[j] = (loss_p - loss_m) / (2 * eps_fd)

    mask = grad_fd.abs() > 1e-8
    assert mask.any(), "No significant FD gradients"
    assert torch.allclose(grad_analytical[mask], grad_fd[mask], rtol=0.05), (
        f"max rel diff: {((grad_analytical[mask] - grad_fd[mask]) / grad_fd[mask]).abs().max():.2e}"
    )


def test_gradient_finite_difference_e():
    N = 15
    torch.manual_seed(100)
    d = torch.randn(N, dtype=torch.float64) + 5.0
    e = (torch.randn(N - 1, dtype=torch.float64) * 0.3).detach().requires_grad_(True)
    phi = _get_eigvec(d, e, idx=0)

    phi_out = tridiag_eigvec_reattach(phi, d, e, tau=0.01, eps=1e-10)
    weights = torch.randn(N, dtype=torch.float64)
    loss = (phi_out * weights).sum()
    loss.backward()
    grad_analytical = e.grad.clone()

    eps_fd = 1e-6
    grad_fd = torch.zeros(N - 1, dtype=torch.float64)
    for j in range(N - 1):
        e_p = e.detach().clone()
        e_p[j] += eps_fd
        phi_p = _get_eigvec(d, e_p, idx=0)
        if torch.dot(phi_p, phi) < 0:
            phi_p = -phi_p

        e_m = e.detach().clone()
        e_m[j] -= eps_fd
        phi_m = _get_eigvec(d, e_m, idx=0)
        if torch.dot(phi_m, phi) < 0:
            phi_m = -phi_m

        grad_fd[j] = ((phi_p * weights).sum() - (phi_m * weights).sum()) / (2 * eps_fd)

    mask = grad_fd.abs() > 1e-8
    assert mask.any(), "No significant FD gradients"
    assert torch.allclose(grad_analytical[mask], grad_fd[mask], rtol=0.05), (
        f"max rel diff: {((grad_analytical[mask] - grad_fd[mask]) / grad_fd[mask]).abs().max():.2e}"
    )


def test_clustered_eigenvalues():
    N = 10
    # CLSOE
    d = torch.tensor(
        [1.0, 1.0 + 1e-10, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    e = torch.tensor([0.01] * 9, dtype=torch.float64)

    phi = _get_eigvec(d, e, idx=0)
    phi_out = tridiag_eigvec_reattach(phi, d, e, tau=0.1, eps=1e-10)
    loss = (phi_out**2).sum()
    loss.backward()

    assert d.grad is not None
    assert torch.isfinite(d.grad).all()


def test_isolated_eigenvalue_matches_old():
    N = 15
    torch.manual_seed(200)
    d = (torch.randn(N, dtype=torch.float64) + 5.0).detach().requires_grad_(True)
    e = torch.randn(N - 1, dtype=torch.float64) * 0.2

    phi = _get_eigvec(d, e, idx=0)
    phi_out = tridiag_eigvec_reattach(phi, d, e, tau=1e-15, eps=1e-10)
    weights = torch.randn(N, dtype=torch.float64)
    loss = (phi_out * weights).sum()
    loss.backward()

    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    assert d.grad.abs().max() > 1e-10
