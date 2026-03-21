import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.eigvec_adjoint import (
    _tridiag_eigvec_adjoint_dense_oracle,
    tridiag_eigvec_adjoint,
    tridiag_eigvec_adjoint_batch,
    tridiag_eigvec_cluster_reattach,
    tridiag_eigvec_reattach,
)


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


def test_structured_backend_matches_dense_oracle_d():
    N = 18
    torch.manual_seed(321)
    d = torch.randn(N, dtype=torch.float64) + 4.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.15
    phi = _get_eigvec(d, e, idx=0)
    weights = torch.randn(N, dtype=torch.float64)

    grad_d_struct, grad_e_struct = tridiag_eigvec_adjoint(phi, d, e, weights, tau=1e-8, eps=1e-10)
    grad_d_dense, grad_e_dense = _tridiag_eigvec_adjoint_dense_oracle(
        phi, d, e, weights, tau=1e-8, eps=1e-10
    )

    torch.testing.assert_close(grad_d_struct, grad_d_dense, rtol=5e-3, atol=5e-5)
    torch.testing.assert_close(grad_e_struct, grad_e_dense, rtol=5e-3, atol=5e-5)


def test_structured_backend_matches_dense_oracle_interior_mode():
    N = 24
    torch.manual_seed(777)
    d = torch.randn(N, dtype=torch.float64) + 6.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.08
    phi = _get_eigvec(d, e, idx=5)
    weights = torch.randn(N, dtype=torch.float64)

    grad_d_struct, grad_e_struct = tridiag_eigvec_adjoint(phi, d, e, weights, tau=1e-8, eps=1e-10)
    grad_d_dense, grad_e_dense = _tridiag_eigvec_adjoint_dense_oracle(
        phi, d, e, weights, tau=1e-8, eps=1e-10
    )

    torch.testing.assert_close(grad_d_struct, grad_d_dense, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(grad_e_struct, grad_e_dense, rtol=1e-2, atol=1e-4)


def test_cluster_reattach_backward_finite():
    N = 14
    d = torch.linspace(1.0, 6.0, N, dtype=torch.float64)
    d[1] = d[0] + 1e-8
    e = (torch.ones(N - 1, dtype=torch.float64) * 0.02).requires_grad_(True)

    A = _build_tridiag_dense(d.detach(), e.detach())
    _, vecs = torch.linalg.eigh(A)
    phi_cluster = vecs[:, :2].T.contiguous()

    phi_out = tridiag_eigvec_cluster_reattach(phi_cluster, d, e, eps=1e-10)
    weights = torch.randn_like(phi_out)
    loss = (phi_out * weights).sum()
    loss.backward()

    assert e.grad is not None
    assert torch.isfinite(e.grad).all()
    assert e.grad.abs().max() > 1e-12


def test_batched_api_matches_sum_of_single_mode_gradients():
    N = 20
    torch.manual_seed(2026)
    d = torch.randn(N, dtype=torch.float64) + 5.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.12
    A = _build_tridiag_dense(d, e)
    _, vecs = torch.linalg.eigh(A)

    phi_batch = vecs[:, :4].T.contiguous()
    grad_phi_batch = torch.randn_like(phi_batch)

    grad_d_batch, grad_e_batch = tridiag_eigvec_adjoint_batch(
        phi_batch, d, e, grad_phi_batch, tau=1e-8, eps=1e-10
    )

    grad_d_ref = torch.zeros_like(d)
    grad_e_ref = torch.zeros_like(e)
    for i in range(phi_batch.shape[0]):
        gd_i, ge_i = tridiag_eigvec_adjoint(
            phi_batch[i], d, e, grad_phi_batch[i], tau=1e-8, eps=1e-10
        )
        grad_d_ref += gd_i
        grad_e_ref += ge_i

    torch.testing.assert_close(grad_d_batch, grad_d_ref, rtol=5e-3, atol=5e-5)
    torch.testing.assert_close(grad_e_batch, grad_e_ref, rtol=5e-3, atol=5e-5)


def test_batched_api_handles_cluster_and_isolated_modes():
    N = 16
    d = torch.linspace(1.0, 6.0, N, dtype=torch.float64)
    d[1] = d[0] + 1e-8
    e = torch.ones(N - 1, dtype=torch.float64) * 0.02
    A = _build_tridiag_dense(d, e)
    _, vecs = torch.linalg.eigh(A)

    phi_batch = vecs[:, :4].T.contiguous()
    grad_phi_batch = torch.randn_like(phi_batch)

    grad_d, grad_e = tridiag_eigvec_adjoint_batch(
        phi_batch, d, e, grad_phi_batch, tau=0.1, eps=1e-10
    )

    assert torch.isfinite(grad_d).all()
    assert torch.isfinite(grad_e).all()
    assert grad_d.abs().max() > 1e-12
    assert grad_e.abs().max() > 1e-12
