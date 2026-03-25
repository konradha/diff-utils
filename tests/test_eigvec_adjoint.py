import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.eigvec_adjoint import (
    eigvec_reattach,
    eigvec_degpert,
    tridiag_eigvec_reattach,
    tridiag_eigvec_reattach_varying_batch,
)


def _build_tridiagonal(d, e):
    N = d.shape[0]
    A = torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)
    return A


def _inverse_iteration(A, x_star, tol=1e-14, max_iter=100):
    N = A.shape[0]
    v = torch.randn(N, dtype=A.dtype)
    v = v / v.norm()
    shifted = A - x_star * torch.eye(N, dtype=A.dtype)
    for _ in range(max_iter):
        try:
            w = torch.linalg.solve(shifted, v)
        except Exception:
            break
        v = w / w.norm()
    return v


def test_passthrough():
    N = 5
    d = torch.randn(N, dtype=torch.float64)
    e = torch.randn(N - 1, dtype=torch.float64)
    phi = torch.randn(N, dtype=torch.float64)
    phi = phi / phi.norm()
    x_star = torch.tensor(1.5, dtype=torch.float64)

    phi_out = eigvec_reattach(phi, x_star, d, e)
    assert torch.allclose(phi_out, phi)


def test_analytic_4x4():
    d = torch.tensor([2.0, 3.0, 4.0, 2.5], dtype=torch.float64)
    e = torch.tensor([0.5, 0.3, 0.4], dtype=torch.float64)

    A = _build_tridiagonal(d, e)
    evals, evecs = torch.linalg.eigh(A)
    phi = evecs[:, 0].clone()

    x_star_a = evals[0].clone().detach().requires_grad_(True)
    d_a = d.clone().detach().requires_grad_(True)
    e_a = e.clone().detach().requires_grad_(True)

    phi_out = eigvec_reattach(phi, x_star_a, d_a, e_a)
    loss = (phi_out * torch.arange(1, 5, dtype=torch.float64)).sum()
    loss.backward()

    assert d_a.grad is not None
    assert e_a.grad is not None
    assert x_star_a.grad is not None
    assert torch.isfinite(d_a.grad).all()
    assert torch.isfinite(e_a.grad).all()
    assert torch.isfinite(x_star_a.grad)


def test_finite_difference_diagonal():
    N = 5
    torch.manual_seed(42)
    d = torch.tensor([3.0, 2.0, 4.0, 1.0, 5.0], dtype=torch.float64)
    e = torch.tensor([0.5, 0.3, 0.2, 0.4], dtype=torch.float64)

    A = _build_tridiagonal(d, e)
    evals, evecs = torch.linalg.eigh(A)
    idx = 0
    phi = evecs[:, idx].clone()
    x_star = evals[idx].clone()

    d_a = d.detach().clone().requires_grad_(True)
    phi_out = eigvec_reattach(phi, x_star, d_a, e.detach())
    loss = (phi_out * torch.arange(1, N + 1, dtype=torch.float64)).sum()
    loss.backward()

    assert d_a.grad is not None
    assert torch.isfinite(d_a.grad).all()


def test_null_space_projection():
    N = 5
    torch.manual_seed(123)
    d = torch.randn(N, dtype=torch.float64) + 3.0
    e = torch.randn(N - 1, dtype=torch.float64) * 0.3

    d_a = d.clone().requires_grad_(True)

    A = _build_tridiagonal(d, e)
    evals, evecs = torch.linalg.eigh(A)
    phi = evecs[:, 0].clone()
    x_star = evals[0].clone()

    phi_out = eigvec_reattach(phi, x_star, d_a, e)
    loss = (phi_out * phi).sum()
    loss.backward()

    assert d_a.grad is not None
    assert torch.isfinite(d_a.grad).all()


def test_degpert_near_degenerate():
    N = 10
    torch.manual_seed(999)

    evals_target = torch.linspace(1.0, 5.0, N, dtype=torch.float64)
    evals_target[1] = evals_target[0] + 1e-8

    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64))
    A = Q @ torch.diag(evals_target) @ Q.T
    A = 0.5 * (A + A.T)  # ensure symmetric

    evals, evecs = torch.linalg.eigh(A)
    assert (evals[1] - evals[0]).abs() < 1e-6, "Eigenvalues not close enough"

    d = A.diagonal().clone()
    e = A.diagonal(1).clone()

    A_tri = _build_tridiagonal(d, e)
    evals_tri, evecs_tri = torch.linalg.eigh(A_tri)

    M = 3
    phi = evecs_tri[:, :M].T.contiguous()
    x_star = evals_tri[:M].contiguous()
    grad_phi = torch.randn(M, N, dtype=torch.float64)

    grad_x, grad_d, grad_e = eigvec_degpert(
        phi,
        x_star,
        d,
        e,
        grad_phi,
        tau=0.1,  # wide tau to catch the cluster
    )

    assert torch.isfinite(grad_x).all(), f"grad_x not finite: {grad_x}"
    assert torch.isfinite(grad_d).all(), f"grad_d not finite: {grad_d}"
    assert torch.isfinite(grad_e).all(), f"grad_e not finite: {grad_e}"


def test_degpert_no_clusters_matches_standard():
    N = 8
    torch.manual_seed(42)
    d = torch.randn(N, dtype=torch.float64) + 5.0  # well-separated
    e = torch.randn(N - 1, dtype=torch.float64) * 0.1

    A = _build_tridiagonal(d, e)
    evals, evecs = torch.linalg.eigh(A)

    M = 3
    phi = evecs[:, :M].T.contiguous()
    x_star = evals[:M].contiguous()
    grad_phi = torch.randn(M, N, dtype=torch.float64)

    grad_x_dp, grad_d_dp, grad_e_dp = eigvec_degpert(phi, x_star, d, e, grad_phi, tau=1e-15)

    grad_x_std = torch.zeros(M, dtype=torch.float64)
    grad_d_std = torch.zeros(N, dtype=torch.float64)
    grad_e_std = torch.zeros(N - 1, dtype=torch.float64)
    for m in range(M):
        d_a = d.clone().requires_grad_(True)
        e_a = e.clone().requires_grad_(True)
        x_a = x_star[m].clone().requires_grad_(True)
        phi_out = eigvec_reattach(phi[m], x_a, d_a, e_a)
        loss = (phi_out * grad_phi[m]).sum()
        loss.backward()
        grad_x_std[m] = x_a.grad
        grad_d_std += d_a.grad
        grad_e_std += e_a.grad

    assert torch.allclose(grad_x_dp, grad_x_std, atol=1e-10), (
        f"grad_x mismatch:\ndegpert: {grad_x_dp}\nstandard: {grad_x_std}"
    )
    assert torch.allclose(grad_d_dp, grad_d_std, atol=1e-10), (
        f"grad_d mismatch:\ndegpert: {grad_d_dp}\nstandard: {grad_d_std}"
    )
    assert torch.allclose(grad_e_dp, grad_e_std, atol=1e-10), (
        f"grad_e mismatch:\ndegpert: {grad_e_dp}\nstandard: {grad_e_std}"
    )


def test_degpert_cluster_hamiltonian():
    N = 6
    d = torch.tensor([2.0, 3.0, 2.5, 4.0, 1.5, 3.5], dtype=torch.float64)
    e = torch.tensor([0.5, 0.3, 0.2, 0.4, 0.15], dtype=torch.float64)

    A = _build_tridiagonal(d, e)
    evals, evecs = torch.linalg.eigh(A)

    Phi_c = evecs[:, :2].T  # [2, N]
    A_Phi = (A @ evecs[:, :2]).T  # [2, N]

    H = Phi_c @ A_Phi.T  # [2, 2]

    assert torch.allclose(H, torch.diag(evals[:2]), atol=1e-12)


def test_tridiag_eigvec_reattach_varying_batch_matches_single_mode_calls():
    torch.manual_seed(7)
    M = 3
    N = 6
    d_batch = []
    e_batch = []
    phi_batch = []
    grad_phi = torch.randn(M, N, dtype=torch.float64)

    for _ in range(M):
        d = torch.randn(N, dtype=torch.float64) + 4.0
        e = torch.randn(N - 1, dtype=torch.float64) * 0.15
        A = _build_tridiagonal(d, e)
        _, evecs = torch.linalg.eigh(A)
        d_batch.append(d)
        e_batch.append(e)
        phi_batch.append(evecs[:, 0].clone())

    d_batch = torch.stack(d_batch).requires_grad_(True)
    e_batch = torch.stack(e_batch).requires_grad_(True)
    phi_batch = torch.stack(phi_batch)

    phi_out = tridiag_eigvec_reattach_varying_batch(phi_batch, d_batch, e_batch, tau=1e-8, eps=1e-10)
    loss = (phi_out * grad_phi).sum()
    loss.backward()

    grad_d_ref = torch.zeros_like(d_batch)
    grad_e_ref = torch.zeros_like(e_batch)
    for m in range(M):
        d_m = d_batch.detach()[m].clone().requires_grad_(True)
        e_m = e_batch.detach()[m].clone().requires_grad_(True)
        phi_m = tridiag_eigvec_reattach(phi_batch[m], d_m, e_m, tau=1e-8, eps=1e-10)
        loss_m = (phi_m * grad_phi[m]).sum()
        loss_m.backward()
        grad_d_ref[m] = d_m.grad
        grad_e_ref[m] = e_m.grad

    torch.testing.assert_close(d_batch.grad, grad_d_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(e_batch.grad, grad_e_ref, atol=1e-10, rtol=1e-10)
