import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.eigvec_adjoint import (
    tridiag_eigvec_reattach,
    tridiag_eigvec_reattach_varying_batch,
)
from diff_utils.solve_tridiag import tridiag_inverse_iteration_batch


def _build_tridiagonal(d, e):
    return torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)


def _varying_batch_grads(phi_batch, d_batch, e_batch, grad_phi, *, tau):
    d_a = d_batch.clone().requires_grad_(True)
    e_a = e_batch.clone().requires_grad_(True)
    phi_out = tridiag_eigvec_reattach_varying_batch(phi_batch, d_a, e_a, tau=tau, eps=1e-10)
    loss = (phi_out.conj() * grad_phi).real.sum()
    loss.backward()
    return d_a.grad, e_a.grad


def test_complex_varying_batch_matches_single_mode_reference():
    torch.manual_seed(11)
    M = 3
    N = 7

    d_real = torch.randn(M, N, dtype=torch.float64) + 4.0
    d_imag = torch.randn(M, N, dtype=torch.float64) * 1e-3
    d_batch = torch.complex(d_real, d_imag)

    e_real = torch.randn(M, N - 1, dtype=torch.float64) * 0.15
    e_batch = torch.complex(e_real, torch.zeros_like(e_real))
    phi_batch = tridiag_inverse_iteration_batch(d_batch.detach(), e_batch.detach()[0])
    grad_phi = torch.randn(M, N, dtype=torch.complex128)

    grad_d, grad_e = _varying_batch_grads(phi_batch, d_batch, e_batch, grad_phi, tau=1e-8)

    grad_d_ref = torch.zeros_like(d_batch)
    grad_e_ref = torch.zeros_like(e_batch)
    for m in range(M):
        d_m = d_batch.detach()[m].clone().requires_grad_(True)
        e_m = e_batch.detach()[m].clone().requires_grad_(True)
        phi_m = tridiag_eigvec_reattach(phi_batch[m], d_m, e_m, tau=1e-8, eps=1e-10)
        loss_m = (phi_m.conj() * grad_phi[m]).real.sum()
        loss_m.backward()
        grad_d_ref[m] = d_m.grad
        grad_e_ref[m] = e_m.grad

    torch.testing.assert_close(grad_d, grad_d_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(grad_e, grad_e_ref, atol=1e-10, rtol=1e-10)


def test_shared_spectrum_cluster_handling_changes_adjoint():
    d = torch.tensor([2.0, 2.0 + 1e-8, 5.0, 6.0], dtype=torch.float64)
    e = torch.tensor([1e-6, 0.0, 0.0], dtype=torch.float64)
    A = _build_tridiagonal(d, e)
    _, vecs = torch.linalg.eigh(A)
    phi = vecs[:, :2].T.contiguous()
    weights = torch.randn_like(phi)

    d_a_small = d.clone().requires_grad_(True)
    phi_small = tridiag_eigvec_reattach(phi, d_a_small, e, tau=1e-15, eps=1e-10)
    (phi_small * weights).sum().backward()
    grad_small = d_a_small.grad.clone()

    d_a_wide = d.clone().requires_grad_(True)
    phi_wide = tridiag_eigvec_reattach(phi, d_a_wide, e, tau=1e-1, eps=1e-10)
    (phi_wide * weights).sum().backward()
    grad_wide = d_a_wide.grad.clone()

    assert not torch.allclose(grad_small, grad_wide, atol=1e-12, rtol=1e-12)


def test_varying_batch_tau_insensitive():
    """Production varying-batch path: tau has no effect because each mode has its
    own tridiag. Cluster grouping across modes with different tridiags is not applicable.
    """
    d = torch.tensor([2.0, 2.0 + 1e-8, 5.0, 6.0], dtype=torch.float64)
    e = torch.tensor([1e-6, 0.0, 0.0], dtype=torch.float64)
    A = _build_tridiagonal(d, e)
    _, vecs = torch.linalg.eigh(A)
    phi_batch = vecs[:, :2].T.contiguous()
    d_batch = torch.stack([d, d])
    e_batch = torch.stack([e, e])
    grad_phi = torch.randn_like(phi_batch)

    grad_d_small, _ = _varying_batch_grads(phi_batch, d_batch, e_batch, grad_phi, tau=1e-15)
    grad_d_wide, _ = _varying_batch_grads(phi_batch, d_batch, e_batch, grad_phi, tau=1e-1)

    assert torch.allclose(grad_d_small, grad_d_wide, atol=1e-12, rtol=1e-12)
