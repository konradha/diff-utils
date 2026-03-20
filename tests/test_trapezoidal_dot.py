from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from banded.trapezoidal_dot import trapezoidal_normalization


def _reference_interior_terms(phi, B1, B1C, rho, h, omega2):
    """Reference implementation matching pyat's acoustic_interior_terms."""
    N1 = phi.shape[0]
    rho_val = float(rho[0])
    rho_omega_h2 = rho_val * omega2 * h * h

    phi_c = phi.to(torch.complex128)
    phi_sq = phi_c * phi_c

    weights = torch.full((N1,), h, dtype=torch.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    w = weights.to(torch.complex128)

    sq_norm = torch.sum(w * phi_sq / rho_val)
    slow = torch.sum(w * (B1.to(torch.complex128) + 2.0) * phi_sq / rho_omega_h2)
    perturb = 1j * torch.sum(w * B1C.to(torch.complex128) * phi_sq / rho_val)
    return sq_norm, slow, perturb


def test_forward_matches_reference():
    torch.manual_seed(42)
    N = 50
    omega = 2 * 3.14159265 * 100.0
    c = 1500.0
    h = 2.0
    k = omega / c
    B1 = torch.full((N + 1,), -2.0 + h * h * k * k, dtype=torch.float64)
    B1C = torch.zeros(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.0, dtype=torch.float64)
    phi = torch.randn(N + 1, dtype=torch.float64)
    omega2 = omega**2

    sq, sl, pr = trapezoidal_normalization(phi, B1, B1C, rho, h, omega2)
    sq_ref, sl_ref, pr_ref = _reference_interior_terms(phi, B1, B1C, rho, h, omega2)

    assert torch.allclose(sq, sq_ref, atol=1e-14)
    assert torch.allclose(sl, sl_ref, atol=1e-14)
    assert torch.allclose(pr, pr_ref, atol=1e-14)


def test_gradcheck_phi_real():
    torch.manual_seed(100)
    N = 10
    h = 1.0
    omega2 = 100.0
    B1 = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    phi = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)

    def fn(p):
        sq, sl, pr = trapezoidal_normalization(p, B1, B1C, rho, h, omega2)
        return sq.real + sl.real + pr.real

    assert gradcheck(fn, (phi,), eps=1e-7, atol=1e-4)


def test_gradcheck_B1():
    torch.manual_seed(200)
    N = 8
    h = 1.0
    omega2 = 100.0
    phi = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)

    def fn(b1):
        sq, sl, pr = trapezoidal_normalization(phi, b1, B1C, rho, h, omega2)
        return sl.real

    assert gradcheck(fn, (B1,), eps=1e-7, atol=1e-4)


def test_gradcheck_B1C():
    torch.manual_seed(300)
    N = 8
    h = 1.0
    omega2 = 100.0
    phi = torch.randn(N + 1, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)

    def fn(b1c):
        sq, sl, pr = trapezoidal_normalization(phi, B1, b1c, rho, h, omega2)
        return pr.real

    assert gradcheck(fn, (B1C,), eps=1e-7, atol=1e-4)


def test_n1_edge_case():
    phi = torch.tensor([3.0], dtype=torch.float64)
    B1 = torch.tensor([1.0], dtype=torch.float64)
    B1C = torch.tensor([0.5], dtype=torch.float64)
    rho = torch.tensor([1.0], dtype=torch.float64)
    h = 1.0
    omega2 = 1.0

    sq, sl, pr = trapezoidal_normalization(phi, B1, B1C, rho, h, omega2)
    # Single point: both first and last, so weight = h*0.5 (half at boundary)
    # But with N1=1, index 0 == N1-1, so weights[0] *= 0.5 is applied once
    # sq_norm = 0.5 * 9.0 / 1.0 = 4.5
    # Actually for N1=1: weights[-1] is same as weights[0], so *= 0.5 applied twice = h*0.25
    # sq_norm = 0.25 * 9.0 / 1.0 = 2.25
    assert abs(sq.real.item() - 2.25) < 1e-14


def test_zero_phi_gives_zero():
    N = 10
    phi = torch.zeros(N + 1, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.0, dtype=torch.float64)

    sq, sl, pr = trapezoidal_normalization(phi, B1, B1C, rho, 1.0, 1.0)
    assert sq.abs().item() < 1e-14
    assert sl.abs().item() < 1e-14
    assert pr.abs().item() < 1e-14
