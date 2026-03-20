from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.trapezoidal_dot import trapezoidal_normalization


def _reference_interior_terms(phi, B1, B1C, rho, h, omega2):
    N1 = phi.shape[0]
    rho_val = float(rho[0])
    rho_omega_h2 = rho_val * omega2 * h * h
    phi_c = phi.to(torch.complex128)
    phi_sq = phi_c * phi_c
    weights = torch.full((N1,), h, dtype=torch.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    w = weights.to(torch.complex128)
    return (
        torch.sum(w * phi_sq / rho_val),
        torch.sum(w * (B1.to(torch.complex128) + 2.0) * phi_sq / rho_omega_h2),
        1j * torch.sum(w * B1C.to(torch.complex128) * phi_sq / rho_val),
    )


def test_forward_matches_reference():
    torch.manual_seed(42)
    N, h = 50, 2.0
    omega = 2 * 3.14159265 * 100.0
    k = omega / 1500.0
    B1 = torch.full((N + 1,), -2.0 + h * h * k * k, dtype=torch.float64)
    B1C = torch.zeros(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.0, dtype=torch.float64)
    phi = torch.randn(N + 1, dtype=torch.float64)

    sq, sl, pr = trapezoidal_normalization(phi, B1, B1C, rho, h, omega**2)
    sq_ref, sl_ref, pr_ref = _reference_interior_terms(phi, B1, B1C, rho, h, omega**2)
    assert torch.allclose(sq, sq_ref, atol=1e-14)
    assert torch.allclose(sl, sl_ref, atol=1e-14)
    assert torch.allclose(pr, pr_ref, atol=1e-14)


def test_gradcheck_phi_real():
    torch.manual_seed(100)
    N = 10
    B1 = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    phi = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)
    fn = lambda p: sum(x.real for x in trapezoidal_normalization(p, B1, B1C, rho, 1.0, 100.0))
    assert gradcheck(fn, (phi,), eps=1e-7, atol=1e-4)


def test_gradcheck_B1():
    torch.manual_seed(200)
    N = 8
    phi = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)
    assert gradcheck(
        lambda b1: trapezoidal_normalization(phi, b1, B1C, rho, 1.0, 100.0)[1].real,
        (B1,),
        eps=1e-7,
        atol=1e-4,
    )


def test_gradcheck_B1C():
    torch.manual_seed(300)
    N = 8
    phi = torch.randn(N + 1, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)
    assert gradcheck(
        lambda b1c: trapezoidal_normalization(phi, B1, b1c, rho, 1.0, 100.0)[2].real,
        (B1C,),
        eps=1e-7,
        atol=1e-4,
    )


def test_n1_edge_case():
    sq, _, _ = trapezoidal_normalization(
        torch.tensor([3.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        1.0,
        1.0,
    )
    assert abs(sq.real.item() - 2.25) < 1e-14


def test_zero_phi_gives_zero():
    N = 10
    sq, sl, pr = trapezoidal_normalization(
        torch.zeros(N + 1, dtype=torch.float64),
        torch.randn(N + 1, dtype=torch.float64),
        torch.randn(N + 1, dtype=torch.float64),
        torch.ones(N + 1, dtype=torch.float64),
        1.0,
        1.0,
    )
    assert all(x.abs().item() < 1e-14 for x in (sq, sl, pr))
