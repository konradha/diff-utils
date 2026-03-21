
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.eigenvalue_ift import eigenvalue_ift


def _simple_dispersion(x, a, b):
    return a * x * x + b * x - 1.0


def test_passthrough():
    x = torch.tensor([1.5, 2.3], dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64)
    b = torch.tensor(-2.0, dtype=torch.float64)

    x_out = eigenvalue_ift(x, _simple_dispersion, a, b)
    assert torch.allclose(x_out, x)


def test_analytic_gradient():
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)

    disc = b.detach() ** 2 + 4 * a.detach()
    x_star_val = (-b.detach() + torch.sqrt(disc)) / (2 * a.detach())
    x_converged = x_star_val.unsqueeze(0)

    x_out = eigenvalue_ift(x_converged, _simple_dispersion, a, b)
    x_out.sum().backward()

    denom = 2 * a.detach() * x_star_val + b.detach()
    expected_da = -(x_star_val**2) / denom
    expected_db = -x_star_val / denom

    assert abs(a.grad.item() - expected_da.item()) < 1e-8, (
        f"da: got {a.grad.item()}, expected {expected_da.item()}"
    )
    assert abs(b.grad.item() - expected_db.item()) < 1e-8, (
        f"db: got {b.grad.item()}, expected {expected_db.item()}"
    )


def test_finite_difference_validation():
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)

    disc = b.detach() ** 2 + 4 * a.detach()
    x_star = (-b.detach() + torch.sqrt(disc)) / (2 * a.detach())
    x_converged = x_star.unsqueeze(0)

    x_out = eigenvalue_ift(x_converged, _simple_dispersion, a, b)
    x_out.sum().backward()
    grad_a_ift = a.grad.item()

    eps = 1e-7
    for sign in [1, -1]:
        a_pert = a.detach() + sign * eps
        disc_pert = b.detach() ** 2 + 4 * a_pert
        x_pert = (-b.detach() + torch.sqrt(disc_pert)) / (2 * a_pert)

    a_p = a.detach() + eps
    disc_p = b.detach() ** 2 + 4 * a_p
    x_p = (-b.detach() + torch.sqrt(disc_p)) / (2 * a_p)

    a_m = a.detach() - eps
    disc_m = b.detach() ** 2 + 4 * a_m
    x_m = (-b.detach() + torch.sqrt(disc_m)) / (2 * a_m)

    fd_grad = (x_p - x_m) / (2 * eps)

    assert abs(grad_a_ift - fd_grad.item()) < 1e-3, (
        f"IFT grad={grad_a_ift}, FD grad={fd_grad.item()}"
    )


def test_batched_accumulation():
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-5.0, dtype=torch.float64, requires_grad=True)

    disc = b.detach() ** 2 + 4 * a.detach()
    root1 = (-b.detach() + torch.sqrt(disc)) / (2 * a.detach())
    root2 = (-b.detach() - torch.sqrt(disc)) / (2 * a.detach())
    x_converged = torch.stack([root1, root2])

    x_out = eigenvalue_ift(x_converged, _simple_dispersion, a, b)
    x_out.sum().backward()

    assert a.grad is not None
    assert b.grad is not None
    assert torch.isfinite(a.grad)
    assert torch.isfinite(b.grad)


def test_complex_eigenvalue():
    def complex_disp(x, c):
        return c * x * x - 1.0

    c = torch.tensor(1.0 + 0.1j, dtype=torch.complex128, requires_grad=True)
    x_star = 1.0 / torch.sqrt(c.detach())
    x_converged = x_star.unsqueeze(0)

    x_out = eigenvalue_ift(x_converged, complex_disp, c)
    loss = (x_out.real**2 + x_out.imag**2).sum()
    loss.backward()

    assert c.grad is not None
    assert torch.isfinite(c.grad.real)
    assert torch.isfinite(c.grad.imag)
