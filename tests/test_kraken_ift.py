import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.acoustic_recurrence import AcousticRecurrenceFn
from diff_utils.kraken_ift import kraken_eigenvalue_ift


def _pekeris_setup(n_points=50, freq=100.0, c_water=1500.0, depth=100.0):
    omega = 2.0 * 3.141592653589793 * freq
    h = depth / n_points
    h2 = h * h
    k_water = omega / c_water
    B1 = torch.full((n_points + 1,), -2.0 + h2 * k_water**2, dtype=torch.float64)
    rho = 1.0
    return B1, rho, h, omega, k_water, n_points


def _eval_delta(x_val, B1, h, rho, loc_start, loc_end):
    h2 = h * h
    h2k2 = torch.tensor([h2 * x_val], dtype=torch.float64)
    g_bot = torch.ones(1, dtype=torch.float64)
    f_bot = torch.zeros(1, dtype=torch.float64)
    p1_init = -2.0 * g_bot
    p2_init = (B1[loc_end] - h2k2) * g_bot - 2.0 * h * f_bot * rho
    f_num, g_val, _ = AcousticRecurrenceFn.apply(
        B1.detach(), h2k2, loc_start, loc_end, p1_init, p2_init
    )
    return -g_val[0].item()


def _eval_delta_bottom_bc(x_val, B1, h, rho, loc_start, loc_end, alpha, beta):
    h2 = h * h
    h2k2 = torch.tensor([h2 * x_val], dtype=torch.float64)
    g_bot = torch.tensor([1.0 + beta * x_val], dtype=torch.float64)
    f_bot = torch.tensor([alpha * x_val], dtype=torch.float64)
    p1_init = -2.0 * g_bot
    p2_init = (B1[loc_end] - h2k2) * g_bot - 2.0 * h * f_bot * rho
    _, g_val, _ = AcousticRecurrenceFn.apply(
        B1.detach(), h2k2, loc_start, loc_end, p1_init, p2_init
    )
    return -g_val[0].item()


def _find_eigenvalue(B1, h, rho, loc_start, loc_end, x_guess, n_iter=80):
    x0 = x_guess * 0.99
    x1 = x_guess * 1.01
    d0 = _eval_delta(x0, B1, h, rho, loc_start, loc_end)
    d1 = _eval_delta(x1, B1, h, rho, loc_start, loc_end)
    for _ in range(n_iter):
        if abs(d1 - d0) < 1e-30:
            break
        x2 = x1 - d1 * (x1 - x0) / (d1 - d0)
        x0, d0 = x1, d1
        x1, d1 = x2, _eval_delta(x2, B1, h, rho, loc_start, loc_end)
        if abs(d1) < 1e-14:
            break
    return x1


def _find_eigenvalue_bottom_bc(B1, h, rho, loc_start, loc_end, x_guess, alpha, beta, n_iter=80):
    x0 = 1e-4
    x1 = max(0.5, 4.0 * x_guess)
    n_scan = 400
    dx = (x1 - x0) / n_scan
    left = x0
    d0 = _eval_delta_bottom_bc(left, B1, h, rho, loc_start, loc_end, alpha, beta)
    right = left
    d1 = d0

    for i in range(1, n_scan + 1):
        right = x0 + i * dx
        d1 = _eval_delta_bottom_bc(right, B1, h, rho, loc_start, loc_end, alpha, beta)
        if d0 == 0.0 or d0 * d1 < 0.0:
            break
        left, d0 = right, d1
    else:
        raise RuntimeError("failed to bracket eigenvalue with bottom-BC x dependence")

    for _ in range(n_iter):
        x_mid = 0.5 * (left + right)
        d_mid = _eval_delta_bottom_bc(x_mid, B1, h, rho, loc_start, loc_end, alpha, beta)
        if abs(d_mid) < 1e-14 or abs(right - left) < 1e-14:
            break
        if d0 * d_mid <= 0.0:
            right, d1 = x_mid, d_mid
        else:
            left, d0 = x_mid, d_mid
    return 0.5 * (left + right)


@pytest.fixture
def pekeris():
    B1, rho, h, omega, k_water, n_points = _pekeris_setup()
    loc_start, loc_end = 0, n_points
    x_star = _find_eigenvalue(B1, h, rho, loc_start, loc_end, k_water**2 * 0.95)
    return B1, rho, h, loc_start, loc_end, x_star


def _make_bc_tensors(M, dtype=torch.float64):
    return (
        torch.ones(M, dtype=dtype),  # f_bc_top
        torch.zeros(M, dtype=dtype),  # g_bc_top
        torch.zeros(M, dtype=dtype),  # f_bc_bot
        torch.ones(M, dtype=dtype),  # g_bc_bot
        torch.zeros(M, dtype=dtype),  # dfdx_top
        torch.zeros(M, dtype=dtype),  # dgdx_top
        torch.zeros(M, dtype=dtype),  # dfdx_bot
        torch.zeros(M, dtype=dtype),  # dgdx_bot
    )


def test_passthrough(pekeris):
    B1, rho, h, loc_start, loc_end, x_star = pekeris
    x_converged = torch.tensor([x_star], dtype=torch.float64)
    bcs = _make_bc_tensors(1)
    x_out = kraken_eigenvalue_ift(x_converged, B1, rho, h, loc_start, loc_end, *bcs, eps=1e-12)
    assert torch.allclose(x_out, x_converged)


def test_gradient_finite_difference(pekeris):
    B1_orig, rho, h, loc_start, loc_end, x_star = pekeris

    B1 = B1_orig.clone().requires_grad_(True)
    x_converged = torch.tensor([x_star], dtype=torch.float64)
    bcs = _make_bc_tensors(1)

    x_out = kraken_eigenvalue_ift(x_converged, B1, rho, h, loc_start, loc_end, *bcs, eps=1e-12)
    x_out.sum().backward()
    grad_analytic = B1.grad.clone()

    eps = 1e-6
    for j in [10, 25, 40]:
        B1_p = B1_orig.clone()
        B1_p[j] += eps
        x_p = _find_eigenvalue(B1_p, h, rho, loc_start, loc_end, x_star)

        B1_m = B1_orig.clone()
        B1_m[j] -= eps
        x_m = _find_eigenvalue(B1_m, h, rho, loc_start, loc_end, x_star)

        fd = (x_p - x_m) / (2 * eps)
        assert abs(grad_analytic[j].item() - fd) < abs(fd) * 0.01 + 1e-10, (
            f"B1[{j}]: IFT={grad_analytic[j].item():.8e}, FD={fd:.8e}"
        )


def test_batched_multiple_modes():
    B1, rho, h, omega, k_water, n_points = _pekeris_setup()
    loc_start, loc_end = 0, n_points

    x1 = _find_eigenvalue(B1, h, rho, loc_start, loc_end, k_water**2 * 0.95)
    x2 = _find_eigenvalue(B1, h, rho, loc_start, loc_end, k_water**2 * 0.80)
    x_converged = torch.tensor([x1, x2], dtype=torch.float64)

    B1_a = B1.clone().requires_grad_(True)
    bcs = _make_bc_tensors(2)

    x_out = kraken_eigenvalue_ift(x_converged, B1_a, rho, h, loc_start, loc_end, *bcs, eps=1e-12)
    x_out.sum().backward()

    assert B1_a.grad is not None
    assert torch.isfinite(B1_a.grad).all()
    assert B1_a.grad[10:-10].abs().max().item() > 1e-10


def test_gradient_finite_difference_bottom_bc_x_dependence():
    B1_orig, rho, h, omega, k_water, n_points = _pekeris_setup()
    loc_start, loc_end = 0, n_points
    alpha = 0.05
    beta = 0.02
    x_star = _find_eigenvalue_bottom_bc(
        B1_orig, h, rho, loc_start, loc_end, k_water**2 * 0.95, alpha, beta
    )

    B1 = B1_orig.clone().requires_grad_(True)
    x_converged = torch.tensor([x_star], dtype=torch.float64)
    bcs = (
        torch.ones(1, dtype=torch.float64),  # f_bc_top
        torch.zeros(1, dtype=torch.float64),  # g_bc_top
        torch.tensor([alpha * x_star], dtype=torch.float64),  # f_bc_bot
        torch.tensor([1.0 + beta * x_star], dtype=torch.float64),  # g_bc_bot
        torch.zeros(1, dtype=torch.float64),  # dfdx_top
        torch.zeros(1, dtype=torch.float64),  # dgdx_top
        torch.tensor([alpha], dtype=torch.float64),  # dfdx_bot
        torch.tensor([beta], dtype=torch.float64),  # dgdx_bot
    )

    x_out = kraken_eigenvalue_ift(x_converged, B1, rho, h, loc_start, loc_end, *bcs, eps=1e-12)
    x_out.sum().backward()
    grad_analytic = B1.grad.clone()

    eps = 1e-6
    for j in [10, 25, 40]:
        B1_p = B1_orig.clone()
        B1_p[j] += eps
        x_p = _find_eigenvalue_bottom_bc(B1_p, h, rho, loc_start, loc_end, x_star, alpha, beta)

        B1_m = B1_orig.clone()
        B1_m[j] -= eps
        x_m = _find_eigenvalue_bottom_bc(B1_m, h, rho, loc_start, loc_end, x_star, alpha, beta)

        fd = (x_p - x_m) / (2 * eps)
        assert abs(grad_analytic[j].item() - fd) < abs(fd) * 0.01 + 1e-10, (
            f"B1[{j}]: IFT={grad_analytic[j].item():.8e}, FD={fd:.8e}"
        )


def test_dispersion_zero_at_eigenvalue(pekeris):
    B1, rho, h, loc_start, loc_end, x_star = pekeris
    delta = _eval_delta(x_star, B1, h, rho, loc_start, loc_end)
    assert abs(delta) < 1e-10
