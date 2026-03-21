from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.acoustic_recurrence import AcousticRecurrenceFn
from diff_utils.kraken_ift import kraken_multilayer_ift


def _pekeris_2layer(
    c1=1500.0, c2=1400.0, rho1=1.0, rho2=1.5, depth1=100.0, depth2=20.0, freq=10.0, n1=101, n2=41
):
    omega = 2.0 * math.pi * freq
    omega2 = omega * omega

    h1 = depth1 / (n1 - 1)
    h2 = depth2 / (n2 - 1)

    B1_1 = torch.full((n1,), -2.0 + h1 * h1 * omega2 / c1**2, dtype=torch.float64)
    B1_2 = torch.full((n2,), -2.0 + h2 * h2 * omega2 / c2**2, dtype=torch.float64)
    B1 = torch.cat([B1_1, B1_2])

    layer_h = torch.tensor([h1, h2], dtype=torch.float64)
    layer_rho = torch.tensor([rho1, rho2], dtype=torch.float64)
    layer_loc = torch.tensor([0, n1], dtype=torch.long)
    layer_n = torch.tensor([n1, n2], dtype=torch.long)

    return B1, layer_h, layer_rho, layer_loc, layer_n, omega2


def _halfspace_bc(x_val, omega2, c_bot=2000.0, rho_bot=2.0):
    gamma2 = x_val - omega2 / c_bot**2
    if gamma2 > 0:
        f = gamma2**0.5
        dfdx = 0.5 / f
    else:
        f = 0.0
        dfdx = 0.0
    g = rho_bot
    dgdx = 0.0
    return f, g, dfdx, dgdx


def _find_eigenvalue_2layer(
    B1, layer_h, layer_rho, layer_loc, layer_n, omega2, c_bot=2000.0, rho_bot=2.0, x_guess=None
):
    n_layers = layer_h.shape[0]

    def eval_delta(x_val):
        f, g, _, _ = _halfspace_bc(x_val, omega2, c_bot, rho_bot)

        for li in range(n_layers - 1, -1, -1):
            h = layer_h[li].item()
            rho = layer_rho[li].item()
            loc = int(layer_loc[li].item())
            n = int(layer_n[li].item())
            loc_e = loc + n - 1

            h2k2 = h * h * x_val
            p1 = -2.0 * g
            p2 = (B1[loc_e].item() - h2k2) * g - 2.0 * h * f * rho

            h2k2_t = torch.tensor([h2k2], dtype=torch.float64)
            f_num, g_val, _ = AcousticRecurrenceFn.apply(
                B1.detach(),
                h2k2_t,
                loc,
                loc_e - 1,
                torch.tensor([p1], dtype=torch.float64),
                torch.tensor([p2], dtype=torch.float64),
            )
            f = f_num[0].item() / (2.0 * h * rho)
            g = g_val[0].item()

        return -g

    if x_guess is None:
        x_guess = omega2 / 1500.0**2 * 0.95
    x0 = x_guess * 0.99
    x1 = x_guess * 1.01
    d0 = eval_delta(x0)
    d1 = eval_delta(x1)
    for _ in range(100):
        if abs(d1 - d0) < 1e-30:
            break
        x2 = x1 - d1 * (x1 - x0) / (d1 - d0)
        x0, d0 = x1, d1
        x1, d1 = x2, eval_delta(x2)
        if abs(d1) < 1e-14:
            break
    return x1


def _make_ift_args(x_vals, omega2, c_bot=2000.0, rho_bot=2.0, dtype=torch.float64):
    if not isinstance(x_vals, (list, tuple)):
        x_vals = [x_vals]
    M = len(x_vals)

    f_bot_list, g_bot_list, dfdx_bot_list, dgdx_bot_list = [], [], [], []
    for x in x_vals:
        f, g, dfdx, dgdx = _halfspace_bc(x, omega2, c_bot, rho_bot)
        f_bot_list.append(f)
        g_bot_list.append(g)
        dfdx_bot_list.append(dfdx)
        dgdx_bot_list.append(dgdx)

    f_bc_top = torch.ones(M, dtype=dtype)
    g_bc_top = torch.zeros(M, dtype=dtype)
    dfdx_top = torch.zeros(M, dtype=dtype)
    dgdx_top = torch.zeros(M, dtype=dtype)

    f_bc_bot = torch.tensor(f_bot_list, dtype=dtype)
    g_bc_bot = torch.tensor(g_bot_list, dtype=dtype)
    dfdx_bot = torch.tensor(dfdx_bot_list, dtype=dtype)
    dgdx_bot = torch.tensor(dgdx_bot_list, dtype=dtype)

    return f_bc_top, g_bc_top, f_bc_bot, g_bc_bot, dfdx_top, dgdx_top, dfdx_bot, dgdx_bot


def test_forward_passthrough():
    B1, lh, lr, ll, ln, omega2 = _pekeris_2layer()
    x_star = _find_eigenvalue_2layer(B1, lh, lr, ll, ln, omega2)
    x_conv = torch.tensor([x_star], dtype=torch.float64)

    f_top, g_top, f_bot, g_bot, dfdx_top, dgdx_top, dfdx_bot, dgdx_bot = _make_ift_args(
        x_star, omega2
    )

    x_out = kraken_multilayer_ift(
        x_conv,
        B1,
        lh,
        lr,
        ll,
        ln,
        f_top,
        g_top,
        f_bot,
        g_bot,
        dfdx_top,
        dgdx_top,
        dfdx_bot,
        dgdx_bot,
        eps=1e-12,
    )
    assert torch.allclose(x_out, x_conv)


def test_gradient_matches_fd():
    B1_orig, lh, lr, ll, ln, omega2 = _pekeris_2layer()
    x_star = _find_eigenvalue_2layer(B1_orig, lh, lr, ll, ln, omega2)
    x_conv = torch.tensor([x_star], dtype=torch.float64)

    f_top, g_top, f_bot, g_bot, dfdx_top, dgdx_top, dfdx_bot, dgdx_bot = _make_ift_args(
        x_star, omega2
    )

    B1 = B1_orig.clone().requires_grad_(True)
    x_out = kraken_multilayer_ift(
        x_conv,
        B1,
        lh,
        lr,
        ll,
        ln,
        f_top,
        g_top,
        f_bot,
        g_bot,
        dfdx_top,
        dgdx_top,
        dfdx_bot,
        dgdx_bot,
        eps=1e-12,
    )
    x_out.sum().backward()
    grad_analytical = B1.grad.clone()

    eps_fd = 1e-6
    test_indices = [10, 50, 90, 110, 130]
    for j in test_indices:
        if j >= B1_orig.shape[0]:
            continue
        B1_p = B1_orig.clone()
        B1_p[j] += eps_fd
        x_p = _find_eigenvalue_2layer(B1_p, lh, lr, ll, ln, omega2, x_guess=x_star)

        B1_m = B1_orig.clone()
        B1_m[j] -= eps_fd
        x_m = _find_eigenvalue_2layer(B1_m, lh, lr, ll, ln, omega2, x_guess=x_star)

        fd = (x_p - x_m) / (2 * eps_fd)
        ratio = grad_analytical[j].item() / fd if abs(fd) > 1e-20 else float("inf")
        assert abs(ratio - 1.0) < 0.02, f"B1[{j}]: ratio={ratio:.4f}"


def test_gradient_layer2():
    B1_orig, lh, lr, ll, ln, omega2 = _pekeris_2layer()
    x_star = _find_eigenvalue_2layer(B1_orig, lh, lr, ll, ln, omega2)
    x_conv = torch.tensor([x_star], dtype=torch.float64)

    f_top, g_top, f_bot, g_bot, dfdx_top, dgdx_top, dfdx_bot, dgdx_bot = _make_ift_args(
        x_star, omega2
    )

    B1 = B1_orig.clone().requires_grad_(True)
    x_out = kraken_multilayer_ift(
        x_conv,
        B1,
        lh,
        lr,
        ll,
        ln,
        f_top,
        g_top,
        f_bot,
        g_bot,
        dfdx_top,
        dgdx_top,
        dfdx_bot,
        dgdx_bot,
        eps=1e-12,
    )
    x_out.sum().backward()

    n1 = int(ln[0].item())
    layer2_grad = B1.grad[n1:].abs().sum().item()
    assert layer2_grad > 1e-15, f"Layer 2 gradient is zero: {layer2_grad}"


def test_multiple_modes():
    B1_orig, lh, lr, ll, ln, omega2 = _pekeris_2layer()
    x1 = _find_eigenvalue_2layer(B1_orig, lh, lr, ll, ln, omega2, x_guess=omega2 / 1500**2 * 0.95)
    x2 = _find_eigenvalue_2layer(B1_orig, lh, lr, ll, ln, omega2, x_guess=omega2 / 1500**2 * 0.80)
    x_conv = torch.tensor([x1, x2], dtype=torch.float64)

    f_top, g_top, f_bot, g_bot, dfdx_top, dgdx_top, dfdx_bot, dgdx_bot = _make_ift_args(
        [x1, x2], omega2
    )

    B1 = B1_orig.clone().requires_grad_(True)
    x_out = kraken_multilayer_ift(
        x_conv,
        B1,
        lh,
        lr,
        ll,
        ln,
        f_top,
        g_top,
        f_bot,
        g_bot,
        dfdx_top,
        dgdx_top,
        dfdx_bot,
        dgdx_bot,
        eps=1e-12,
    )
    x_out.sum().backward()
    assert B1.grad is not None
    assert torch.isfinite(B1.grad).all()
    assert B1.grad.abs().max() > 1e-15
