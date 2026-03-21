
import sys
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.elastic_propagation import elastic_propagation, ElasticPropagationFn


def _make_elastic_layer(n_steps=5, seed=42):
    torch.manual_seed(seed)
    N = n_steps + 2
    B1 = torch.randn(N, dtype=torch.float64) * 0.01
    B2 = torch.randn(N, dtype=torch.float64) * 0.01
    B3 = torch.randn(N, dtype=torch.float64) * 0.01
    B4 = torch.randn(N, dtype=torch.float64) * 0.01
    rho = torch.ones(N, dtype=torch.float64) + torch.randn(N, dtype=torch.float64) * 0.01
    x = 0.5
    h_step = 0.1
    y_init = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    return B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, 0


def test_forward_runs():
    B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start = _make_elastic_layer()
    y_out, i_power = elastic_propagation(
        B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start, going_up=True
    )
    assert y_out.shape == (5,)
    assert torch.isfinite(y_out).all()


def test_forward_direction_consistency():
    B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start = _make_elastic_layer(n_steps=3)

    y_up, _ = elastic_propagation(
        B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start, going_up=True
    )
    y_down, _ = elastic_propagation(
        B1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start, going_up=False
    )

    assert torch.isfinite(y_up).all()
    assert torch.isfinite(y_down).all()


def test_single_step():
    B1, B2, B3, B4, rho, x, y_init, h_step, _, loc_start = _make_elastic_layer(n_steps=1)
    y_out, i_power = elastic_propagation(
        B1, B2, B3, B4, rho, x, y_init, h_step, 1, loc_start, going_up=True
    )
    assert y_out.shape == (5,)
    assert torch.isfinite(y_out).all()


def test_gradcheck_B1():
    n_steps = 3
    B1, B2, B3, B4, rho, x, y_init, h_step, _, loc_start = _make_elastic_layer(n_steps=n_steps)
    B1 = B1.requires_grad_(True)

    def fn(b1):
        y_out, _, _ = ElasticPropagationFn.apply(
            b1, B2, B3, B4, rho, x, y_init, h_step, n_steps, loc_start, True
        )
        return y_out.sum()

    assert gradcheck(fn, (B1,), eps=1e-6, atol=1e-4)


def test_gradcheck_y_init():
    n_steps = 3
    B1, B2, B3, B4, rho, x, _, h_step, _, loc_start = _make_elastic_layer(n_steps=n_steps)
    y_init = torch.tensor([1.0, 0.1, 0.0, -0.1, 0.2], dtype=torch.float64, requires_grad=True)

    def fn(yi):
        y_out, _, _ = ElasticPropagationFn.apply(
            B1, B2, B3, B4, rho, x, yi, h_step, n_steps, loc_start, True
        )
        return y_out.sum()

    assert gradcheck(fn, (y_init,), eps=1e-6, atol=1e-4)


def test_gradcheck_x():
    n_steps = 3
    B1, B2, B3, B4, rho, x, y_init, h_step, _, loc_start = _make_elastic_layer(n_steps=n_steps)

    eps = 1e-7
    y_p, _ = elastic_propagation(
        B1, B2, B3, B4, rho, x + eps, y_init, h_step, n_steps, loc_start, True
    )
    y_m, _ = elastic_propagation(
        B1, B2, B3, B4, rho, x - eps, y_init, h_step, n_steps, loc_start, True
    )
    fd_grad = (y_p.sum() - y_m.sum()) / (2 * eps)

    assert torch.isfinite(torch.tensor(fd_grad))
