import math
import sys
from pathlib import Path

import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.acoustic_recurrence import (
    AcousticRecurrenceFn,
    acoustic_recurrence_nograd,
    acoustic_recurrence_scalar_counted,
)


def _pekeris_B1(c=1500.0, freq=100.0, depth=100.0, N=1001):
    omega2 = (2.0 * math.pi * freq) ** 2
    h = depth / (N - 1)
    B1 = torch.full((N,), -2.0 + h * h * omega2 / c**2, dtype=torch.float64)
    return B1, h, omega2


def test_nograd_matches_autograd():
    B1, h, omega2 = _pekeris_B1()
    x = omega2 / 1500**2 * 0.9
    h2k2 = h * h * x
    N = B1.shape[0]

    p1 = torch.tensor([-2.0], dtype=torch.float64)
    p2 = torch.tensor([(B1[-1].item() - h2k2) * 1.0 - 2.0 * h * 0.5 * 1.0], dtype=torch.float64)
    h2k2_t = torch.tensor([h2k2], dtype=torch.float64)

    f1, g1, ph1 = AcousticRecurrenceFn.apply(B1, h2k2_t, 0, N - 2, p1, p2)
    f2, g2, ph2 = acoustic_recurrence_nograd(B1, h2k2_t, 0, N - 2, p1, p2)

    assert torch.allclose(f1, f2, atol=1e-14)
    assert torch.allclose(g1, g2, atol=1e-14)
    assert torch.allclose(ph1, ph2, atol=1e-14)


def test_scalar_counted_matches_tensor():
    B1, h, omega2 = _pekeris_B1()
    x = omega2 / 1500**2 * 0.9
    h2k2 = h * h * x
    N = B1.shape[0]

    p1_val = -2.0
    p2_val = (B1[-1].item() - h2k2) * 1.0 - 2.0 * h * 0.5 * 1.0

    f_scalar, g_scalar, mc = acoustic_recurrence_scalar_counted(
        B1,
        h2k2,
        0,
        N - 2,
        p1_val,
        p2_val,
    )

    p1_t = torch.tensor([p1_val], dtype=torch.float64)
    p2_t = torch.tensor([p2_val], dtype=torch.float64)
    h2k2_t = torch.tensor([h2k2], dtype=torch.float64)
    f_t, g_t, _ = acoustic_recurrence_nograd(B1, h2k2_t, 0, N - 2, p1_t, p2_t)

    assert abs(f_scalar - f_t[0].item()) < 1e-12
    assert abs(g_scalar - g_t[0].item()) < 1e-12


def test_scalar_counted_mode_count():
    B1, h, omega2 = _pekeris_B1()
    N = B1.shape[0]
    x_min = omega2 / 1600**2
    h2k2_min = h * h * x_min
    _, _, mc_min = acoustic_recurrence_scalar_counted(
        B1,
        h2k2_min,
        0,
        N - 2,
        -2.0,
        (B1[-1].item() - h2k2_min) - 2.0 * h * 0.5,
    )
    x_max = omega2 / 1400**2
    h2k2_max = h * h * x_max
    _, _, mc_max = acoustic_recurrence_scalar_counted(
        B1,
        h2k2_max,
        0,
        N - 2,
        -2.0,
        (B1[-1].item() - h2k2_max) - 2.0 * h * 0.5,
    )

    assert mc_min >= 0
    assert mc_max >= 0
    assert mc_min >= mc_max


def test_scalar_counted_performance():
    B1, h, omega2 = _pekeris_B1(N=7501)
    N = B1.shape[0]
    x = omega2 / 1500**2 * 0.9
    h2k2 = h * h * x

    t0 = time.perf_counter()
    for _ in range(1000):
        acoustic_recurrence_scalar_counted(B1, h2k2, 0, N - 2, -2.0, 0.5)
    t1 = time.perf_counter()
    scalar_ms = (t1 - t0) / 1000 * 1000

    p1_t = torch.tensor([-2.0], dtype=torch.float64)
    p2_t = torch.tensor([0.5], dtype=torch.float64)
    h2k2_t = torch.tensor([h2k2], dtype=torch.float64)
    t2 = time.perf_counter()
    for _ in range(1000):
        acoustic_recurrence_nograd(B1, h2k2_t, 0, N - 2, p1_t, p2_t)
    t3 = time.perf_counter()
    nograd_ms = (t3 - t2) / 1000 * 1000

    t4 = time.perf_counter()
    for _ in range(1000):
        AcousticRecurrenceFn.apply(B1, h2k2_t, 0, N - 2, p1_t, p2_t)
    t5 = time.perf_counter()
    autograd_ms = (t5 - t4) / 1000 * 1000

    assert scalar_ms < autograd_ms, (
        f"scalar ({scalar_ms:.3f}ms) not faster than autograd ({autograd_ms:.3f}ms)"
    )
