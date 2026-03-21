import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.acoustic_solver import acoustic_solve1, acoustic_solve2, BC_VACUUM, BC_ACOUSTIC


def _pekeris_setup(freq=10.0, c=1500.0, depth=5000.0, N=1001, c_bot=2000.0, rho_bot=2.0):
    omega = 2.0 * math.pi * freq
    omega2 = omega * omega
    h = depth / (N - 1)
    B1 = torch.full((N,), -2.0 + h * h * omega2 / c**2, dtype=torch.float64)
    layer_loc = torch.tensor([0], dtype=torch.int64)
    layer_n = torch.tensor([N], dtype=torch.int64)
    layer_h = torch.tensor([h], dtype=torch.float64)
    layer_rho = torch.tensor([1.0], dtype=torch.float64)
    x_min = 1.00001 * omega2 / c_bot**2
    x_max = omega2 / c**2
    return B1, layer_loc, layer_n, layer_h, layer_rho, omega2, x_min, x_max


def test_solve1_pekeris_finds_modes():
    B1, ll, ln, lh, lr, omega2, x_min, x_max = _pekeris_setup()
    ev, M = acoustic_solve1(
        B1,
        ll,
        ln,
        lh,
        lr,
        omega2,
        BC_ACOUSTIC,
        2000.0,
        0.0,
        2.0,
        BC_VACUUM,
        0.0,
        0.0,
        0.0,
        x_min,
        x_max,
    )
    assert M > 0
    assert ev.shape[0] == M
    # ascending sorted spectrum
    assert (ev >= x_min - 1e-10).all()
    assert (ev <= x_max + 1e-10).all()


def test_solve1_pekeris_matches_reference():
    B1, ll, ln, lh, lr, omega2, x_min, x_max = _pekeris_setup()
    ev, M = acoustic_solve1(
        B1,
        ll,
        ln,
        lh,
        lr,
        omega2,
        BC_ACOUSTIC,
        2000.0,
        0.0,
        2.0,
        BC_VACUUM,
        0.0,
        0.0,
        0.0,
        x_min,
        x_max,
    )
    k_last = ev[M - 1] ** 0.5
    k_approx = (omega2**0.5) / 1500.0
    assert abs(k_last / k_approx - 1.0) < 0.1


def test_solve1_arctic_scale():
    B1, ll, ln, lh, lr, omega2, x_min, x_max = _pekeris_setup(
        freq=200.0,
        c=1436.0,
        depth=3750.0,
        N=7501,
        c_bot=1510.0,
        rho_bot=1.0,
    )
    x_max = omega2 / 1436.0**2
    x_min = 1.00001 * omega2 / 1510.0**2

    import time

    t0 = time.perf_counter()
    ev, M = acoustic_solve1(
        B1,
        ll,
        ln,
        lh,
        lr,
        omega2,
        BC_ACOUSTIC,
        1510.0,
        0.0,
        1.0,
        BC_VACUUM,
        0.0,
        0.0,
        0.0,
        x_min,
        x_max,
        precision=15.0,
    )
    t1 = time.perf_counter()
    assert M > 50
    assert t1 - t0 < 1.0, f"solve1 too slow: {t1 - t0:.3f}s for M={M}"


def test_solve2_refines_from_solve1():
    B1_1, ll_1, ln_1, lh_1, lr_1, omega2, x_min, x_max = _pekeris_setup(N=1001)
    ev1, M = acoustic_solve1(
        B1_1,
        ll_1,
        ln_1,
        lh_1,
        lr_1,
        omega2,
        BC_ACOUSTIC,
        2000.0,
        0.0,
        2.0,
        BC_VACUUM,
        0.0,
        0.0,
        0.0,
        x_min,
        x_max,
    )
    assert M > 0

    B1_2, ll_2, ln_2, lh_2, lr_2, _, _, _ = _pekeris_setup(N=2001)
    ev2, M2 = acoustic_solve2(
        B1_2,
        ll_2,
        ln_2,
        lh_2,
        lr_2,
        omega2,
        BC_ACOUSTIC,
        2000.0,
        0.0,
        2.0,
        BC_VACUUM,
        0.0,
        0.0,
        0.0,
        ev1,
        M,
        precision=15.0,
        c_high=2000.0,
    )
    assert M2 > 0
    for i in range(min(M, M2)):
        assert abs(ev2[i].item() - ev1[i].item()) / abs(ev1[i].item()) < 0.01


def test_solve1_two_layer():
    omega = 2.0 * math.pi * 10.0
    omega2 = omega * omega
    n1, n2 = 101, 41
    h1 = 100.0 / (n1 - 1)
    h2 = 20.0 / (n2 - 1)
    B1_1 = torch.full((n1,), -2.0 + h1 * h1 * omega2 / 1500**2, dtype=torch.float64)
    B1_2 = torch.full((n2,), -2.0 + h2 * h2 * omega2 / 1400**2, dtype=torch.float64)
    B1 = torch.cat([B1_1, B1_2])

    layer_loc = torch.tensor([0, n1], dtype=torch.int64)
    layer_n = torch.tensor([n1, n2], dtype=torch.int64)
    layer_h = torch.tensor([h1, h2], dtype=torch.float64)
    layer_rho = torch.tensor([1.0, 1.5], dtype=torch.float64)

    x_min = 1.00001 * omega2 / 2000**2
    x_max = omega2 / 1400**2

    ev, M = acoustic_solve1(
        B1,
        layer_loc,
        layer_n,
        layer_h,
        layer_rho,
        omega2,
        BC_ACOUSTIC,
        2000.0,
        0.0,
        2.0,
        BC_VACUUM,
        0.0,
        0.0,
        0.0,
        x_min,
        x_max,
    )
    assert M > 0
    assert torch.isfinite(ev).all()
