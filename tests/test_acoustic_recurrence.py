import sys
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diff_utils.acoustic_recurrence import acoustic_recurrence, AcousticRecurrenceFn


def _reference_recurrence(B1, h2k2_val, loc_start, loc_end, p1_init, p2_init):
    p0 = p1_init.clone()
    p1 = p1_init.clone()
    p2 = p2_init.clone()
    for jj in range(loc_end, loc_start - 1, -1):
        p0 = p1.clone()
        p1 = p2.clone()
        p2 = (h2k2_val - B1[jj]) * p1 - p0
    return -(p2 - p0), -p1


@pytest.mark.parametrize("M", [1, 4])
def test_forward_matches_reference_pekeris(M):
    torch.manual_seed(42)
    N = 50
    B1 = torch.randn(N, dtype=torch.float64) * 0.1 - 2.0
    h2k2 = torch.randn(M, dtype=torch.float64) * 0.01 + 0.5
    p1_init = torch.ones(M, dtype=torch.float64)
    p2_init = torch.ones(M, dtype=torch.float64) * 0.99
    loc_start, loc_end = 5, N - 5

    f_our, g_our = acoustic_recurrence(B1, h2k2, loc_start, loc_end, p1_init, p2_init)
    for m in range(M):
        f_ref, g_ref = _reference_recurrence(
            B1, h2k2[m], loc_start, loc_end, p1_init[m], p2_init[m]
        )
        assert torch.allclose(f_our[m], f_ref, atol=1e-14)
        assert torch.allclose(g_our[m], g_ref, atol=1e-14)


def test_gradcheck_real_B1():
    torch.manual_seed(100)
    N, M = 20, 2
    B1 = (torch.randn(N, dtype=torch.float64) * 0.1 - 2.0).requires_grad_(True)
    h2k2 = (torch.randn(M, dtype=torch.float64) * 0.01 + 0.5).requires_grad_(True)
    p1_init = torch.ones(M, dtype=torch.float64, requires_grad=True)
    p2_init = (torch.ones(M, dtype=torch.float64) * 0.99).requires_grad_(True)
    loc_start, loc_end = 2, N - 3

    def fn(b1, hk, p1i, p2i):
        f, g = acoustic_recurrence(b1, hk, loc_start, loc_end, p1i, p2i)
        return f.sum() + g.sum()

    assert gradcheck(fn, (B1, h2k2, p1_init, p2_init), eps=1e-7, atol=1e-4)


def test_gradcheck_complex():
    torch.manual_seed(200)
    N, M = 15, 2
    B1 = (torch.randn(N, dtype=torch.complex128) * 0.1 - 2.0).requires_grad_(True)
    h2k2 = (torch.randn(M, dtype=torch.complex128) * 0.01 + 0.5).requires_grad_(True)
    p1_init = torch.ones(M, dtype=torch.complex128, requires_grad=True)
    p2_init = (torch.ones(M, dtype=torch.complex128) * 0.99).requires_grad_(True)
    loc_start, loc_end = 1, N - 2

    def fn(b1, hk, p1i, p2i):
        f, g = acoustic_recurrence(b1, hk, loc_start, loc_end, p1i, p2i)
        return (f.abs() ** 2).sum() + (g.abs() ** 2).sum()

    assert gradcheck(fn, (B1, h2k2, p1_init, p2_init), eps=1e-7, atol=1e-4)


def test_batched_m1_matches_unbatched():
    torch.manual_seed(300)
    N = 30
    B1 = torch.randn(N, dtype=torch.float64) * 0.1 - 2.0
    h2k2 = torch.tensor([0.5], dtype=torch.float64)
    p1_init = torch.tensor([1.0], dtype=torch.float64)
    p2_init = torch.tensor([0.99], dtype=torch.float64)

    f_batch, g_batch = acoustic_recurrence(B1, h2k2, 0, N - 1, p1_init, p2_init)
    f_ref, g_ref = _reference_recurrence(
        B1,
        0.5,
        0,
        N - 1,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.99, dtype=torch.float64),
    )
    assert torch.allclose(f_batch[0], f_ref, atol=1e-14)
    assert torch.allclose(g_batch[0], g_ref, atol=1e-14)


def test_no_overflow_at_eigenvalue():
    N = 50
    omega = 2.0 * 3.14159265 * 100.0
    c, depth = 1500.0, 100.0
    h = depth / N
    k = omega / c
    B1 = torch.full((N + 1,), -2.0 + h * h * k * k, dtype=torch.float64)
    k_mode = torch.sqrt(torch.tensor(k * k - (3.14159265 / depth) ** 2))
    h2k2 = torch.tensor([h * h * k_mode * k_mode], dtype=torch.float64)

    _, _, p_history = AcousticRecurrenceFn.apply(
        B1,
        h2k2,
        0,
        N,
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )
    assert p_history.abs().max().item() < 1e40
