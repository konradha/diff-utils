from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banded.interp import searchsorted_lerp, SearchsortedLerpFn


def test_forward_matches_np_interp():
    z = torch.linspace(0, 10, 20, dtype=torch.float64)
    v = torch.sin(z)
    q = torch.tensor([0.5, 2.3, 5.0, 7.7, 9.9], dtype=torch.float64)

    result = searchsorted_lerp(z, v, q)
    expected = np.interp(q.numpy(), z.numpy(), v.numpy())

    assert torch.allclose(result, torch.from_numpy(expected), atol=1e-14)


def test_exact_at_knot_points():
    z = torch.linspace(0, 5, 11, dtype=torch.float64)
    v = z**2
    result = searchsorted_lerp(z, v, z)
    assert torch.allclose(result, v, atol=1e-14)


def test_endpoint_clamping():
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    v = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)

    # Query below range
    q = torch.tensor([0.0, 0.5], dtype=torch.float64)
    result = searchsorted_lerp(z, v, q)
    assert result[0].item() == pytest.approx(10.0, abs=1e-14)

    # Query above range
    q = torch.tensor([3.5, 4.0], dtype=torch.float64)
    result = searchsorted_lerp(z, v, q)
    assert result[0].item() == pytest.approx(30.0, abs=1e-14)


def test_single_segment():
    z = torch.tensor([0.0, 1.0], dtype=torch.float64)
    v = torch.tensor([3.0, 7.0], dtype=torch.float64)
    q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)

    result = searchsorted_lerp(z, v, q)
    expected = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-14)


def test_gradcheck_real():
    z = torch.linspace(0, 5, 10, dtype=torch.float64)
    v = (torch.sin(z) + 1.0).requires_grad_(True)
    q = torch.tensor([0.7, 2.1, 4.3], dtype=torch.float64)

    def fn(val):
        return searchsorted_lerp(z, val, q).sum()

    assert gradcheck(fn, (v,), eps=1e-7, atol=1e-5)


def test_gradcheck_complex():
    z = torch.linspace(0, 5, 10, dtype=torch.float64)
    v_re = torch.sin(z)
    v_im = torch.cos(z)
    v = (v_re + 1j * v_im).to(torch.complex128).requires_grad_(True)
    q = torch.tensor([0.7, 2.1, 4.3], dtype=torch.float64)

    def fn(val):
        out = searchsorted_lerp(z, val, q)
        return (out.abs() ** 2).sum()

    assert gradcheck(fn, (v,), eps=1e-7, atol=1e-5)


def test_gradient_is_sparse():
    """Only two knots should receive gradient per query."""
    z = torch.linspace(0, 10, 20, dtype=torch.float64)
    v = torch.randn(20, dtype=torch.float64, requires_grad=True)
    q = torch.tensor([3.7], dtype=torch.float64)  # between knots 7 and 8

    result = searchsorted_lerp(z, v, q)
    result.backward()

    assert v.grad is not None
    nonzero = (v.grad != 0).sum().item()
    assert nonzero == 2, f"Expected 2 nonzero gradients, got {nonzero}"
