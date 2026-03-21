
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.interp import searchsorted_lerp


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
    assert torch.allclose(searchsorted_lerp(z, v, z), v, atol=1e-14)


def test_endpoint_clamping():
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    v = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    assert searchsorted_lerp(z, v, torch.tensor([0.0], dtype=torch.float64))[
        0
    ].item() == pytest.approx(10.0, abs=1e-14)
    assert searchsorted_lerp(z, v, torch.tensor([3.5], dtype=torch.float64))[
        0
    ].item() == pytest.approx(30.0, abs=1e-14)


def test_single_segment():
    z = torch.tensor([0.0, 1.0], dtype=torch.float64)
    v = torch.tensor([3.0, 7.0], dtype=torch.float64)
    q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    assert torch.allclose(
        searchsorted_lerp(z, v, q), torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64), atol=1e-14
    )


def test_gradcheck_real():
    z = torch.linspace(0, 5, 10, dtype=torch.float64)
    v = (torch.sin(z) + 1.0).requires_grad_(True)
    q = torch.tensor([0.7, 2.1, 4.3], dtype=torch.float64)
    assert gradcheck(lambda val: searchsorted_lerp(z, val, q).sum(), (v,), eps=1e-7, atol=1e-5)


def test_gradcheck_complex():
    z = torch.linspace(0, 5, 10, dtype=torch.float64)
    v = (torch.sin(z) + 1j * torch.cos(z)).to(torch.complex128).requires_grad_(True)
    q = torch.tensor([0.7, 2.1, 4.3], dtype=torch.float64)
    assert gradcheck(
        lambda val: (searchsorted_lerp(z, val, q).abs() ** 2).sum(), (v,), eps=1e-7, atol=1e-5
    )


def test_gradient_is_sparse():
    z = torch.linspace(0, 10, 20, dtype=torch.float64)
    v = torch.randn(20, dtype=torch.float64, requires_grad=True)
    searchsorted_lerp(z, v, torch.tensor([3.7], dtype=torch.float64)).backward()
    assert (v.grad != 0).sum().item() == 2
