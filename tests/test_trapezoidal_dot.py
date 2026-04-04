import sys
from pathlib import Path

import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.trapezoidal_dot import (
    trapezoidal_multilayer_normalization,
)


def _reference_multilayer_terms(phi, B1, B1C, layer_sizes, h, layer_rho, omega2):
    layer_sizes_list = [int(x) for x in layer_sizes.tolist()]
    h_list = [float(x) for x in h.tolist()]
    rho_list = [float(x) for x in layer_rho.tolist()]

    sq = torch.zeros(phi.shape[1:] or (), dtype=torch.complex128)
    sl = torch.zeros_like(sq)
    pr = torch.zeros_like(sq)

    phi_offset = 0
    b_offset = 0
    for n_layer, h_layer, rho_layer in zip(layer_sizes_list, h_list, rho_list):
        phi_slice = phi[phi_offset : phi_offset + n_layer].to(torch.complex128)
        b1_slice = B1[b_offset : b_offset + n_layer]
        b1c_slice = B1C[b_offset : b_offset + n_layer]

        weights = torch.full((n_layer,), h_layer, dtype=torch.float64)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        weights_c = weights.to(torch.complex128).reshape((n_layer,) + (1,) * (phi.ndim - 1))
        phi_sq = phi_slice * phi_slice

        sq = sq + torch.sum(weights_c * phi_sq / rho_layer, dim=0)
        sl = sl + torch.sum(
            weights_c
            * (b1_slice.to(torch.complex128) + 2.0).reshape(weights_c.shape)
            * phi_sq
            / (rho_layer * omega2 * h_layer * h_layer),
            dim=0,
        )
        pr = pr + 1j * torch.sum(
            weights_c
            * b1c_slice.to(torch.complex128).reshape(weights_c.shape)
            * phi_sq
            / rho_layer,
            dim=0,
        )

        phi_offset += n_layer - 1
        b_offset += n_layer

    return sq, sl, pr


def test_multilayer_forward_matches_reference():
    torch.manual_seed(123)
    layer_sizes = torch.tensor([4, 3, 5], dtype=torch.long)
    h = torch.tensor([0.5, 1.0, 0.25], dtype=torch.float64)
    rho = torch.tensor([1.0, 1.7, 0.8], dtype=torch.float64)
    phi = torch.randn(int(layer_sizes.sum().item() - layer_sizes.numel() + 1), dtype=torch.float64)
    B1 = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    B1C = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    omega2 = 17.5

    got = trapezoidal_multilayer_normalization(phi, B1, B1C, layer_sizes, h, rho, omega2)
    ref = _reference_multilayer_terms(phi, B1, B1C, layer_sizes, h, rho, omega2)
    for got_i, ref_i in zip(got, ref):
        assert torch.allclose(got_i, ref_i, atol=1e-14)


def test_multilayer_batched_modes_match_reference():
    torch.manual_seed(321)
    layer_sizes = torch.tensor([3, 4], dtype=torch.long)
    h = torch.tensor([1.5, 0.75], dtype=torch.float64)
    rho = torch.tensor([1.2, 2.0], dtype=torch.float64)
    phi = torch.randn(
        int(layer_sizes.sum().item() - layer_sizes.numel() + 1), 4, dtype=torch.float64
    )
    B1 = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    B1C = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    omega2 = 30.0

    got = trapezoidal_multilayer_normalization(phi, B1, B1C, layer_sizes, h, rho, omega2)
    ref = _reference_multilayer_terms(phi, B1, B1C, layer_sizes, h, rho, omega2)
    for got_i, ref_i in zip(got, ref):
        assert got_i.shape == (4,)
        assert torch.allclose(got_i, ref_i, atol=1e-14)


def test_multilayer_gradcheck_phi_real():
    torch.manual_seed(500)
    layer_sizes = torch.tensor([3, 4], dtype=torch.long)
    h = torch.tensor([1.0, 0.5], dtype=torch.float64)
    rho = torch.tensor([1.4, 2.1], dtype=torch.float64)
    phi = torch.randn(
        int(layer_sizes.sum().item() - layer_sizes.numel() + 1),
        dtype=torch.float64,
        requires_grad=True,
    )
    B1 = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    B1C = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    fn = lambda p: sum(
        x.real for x in trapezoidal_multilayer_normalization(p, B1, B1C, layer_sizes, h, rho, 100.0)
    )
    assert gradcheck(fn, (phi,), eps=1e-7, atol=1e-4)


def test_multilayer_gradcheck_B1_and_B1C():
    torch.manual_seed(600)
    layer_sizes = torch.tensor([4, 2], dtype=torch.long)
    h = torch.tensor([0.75, 1.25], dtype=torch.float64)
    rho = torch.tensor([1.1, 0.95], dtype=torch.float64)
    phi = torch.randn(int(layer_sizes.sum().item() - layer_sizes.numel() + 1), dtype=torch.float64)
    B1 = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64, requires_grad=True)
    B1C = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64, requires_grad=True)

    assert gradcheck(
        lambda b1: trapezoidal_multilayer_normalization(
            phi, b1, B1C.detach(), layer_sizes, h, rho, 40.0
        )[1].real,
        (B1,),
        eps=1e-7,
        atol=1e-4,
    )
    assert gradcheck(
        lambda b1c: trapezoidal_multilayer_normalization(
            phi, B1.detach(), b1c, layer_sizes, h, rho, 40.0
        )[2].real,
        (B1C,),
        eps=1e-7,
        atol=1e-4,
    )


def test_multilayer_interface_gradient_accumulates_from_both_sides():
    layer_sizes = torch.tensor([3, 3], dtype=torch.long)
    h = torch.tensor([1.0, 2.0], dtype=torch.float64)
    rho = torch.tensor([1.0, 4.0], dtype=torch.float64)
    phi = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0], dtype=torch.float64, requires_grad=True)
    B1 = torch.zeros(int(layer_sizes.sum().item()), dtype=torch.float64)
    B1C = torch.zeros(int(layer_sizes.sum().item()), dtype=torch.float64)

    sq, _, _ = trapezoidal_multilayer_normalization(phi, B1, B1C, layer_sizes, h, rho, 1.0)
    sq.real.backward()

    expected_interface_grad = 2.0 * phi.detach()[2] * (0.5 * h[0] / rho[0] + 0.5 * h[1] / rho[1])
    assert torch.allclose(phi.grad[2], expected_interface_grad)
