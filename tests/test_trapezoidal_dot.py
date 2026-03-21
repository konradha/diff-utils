import sys
from pathlib import Path

import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.trapezoidal_dot import (
    trapezoidal_multilayer_normalization,
    trapezoidal_normalization,
)


def _reference_interior_terms(phi, B1, B1C, rho, h, omega2):
    N1 = phi.shape[0]
    rho_val = float(rho[0])
    rho_omega_h2 = rho_val * omega2 * h * h
    phi_c = phi.to(torch.complex128)
    phi_sq = phi_c * phi_c
    weights = torch.full((N1,), h, dtype=torch.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    w = weights.to(torch.complex128)
    return (
        torch.sum(w * phi_sq / rho_val),
        torch.sum(w * (B1.to(torch.complex128) + 2.0) * phi_sq / rho_omega_h2),
        1j * torch.sum(w * B1C.to(torch.complex128) * phi_sq / rho_val),
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


def test_forward_matches_reference():
    torch.manual_seed(42)
    N, h = 50, 2.0
    omega = 2 * 3.14159265 * 100.0
    k = omega / 1500.0
    B1 = torch.full((N + 1,), -2.0 + h * h * k * k, dtype=torch.float64)
    B1C = torch.zeros(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.0, dtype=torch.float64)
    phi = torch.randn(N + 1, dtype=torch.float64)

    sq, sl, pr = trapezoidal_normalization(phi, B1, B1C, rho, h, omega**2)
    sq_ref, sl_ref, pr_ref = _reference_interior_terms(phi, B1, B1C, rho, h, omega**2)
    assert torch.allclose(sq, sq_ref, atol=1e-14)
    assert torch.allclose(sl, sl_ref, atol=1e-14)
    assert torch.allclose(pr, pr_ref, atol=1e-14)


def test_gradcheck_phi_real():
    torch.manual_seed(100)
    N = 10
    B1 = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    phi = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)
    fn = lambda p: sum(x.real for x in trapezoidal_normalization(p, B1, B1C, rho, 1.0, 100.0))
    assert gradcheck(fn, (phi,), eps=1e-7, atol=1e-4)


def test_gradcheck_B1():
    torch.manual_seed(200)
    N = 8
    phi = torch.randn(N + 1, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)
    assert gradcheck(
        lambda b1: trapezoidal_normalization(phi, b1, B1C, rho, 1.0, 100.0)[1].real,
        (B1,),
        eps=1e-7,
        atol=1e-4,
    )


def test_gradcheck_B1C():
    torch.manual_seed(300)
    N = 8
    phi = torch.randn(N + 1, dtype=torch.float64)
    B1 = torch.randn(N + 1, dtype=torch.float64)
    rho = torch.full((N + 1,), 1.5, dtype=torch.float64)
    B1C = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)
    assert gradcheck(
        lambda b1c: trapezoidal_normalization(phi, B1, b1c, rho, 1.0, 100.0)[2].real,
        (B1C,),
        eps=1e-7,
        atol=1e-4,
    )


def test_n1_edge_case():
    sq, _, _ = trapezoidal_normalization(
        torch.tensor([3.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        1.0,
        1.0,
    )
    assert abs(sq.real.item() - 2.25) < 1e-14


def test_zero_phi_gives_zero():
    N = 10
    sq, sl, pr = trapezoidal_normalization(
        torch.zeros(N + 1, dtype=torch.float64),
        torch.randn(N + 1, dtype=torch.float64),
        torch.randn(N + 1, dtype=torch.float64),
        torch.ones(N + 1, dtype=torch.float64),
        1.0,
        1.0,
    )
    assert all(x.abs().item() < 1e-14 for x in (sq, sl, pr))


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


def test_multilayer_matches_sum_of_single_layer_calls():
    torch.manual_seed(999)
    layer_sizes = torch.tensor([5, 4, 3], dtype=torch.long)
    h = torch.tensor([0.8, 1.1, 0.4], dtype=torch.float64)
    rho = torch.tensor([1.3, 0.9, 1.8], dtype=torch.float64)
    phi = torch.randn(int(layer_sizes.sum().item() - layer_sizes.numel() + 1), dtype=torch.float64)
    B1 = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    B1C = torch.randn(int(layer_sizes.sum().item()), dtype=torch.float64)
    omega2 = 11.0

    sq_ref = torch.zeros((), dtype=torch.complex128)
    sl_ref = torch.zeros((), dtype=torch.complex128)
    pr_ref = torch.zeros((), dtype=torch.complex128)
    phi_offset = 0
    b_offset = 0
    for n_layer, h_layer, rho_layer in zip(layer_sizes.tolist(), h.tolist(), rho.tolist()):
        phi_slice = phi[phi_offset : phi_offset + n_layer]
        b1_slice = B1[b_offset : b_offset + n_layer]
        b1c_slice = B1C[b_offset : b_offset + n_layer]
        rho_slice = torch.full((n_layer,), rho_layer, dtype=torch.float64)
        sq_i, sl_i, pr_i = trapezoidal_normalization(
            phi_slice, b1_slice, b1c_slice, rho_slice, h_layer, omega2
        )
        sq_ref = sq_ref + sq_i
        sl_ref = sl_ref + sl_i
        pr_ref = pr_ref + pr_i
        phi_offset += n_layer - 1
        b_offset += n_layer

    sq, sl, pr = trapezoidal_multilayer_normalization(phi, B1, B1C, layer_sizes, h, rho, omega2)
    assert torch.allclose(sq, sq_ref, atol=1e-14)
    assert torch.allclose(sl, sl_ref, atol=1e-14)
    assert torch.allclose(pr, pr_ref, atol=1e-14)


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
