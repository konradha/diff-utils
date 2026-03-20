from __future__ import annotations

import torch


class TrapezoidalNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        phi: torch.Tensor,
        B1: torch.Tensor,
        B1C: torch.Tensor,
        rho: torch.Tensor,
        h: float,
        omega2: float,
    ):
        sq_norm, slow, perturb = _forward_python(
            phi.detach(), B1.detach(), B1C.detach(), rho.detach(), h, omega2
        )
        return sq_norm, slow, perturb

    @staticmethod
    def setup_context(ctx, inputs, output):
        phi, B1, B1C, rho, h, omega2 = inputs
        ctx.h = h
        ctx.omega2 = omega2
        ctx.save_for_backward(phi, B1, B1C, rho)

    @staticmethod
    def backward(ctx, grad_sq_norm, grad_slow, grad_perturb):
        phi, B1, B1C, rho = ctx.saved_tensors
        h = ctx.h
        omega2 = ctx.omega2

        grad_phi, grad_B1, grad_B1C, grad_rho = _backward_python(
            grad_sq_norm,
            grad_slow,
            grad_perturb,
            phi,
            B1,
            B1C,
            rho,
            h,
            omega2,
        )

        return grad_phi, grad_B1, grad_B1C, grad_rho, None, None


def _forward_python(phi, B1, B1C, rho, h, omega2):
    N1 = phi.shape[0]
    device = phi.device

    phi_c = phi.to(dtype=torch.complex128, device=device)
    B1_f = B1.to(dtype=torch.float64, device=device)
    B1C_f = B1C.to(dtype=torch.float64, device=device)

    rho_val = float(rho[0])
    rho_omega_h2 = rho_val * omega2 * h * h

    weights = torch.full((N1,), h, dtype=torch.float64, device=device)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    weights_c = weights.to(torch.complex128)

    phi_sq = phi_c * phi_c

    sq_norm = torch.sum(weights_c * phi_sq / rho_val)
    if rho_omega_h2 != 0.0:
        slow = torch.sum(weights_c * (B1_f.to(torch.complex128) + 2.0) * phi_sq / rho_omega_h2)
    else:
        slow = torch.zeros((), dtype=torch.complex128, device=device)
    perturb = 1j * torch.sum(weights_c * B1C_f.to(torch.complex128) * phi_sq / rho_val)

    return sq_norm, slow, perturb


def _backward_python(grad_sq, grad_sl, grad_pr, phi, B1, B1C, rho, h, omega2):
    N1 = phi.shape[0]
    device = phi.device

    rho_val = float(rho[0])
    rho_omega_h2 = rho_val * omega2 * h * h

    weights = torch.full((N1,), h, dtype=torch.float64, device=device)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    phi_c = phi.to(torch.complex128)
    B1_f = B1.to(torch.float64)
    B1C_f = B1C.to(torch.float64)

    gsq = grad_sq.to(torch.complex128)
    gsl = grad_sl.to(torch.complex128)
    gpr = grad_pr.to(torch.complex128)

    # grad_phi
    d_phi = gsq * (2.0 * weights / rho_val).to(torch.complex128) * phi_c
    if rho_omega_h2 != 0.0:
        d_phi = (
            d_phi + gsl * (2.0 * weights * (B1_f + 2.0) / rho_omega_h2).to(torch.complex128) * phi_c
        )
    d_phi = d_phi + gpr * 1j * (2.0 * weights * B1C_f / rho_val).to(torch.complex128) * phi_c

    if phi.is_complex():
        grad_phi = d_phi.to(phi.dtype)
    else:
        grad_phi = d_phi.real.to(phi.dtype)

    # grad_B1 [slow term]
    phi_sq = phi_c * phi_c
    grad_B1 = torch.zeros(N1, dtype=torch.float64, device=device)
    if rho_omega_h2 != 0.0:
        d_b1 = gsl * weights.to(torch.complex128) * phi_sq / rho_omega_h2
        grad_B1 = d_b1.real

    # grad_B1C [pertubation]
    d_b1c = gpr * 1j * weights.to(torch.complex128) * phi_sq / rho_val
    grad_B1C = d_b1c.real

    grad_rho = torch.zeros(N1, dtype=torch.float64, device=device)

    return grad_phi, grad_B1, grad_B1C, grad_rho


def trapezoidal_normalization(
    phi: torch.Tensor,
    B1: torch.Tensor,
    B1C: torch.Tensor,
    rho: torch.Tensor,
    h: float,
    omega2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return TrapezoidalNormFn.apply(phi, B1, B1C, rho, h, omega2)


def _multilayer_layout(phi, B1, B1C, layer_sizes, h, layer_rho):
    if layer_sizes.ndim != 1 or h.ndim != 1 or layer_rho.ndim != 1:
        raise ValueError("layer_sizes, h, and layer_rho must be 1D tensors")
    if layer_sizes.numel() == 0:
        raise ValueError("expected at least one layer")
    if h.numel() != layer_sizes.numel() or layer_rho.numel() != layer_sizes.numel():
        raise ValueError("layer metadata must have one entry per layer")
    if torch.any(layer_sizes <= 0):
        raise ValueError("layer_sizes must be positive")

    device = phi.device
    layer_sizes = layer_sizes.to(device=device, dtype=torch.long)
    h = h.to(device=device, dtype=torch.float64)
    layer_rho = layer_rho.to(device=device, dtype=torch.float64)

    num_layers = int(layer_sizes.numel())
    phi_expected = int(layer_sizes.sum().item() - num_layers + 1)
    b_expected = int(layer_sizes.sum().item())
    if phi.shape[0] != phi_expected:
        raise ValueError(f"phi has length {phi.shape[0]}, expected {phi_expected}")
    if B1.numel() != b_expected or B1C.numel() != b_expected:
        raise ValueError(f"B1/B1C must each have length {b_expected}")

    max_layer = int(layer_sizes.max().item())
    local = torch.arange(max_layer, device=device, dtype=torch.long)

    phi_offsets = torch.zeros(num_layers, device=device, dtype=torch.long)
    b_offsets = torch.zeros(num_layers, device=device, dtype=torch.long)
    if num_layers > 1:
        phi_offsets[1:] = torch.cumsum(layer_sizes[:-1] - 1, dim=0)
        b_offsets[1:] = torch.cumsum(layer_sizes[:-1], dim=0)

    valid = local.unsqueeze(0) < layer_sizes.unsqueeze(1)
    phi_idx = (phi_offsets.unsqueeze(1) + local.unsqueeze(0)).clamp_max(phi.shape[0] - 1)
    b_idx = (b_offsets.unsqueeze(1) + local.unsqueeze(0)).clamp_max(B1.numel() - 1)

    phi_g = phi.index_select(0, phi_idx.reshape(-1)).reshape(num_layers, max_layer, *phi.shape[1:])
    B1_g = (
        B1.to(device=device, dtype=torch.float64)
        .index_select(0, b_idx.reshape(-1))
        .reshape(num_layers, max_layer)
    )
    B1C_g = (
        B1C.to(device=device, dtype=torch.float64)
        .index_select(0, b_idx.reshape(-1))
        .reshape(num_layers, max_layer)
    )

    weights = h[:, None].expand(num_layers, max_layer).clone()
    weights[:, 0] *= 0.5
    end_mask = local.unsqueeze(0) == (layer_sizes - 1).unsqueeze(1)
    weights[end_mask] *= 0.5
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    return phi_g, B1_g, B1C_g, phi_idx, b_idx, valid, weights, h, layer_rho


def _multilayer_forward_python(phi, B1, B1C, layer_sizes, h, layer_rho, omega2):
    phi_g, B1_g, B1C_g, _, _, _, weights, h, layer_rho = _multilayer_layout(
        phi, B1, B1C, layer_sizes, h, layer_rho
    )

    extra_dims = (1,) * (phi.ndim - 1)
    phi_c = phi_g.to(torch.complex128)
    phi_sq = phi_c * phi_c

    sq_coeff = (
        (weights / layer_rho[:, None]).to(torch.complex128).reshape(*weights.shape, *extra_dims)
    )

    rho_omega_h2 = layer_rho * omega2 * h * h
    slow_scale = torch.zeros_like(weights, dtype=torch.float64)
    nonzero = rho_omega_h2 != 0.0
    slow_scale[nonzero] = weights[nonzero] / rho_omega_h2[nonzero, None]
    slow_coeff = (
        (slow_scale * (B1_g + 2.0)).to(torch.complex128).reshape(*weights.shape, *extra_dims)
    )

    perturb_coeff = (
        (1j * weights * B1C_g / layer_rho[:, None])
        .to(torch.complex128)
        .reshape(*weights.shape, *extra_dims)
    )

    sq_norm = torch.sum(sq_coeff * phi_sq, dim=(0, 1))
    slow = torch.sum(slow_coeff * phi_sq, dim=(0, 1))
    perturb = torch.sum(perturb_coeff * phi_sq, dim=(0, 1))
    return sq_norm, slow, perturb


def _multilayer_backward_python(
    grad_sq, grad_sl, grad_pr, phi, B1, B1C, layer_sizes, h, layer_rho, omega2
):
    phi_g, B1_g, B1C_g, phi_idx, b_idx, valid, weights, h, layer_rho = _multilayer_layout(
        phi, B1, B1C, layer_sizes, h, layer_rho
    )

    phi_c = phi_g.to(torch.complex128)
    phi_sq = phi_c * phi_c
    extra_dims = (1,) * (phi.ndim - 1)

    sq_base = (
        (weights / layer_rho[:, None]).to(torch.complex128).reshape(*weights.shape, *extra_dims)
    )

    rho_omega_h2 = layer_rho * omega2 * h * h
    slow_base = torch.zeros_like(weights, dtype=torch.float64)
    nonzero = rho_omega_h2 != 0.0
    slow_base[nonzero] = weights[nonzero] / rho_omega_h2[nonzero, None]
    slow_base = slow_base.to(torch.complex128).reshape(*weights.shape, *extra_dims)

    perturb_base = (
        (1j * weights / layer_rho[:, None])
        .to(torch.complex128)
        .reshape(*weights.shape, *extra_dims)
    )

    gsq = grad_sq.to(torch.complex128).reshape((1, 1) + grad_sq.shape)
    gsl = grad_sl.to(torch.complex128).reshape((1, 1) + grad_sl.shape)
    gpr = grad_pr.to(torch.complex128).reshape((1, 1) + grad_pr.shape)

    grad_phi_local = (
        2.0 * gsq * sq_base * phi_c
        + 2.0
        * gsl
        * slow_base
        * (B1_g + 2.0).to(torch.complex128).reshape(*weights.shape, *extra_dims)
        * phi_c
        + 2.0
        * gpr
        * perturb_base
        * B1C_g.to(torch.complex128).reshape(*weights.shape, *extra_dims)
        * phi_c
    )

    grad_phi_flat = torch.zeros(
        phi.shape[0],
        int(torch.tensor(phi.shape[1:]).prod().item()) if phi.ndim > 1 else 1,
        dtype=torch.complex128,
        device=phi.device,
    )
    local_src = grad_phi_local.reshape(phi_idx.numel(), -1)
    valid_flat = valid.reshape(-1)
    grad_phi_flat.index_add_(0, phi_idx.reshape(-1)[valid_flat], local_src[valid_flat])
    grad_phi = grad_phi_flat.reshape(phi.shape)
    if not phi.is_complex():
        grad_phi = grad_phi.real.to(phi.dtype)
    else:
        grad_phi = grad_phi.to(phi.dtype)

    grad_B1 = torch.zeros_like(B1, dtype=torch.float64, device=phi.device)
    grad_B1_local = (gsl * slow_base * phi_sq).real.reshape(b_idx.numel(), -1).sum(dim=1)
    grad_B1.index_add_(0, b_idx.reshape(-1)[valid_flat], grad_B1_local[valid_flat])

    grad_B1C = torch.zeros_like(B1C, dtype=torch.float64, device=phi.device)
    grad_B1C_local = (gpr * perturb_base * phi_sq).real.reshape(b_idx.numel(), -1).sum(dim=1)
    grad_B1C.index_add_(0, b_idx.reshape(-1)[valid_flat], grad_B1C_local[valid_flat])

    grad_layer_rho = torch.zeros_like(layer_rho, dtype=torch.float64, device=phi.device)
    grad_h = torch.zeros_like(h, dtype=torch.float64, device=phi.device)
    return grad_phi, grad_B1, grad_B1C, None, grad_h, grad_layer_rho


class TrapezoidalMultiLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        phi: torch.Tensor,
        B1: torch.Tensor,
        B1C: torch.Tensor,
        layer_sizes: torch.Tensor,
        h: torch.Tensor,
        layer_rho: torch.Tensor,
        omega2: float,
    ):
        sq_norm, slow, perturb = _multilayer_forward_python(
            phi.detach(),
            B1.detach(),
            B1C.detach(),
            layer_sizes.detach(),
            h.detach(),
            layer_rho.detach(),
            omega2,
        )
        return sq_norm, slow, perturb

    @staticmethod
    def setup_context(ctx, inputs, output):
        phi, B1, B1C, layer_sizes, h, layer_rho, omega2 = inputs
        ctx.omega2 = omega2
        ctx.save_for_backward(phi, B1, B1C, layer_sizes, h, layer_rho)

    @staticmethod
    def backward(ctx, grad_sq_norm, grad_slow, grad_perturb):
        phi, B1, B1C, layer_sizes, h, layer_rho = ctx.saved_tensors
        grad_phi, grad_B1, grad_B1C, _, grad_h, grad_layer_rho = _multilayer_backward_python(
            grad_sq_norm,
            grad_slow,
            grad_perturb,
            phi,
            B1,
            B1C,
            layer_sizes,
            h,
            layer_rho,
            ctx.omega2,
        )
        return grad_phi, grad_B1, grad_B1C, None, grad_h, grad_layer_rho, None


def trapezoidal_multilayer_normalization(
    phi: torch.Tensor,
    B1: torch.Tensor,
    B1C: torch.Tensor,
    layer_sizes: torch.Tensor,
    h: torch.Tensor,
    layer_rho: torch.Tensor,
    omega2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return TrapezoidalMultiLayerNormFn.apply(phi, B1, B1C, layer_sizes, h, layer_rho, omega2)


__all__ = [
    "TrapezoidalNormFn",
    "trapezoidal_normalization",
    "TrapezoidalMultiLayerNormFn",
    "trapezoidal_multilayer_normalization",
]
