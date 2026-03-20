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

    # grad_B1 (from slow term)
    phi_sq = phi_c * phi_c
    grad_B1 = torch.zeros(N1, dtype=torch.float64, device=device)
    if rho_omega_h2 != 0.0:
        d_b1 = gsl * weights.to(torch.complex128) * phi_sq / rho_omega_h2
        grad_B1 = d_b1.real

    # grad_B1C (from perturb term)
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


__all__ = ["TrapezoidalNormFn", "trapezoidal_normalization"]
