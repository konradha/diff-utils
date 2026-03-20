"""Batched IFT backward for KRAKEN/KRAKENC eigenvalue differentiation.

This replaces the generic eigenvalue_gate for the KRAKEN-specific dispersion
function. Instead of looping over modes in Python and calling torch.autograd.grad,
it computes all M modes' IFT gradients in one batched acoustic_recurrence call.

Dispersion function structure:
    Δ_m(x_m; B1, ρ, h, BCs) = f_m · g_top_m - g_m · f_top_m

where (f_m, g_m) come from the acoustic recurrence with h2k2_m = h² · x_m,
and (f_top_m, g_top_m) come from boundary conditions.

IFT: dx_m/dθ = -(∂Δ_m/∂θ) / (∂Δ_m/∂x_m)

Near-degenerate eigenvalues (∂Δ/∂x ≈ 0) are regularized via Lorentz
broadening: 1/z → z̄/(|z|² + ε²), following the same pattern as the
tSVD F-matrix regularization.
"""

from __future__ import annotations

import torch

from banded._ext import _cpu_ext, _tensor_has_storage
from banded.acoustic_recurrence import AcousticRecurrenceFn, _backward_python


def _run_recurrence_bwd(
    grad_f_num,
    grad_g_val,
    B1,
    h2k2,
    p_history,
    loc_start,
    loc_end,
    p1_init,
    p2_init,
    is_complex,
):
    """Dispatch recurrence backward to C++ or Python."""
    ext = _cpu_ext()
    if ext is not None and _tensor_has_storage(B1):
        return ext.acoustic_recurrence_bwd(
            grad_f_num.contiguous(),
            grad_g_val.contiguous(),
            B1.contiguous(),
            h2k2.contiguous(),
            p_history.contiguous(),
            loc_start,
            loc_end,
            p1_init.contiguous(),
            p2_init.contiguous(),
            is_complex,
        )
    return _backward_python(
        grad_f_num,
        grad_g_val,
        B1,
        h2k2,
        p_history,
        loc_start,
        loc_end,
        p1_init,
        p2_init,
        is_complex,
    )


def _lorentz_inv(z: torch.Tensor, eps: float) -> torch.Tensor:
    """Lorentz-broadened inverse: z̄/(|z|² + ε²).

    For real z this is z/(z² + ε²).
    For complex z this is conj(z)/(|z|² + ε²).
    Smoothly regularizes 1/z near z=0 with maximum magnitude 1/(2ε).
    """
    if z.is_complex():
        return z.conj() / (z.real**2 + z.imag**2 + eps * eps)
    return z / (z * z + eps * eps)


class KrakenEigenvalueIFT(torch.autograd.Function):
    """Batched IFT for KRAKEN eigenvalues.

    Forward: passthrough of converged eigenvalues.
    Backward: batched dispersion derivative via acoustic recurrence backward.
    Near-zero ∂Δ/∂x regularized via Lorentz broadening (eps parameter).
    """

    @staticmethod
    def forward(
        x_converged: torch.Tensor,  # [M]
        B1: torch.Tensor,  # [N]
        rho_med: float,
        h_med: float,
        loc_start: int,
        loc_end: int,
        f_bc_top: torch.Tensor,  # [M]
        g_bc_top: torch.Tensor,  # [M]
        f_bc_bot: torch.Tensor,  # [M]
        g_bc_bot: torch.Tensor,  # [M]
        dfdx_top: torch.Tensor,  # [M]
        dgdx_top: torch.Tensor,  # [M]
        dfdx_bot: torch.Tensor,  # [M]
        dgdx_bot: torch.Tensor,  # [M]
        eps: float,  # Lorentz broadening width
    ):
        return x_converged.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            x_converged,
            B1,
            rho_med,
            h_med,
            loc_start,
            loc_end,
            f_bc_top,
            g_bc_top,
            f_bc_bot,
            g_bc_bot,
            dfdx_top,
            dgdx_top,
            dfdx_bot,
            dgdx_bot,
            eps,
        ) = inputs
        ctx.rho_med = rho_med
        ctx.h_med = h_med
        ctx.loc_start = loc_start
        ctx.loc_end = loc_end
        ctx.eps = eps
        ctx.save_for_backward(
            x_converged,
            B1,
            f_bc_top,
            g_bc_top,
            f_bc_bot,
            g_bc_bot,
            dfdx_top,
            dgdx_top,
            dfdx_bot,
            dgdx_bot,
        )

    @staticmethod
    def backward(ctx, grad_x):
        (
            x_converged,
            B1,
            f_bc_top,
            g_bc_top,
            f_bc_bot,
            g_bc_bot,
            dfdx_top,
            dgdx_top,
            dfdx_bot,
            dgdx_bot,
        ) = ctx.saved_tensors
        rho_med = ctx.rho_med
        h_med = ctx.h_med
        loc_start = ctx.loc_start
        loc_end = ctx.loc_end
        eps = ctx.eps

        M = x_converged.shape[0]
        h2 = h_med * h_med
        scale = 2.0 * h_med * rho_med
        is_complex = x_converged.is_complex()

        # --- Forward recurrence at converged eigenvalues ---
        h2k2 = h2 * x_converged.detach()

        g_bot = g_bc_bot.detach()
        f_bot = f_bc_bot.detach()
        B1_end = B1[loc_end].detach()

        p1_init = -2.0 * g_bot
        p2_init = (B1_end - h2k2) * g_bot - 2.0 * h_med * f_bot * rho_med

        f_num, g_val, p_history = AcousticRecurrenceFn.apply(
            B1.detach(),
            h2k2,
            loc_start,
            loc_end,
            p1_init,
            p2_init,
        )

        f_interior = f_num / scale
        g_interior = g_val

        # --- Dispersion partial derivatives ---
        g_top = g_bc_top.detach()
        f_top = f_bc_top.detach()

        grad_f_num_unit = g_top / scale
        grad_g_val_unit = -f_top

        # Recurrence backward → ∂Δ/∂B1, ∂Δ/∂h2k2, ∂Δ/∂p_init
        grad_B1_delta, grad_h2k2_delta, grad_p1i_delta, grad_p2i_delta = _run_recurrence_bwd(
            grad_f_num_unit,
            grad_g_val_unit,
            B1.detach(),
            h2k2,
            p_history,
            loc_start,
            loc_end,
            p1_init,
            p2_init,
            is_complex,
        )

        # Chain through initial conditions
        grad_B1_delta[loc_end] += (grad_p2i_delta * g_bot).sum()
        grad_h2k2_delta += grad_p2i_delta * (-g_bot)

        # ∂Δ/∂x = (interior via h2k2) + (boundary)
        dDelta_dx = grad_h2k2_delta * h2 + (
            f_interior.detach() * dgdx_top.detach() - g_interior.detach() * dfdx_top.detach()
        )

        # --- IFT with Lorentz-broadened inverse ---
        # ift_scale_m = -grad_x_m / (∂Δ_m/∂x_m)
        #             → -grad_x_m · conj(∂Δ/∂x) / (|∂Δ/∂x|² + ε²)
        ift_scale = -grad_x * _lorentz_inv(dDelta_dx, eps)

        # Rerun backward with IFT-scaled gradients → grad_B1
        grad_f_num_ift = ift_scale * g_top / scale
        grad_g_val_ift = ift_scale * (-f_top)

        grad_B1_ift, _, grad_p1i_ift, grad_p2i_ift = _run_recurrence_bwd(
            grad_f_num_ift,
            grad_g_val_ift,
            B1.detach(),
            h2k2,
            p_history,
            loc_start,
            loc_end,
            p1_init,
            p2_init,
            is_complex,
        )

        grad_B1_ift[loc_end] += (grad_p2i_ift * g_bot).sum()

        return (
            None,
            grad_B1_ift,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def kraken_eigenvalue_ift(
    x_converged: torch.Tensor,
    B1: torch.Tensor,
    rho_med: float,
    h_med: float,
    loc_start: int,
    loc_end: int,
    f_bc_top: torch.Tensor,
    g_bc_top: torch.Tensor,
    f_bc_bot: torch.Tensor,
    g_bc_bot: torch.Tensor,
    dfdx_top: torch.Tensor,
    dgdx_top: torch.Tensor,
    dfdx_bot: torch.Tensor,
    dgdx_bot: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Batched IFT gate for KRAKEN eigenvalues.

    Returns eigenvalues with gradients flowing back to B1.
    All M modes processed in one batched call — no Python loop.

    Args:
        eps: Lorentz broadening width for regularizing 1/(∂Δ/∂x) near
             degenerate eigenvalues. Default 1e-12. Set larger (e.g. 1e-6)
             for problems with near-degenerate modes.
    """
    return KrakenEigenvalueIFT.apply(
        x_converged,
        B1,
        rho_med,
        h_med,
        loc_start,
        loc_end,
        f_bc_top,
        g_bc_top,
        f_bc_bot,
        g_bc_bot,
        dfdx_top,
        dgdx_top,
        dfdx_bot,
        dgdx_bot,
        eps,
    )


__all__ = ["KrakenEigenvalueIFT", "kraken_eigenvalue_ift"]
