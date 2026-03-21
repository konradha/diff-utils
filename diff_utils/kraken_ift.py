import torch

from diff_utils._ext import _cpu_ext, _tensor_has_storage
from diff_utils.acoustic_recurrence import AcousticRecurrenceFn, _backward_python


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
    if z.is_complex():
        return z.conj() / (z.real**2 + z.imag**2 + eps * eps)
    return z / (z * z + eps * eps)


class KrakenEigenvalueIFT(torch.autograd.Function):
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

        g_top = g_bc_top.detach()
        f_top = f_bc_top.detach()

        grad_f_num_unit = g_top / scale
        grad_g_val_unit = -f_top

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

        grad_B1_delta[loc_end] += (grad_p2i_delta * g_bot).sum()
        grad_h2k2_delta += grad_p2i_delta * (-g_bot)

        dp1_init_dx = -2.0 * dgdx_bot.detach()
        dp2_init_dx = (
            +(B1_end - h2k2) * dgdx_bot.detach() - 2.0 * h_med * rho_med * dfdx_bot.detach()
        )
        dDelta_dx = (
            grad_h2k2_delta * h2
            + grad_p1i_delta * dp1_init_dx
            + grad_p2i_delta * dp2_init_dx
            + f_interior.detach() * dgdx_top.detach()
            - g_interior.detach() * dfdx_top.detach()
        )

        ift_scale = -grad_x * _lorentz_inv(dDelta_dx, eps)

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


class KrakenMultiLayerIFT(torch.autograd.Function):
    @staticmethod
    def forward(
        x_converged: torch.Tensor,
        B1: torch.Tensor,
        layer_h: torch.Tensor,
        layer_rho: torch.Tensor,
        layer_loc: torch.Tensor,
        layer_n: torch.Tensor,
        f_bc_top: torch.Tensor,
        g_bc_top: torch.Tensor,
        f_bc_bot: torch.Tensor,
        g_bc_bot: torch.Tensor,
        dfdx_top: torch.Tensor,
        dgdx_top: torch.Tensor,
        dfdx_bot: torch.Tensor,
        dgdx_bot: torch.Tensor,
        eps: float,
    ):
        return x_converged.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            x_converged,
            B1,
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
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
        ctx.eps = eps
        ctx.save_for_backward(
            x_converged,
            B1,
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
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
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
            f_bc_top,
            g_bc_top,
            f_bc_bot,
            g_bc_bot,
            dfdx_top,
            dgdx_top,
            dfdx_bot,
            dgdx_bot,
        ) = ctx.saved_tensors
        eps = ctx.eps

        M = x_converged.shape[0]
        n_layers = layer_h.shape[0]
        dtype = B1.dtype

        def _eval_delta(B1_d, x_t, f_bot_t, g_bot_t, f_top_t, g_top_t):
            f_c = f_bot_t
            g_c = g_bot_t
            for layer_idx in range(n_layers - 1, -1, -1):
                h = layer_h[layer_idx].item()
                rho = layer_rho[layer_idx].item()
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_e = loc + n_pts - 1

                h2k2 = h * h * x_t
                p1 = (-2.0 * g_c).unsqueeze(0)
                p2 = ((B1_d[loc_e] - h2k2) * g_c - 2.0 * h * f_c * rho).unsqueeze(0)

                f_num, g_val, _ = AcousticRecurrenceFn.apply(
                    B1_d,
                    h2k2.unsqueeze(0),
                    loc,
                    loc_e - 1,
                    p1,
                    p2,
                )

                f_c = f_num[0] / (2.0 * h * rho)
                g_c = g_val[0]

            return f_c * g_top_t - g_c * f_top_t

        grad_B1_total = torch.zeros_like(B1)

        with torch.enable_grad():
            for m in range(M):
                x_m = x_converged[m].item()

                B1_track = B1.detach().clone().requires_grad_(True)
                x_track = torch.tensor(x_m, dtype=dtype, requires_grad=True)
                f_bot_t = torch.tensor(f_bc_bot[m].item(), dtype=dtype, requires_grad=True)
                g_bot_t = torch.tensor(g_bc_bot[m].item(), dtype=dtype, requires_grad=True)
                f_top_t = torch.tensor(f_bc_top[m].item(), dtype=dtype, requires_grad=True)
                g_top_t = torch.tensor(g_bc_top[m].item(), dtype=dtype, requires_grad=True)

                delta = _eval_delta(B1_track, x_track, f_bot_t, g_bot_t, f_top_t, g_top_t)

                grads = torch.autograd.grad(
                    delta,
                    [B1_track, x_track, f_bot_t, g_bot_t, f_top_t, g_top_t],
                )
                dD_dB1, dD_dx_direct, dD_df_bot, dD_dg_bot, dD_df_top, dD_dg_top = grads

                dD_dx = (
                    dD_dx_direct
                    + dD_df_bot * dfdx_bot[m]
                    + dD_dg_bot * dgdx_bot[m]
                    + dD_df_top * dfdx_top[m]
                    + dD_dg_top * dgdx_top[m]
                )

                dD_dx_val = dD_dx.item()
                if abs(dD_dx_val) < 1e-30:
                    continue

                ift_factor = -grad_x[m].item() / dD_dx_val
                grad_B1_total = grad_B1_total + ift_factor * dD_dB1

        return (
            None,
            grad_B1_total,
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


def kraken_multilayer_ift(
    x_converged: torch.Tensor,
    B1: torch.Tensor,
    layer_h: torch.Tensor,
    layer_rho: torch.Tensor,
    layer_loc: torch.Tensor,
    layer_n: torch.Tensor,
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
    return KrakenMultiLayerIFT.apply(
        x_converged,
        B1,
        layer_h,
        layer_rho,
        layer_loc,
        layer_n,
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


__all__ = ["KrakenEigenvalueIFT", "kraken_eigenvalue_ift", "kraken_multilayer_ift"]
