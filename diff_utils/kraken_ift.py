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

        grad_B1_total = torch.zeros_like(B1)
        is_complex = x_converged.is_complex()

        f_in_layers = [None] * n_layers
        g_in_layers = [None] * n_layers
        h2k2_layers = [None] * n_layers
        p1_init_layers = [None] * n_layers
        p2_init_layers = [None] * n_layers
        p_history_layers = [None] * n_layers

        for m in range(M):
            x_m = x_converged[m].detach()
            f_cur = f_bc_bot[m].detach()
            g_cur = g_bc_bot[m].detach()

            for rev_idx in range(n_layers):
                layer_idx = n_layers - 1 - rev_idx
                h = layer_h[layer_idx]
                rho = layer_rho[layer_idx]
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_end = loc + n_pts - 1

                f_in_layers[layer_idx] = f_cur
                g_in_layers[layer_idx] = g_cur

                h2k2 = h * h * x_m
                p1_init = (-2.0 * g_cur).reshape(1)
                p2_init = ((B1[loc_end].detach() - h2k2) * g_cur - 2.0 * h * f_cur * rho).reshape(1)
                f_num, g_val, p_history = AcousticRecurrenceFn.apply(
                    B1.detach(),
                    h2k2.reshape(1),
                    loc,
                    loc_end - 1,
                    p1_init,
                    p2_init,
                )

                h2k2_layers[layer_idx] = h2k2
                p1_init_layers[layer_idx] = p1_init
                p2_init_layers[layer_idx] = p2_init
                p_history_layers[layer_idx] = p_history

                scale = 2.0 * h * rho
                f_cur = f_num[0] / scale
                g_cur = g_val[0]

            f_top_interior = f_cur
            g_top_interior = g_cur

            grad_f_cur = g_bc_top[m].detach()
            grad_g_cur = -f_bc_top[m].detach()
            dDelta_dx = (
                f_top_interior * dgdx_top[m].detach() - g_top_interior * dfdx_top[m].detach()
            )
            grad_B1_m = torch.zeros_like(B1)

            for layer_idx in range(n_layers):
                h = layer_h[layer_idx]
                rho = layer_rho[layer_idx]
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_end = loc + n_pts - 1
                g_in = g_in_layers[layer_idx]

                grad_f_num = (grad_f_cur / (2.0 * h * rho)).reshape(1)
                grad_g_val = grad_g_cur.reshape(1)
                grad_B1_layer, grad_h2k2_layer, grad_p1i_layer, grad_p2i_layer = (
                    _run_recurrence_bwd(
                        grad_f_num,
                        grad_g_val,
                        B1.detach(),
                        h2k2_layers[layer_idx].reshape(1),
                        p_history_layers[layer_idx],
                        loc,
                        loc_end - 1,
                        p1_init_layers[layer_idx],
                        p2_init_layers[layer_idx],
                        is_complex,
                    )
                )

                grad_B1_m += grad_B1_layer
                grad_B1_m[loc_end] += grad_p2i_layer[0] * g_in

                grad_h2k2_total = grad_h2k2_layer[0] - grad_p2i_layer[0] * g_in
                dDelta_dx = dDelta_dx + grad_h2k2_total * (h * h)

                grad_f_cur = -2.0 * h * rho * grad_p2i_layer[0]
                grad_g_cur = (
                    -2.0 * grad_p1i_layer[0]
                    + (B1[loc_end].detach() - h2k2_layers[layer_idx]) * grad_p2i_layer[0]
                )

            dDelta_dx = (
                dDelta_dx + grad_f_cur * dfdx_bot[m].detach() + grad_g_cur * dgdx_bot[m].detach()
            )

            ift_factor = -grad_x[m] * _lorentz_inv(dDelta_dx, eps)
            grad_B1_total = grad_B1_total + ift_factor * grad_B1_m

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


class KrakenAcousticBottomIFT(torch.autograd.Function):
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
        dfdx_top: torch.Tensor,
        dgdx_top: torch.Tensor,
        c_bot: torch.Tensor,
        rho_bot: torch.Tensor,
        omega2: float,
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
            dfdx_top,
            dgdx_top,
            c_bot,
            rho_bot,
            omega2,
            eps,
        ) = inputs
        ctx.omega2 = omega2
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
            dfdx_top,
            dgdx_top,
            c_bot,
            rho_bot,
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
            dfdx_top,
            dgdx_top,
            c_bot,
            rho_bot,
        ) = ctx.saved_tensors
        omega2 = ctx.omega2
        eps = ctx.eps

        M = x_converged.shape[0]
        n_layers = layer_h.shape[0]
        grad_B1_total = torch.zeros_like(B1)
        grad_c_bot = torch.zeros_like(c_bot)
        grad_rho_bot = torch.zeros_like(rho_bot)
        floor = torch.tensor(1e-30, dtype=x_converged.dtype, device=x_converged.device)

        f_in_layers = [None] * n_layers
        g_in_layers = [None] * n_layers
        h2k2_layers = [None] * n_layers
        p1_init_layers = [None] * n_layers
        p2_init_layers = [None] * n_layers
        p_history_layers = [None] * n_layers

        for m in range(M):
            x_m = x_converged[m].detach()
            gamma2 = x_m - omega2 / (c_bot.detach() * c_bot.detach())
            gamma2_clamped = torch.clamp(gamma2, min=floor)
            f_bot = torch.sqrt(gamma2_clamped)
            g_bot = rho_bot.detach()

            active = gamma2 > floor
            if bool(active):
                dfdx_bot = 0.5 / f_bot
                dfdc_bot = omega2 / (c_bot.detach() * c_bot.detach() * c_bot.detach() * f_bot)
            else:
                dfdx_bot = x_m.new_zeros(())
                dfdc_bot = x_m.new_zeros(())

            f_cur = f_bot
            g_cur = g_bot
            f_top_interior = None
            g_top_interior = None

            for rev_idx in range(n_layers):
                layer_idx = n_layers - 1 - rev_idx
                h = layer_h[layer_idx]
                rho = layer_rho[layer_idx]
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_end = loc + n_pts - 1

                f_in_layers[layer_idx] = f_cur
                g_in_layers[layer_idx] = g_cur

                h2k2 = h * h * x_m
                p1_init = (-2.0 * g_cur).reshape(1)
                p2_init = ((B1[loc_end].detach() - h2k2) * g_cur - 2.0 * h * f_cur * rho).reshape(1)
                f_num, g_val, p_history = AcousticRecurrenceFn.apply(
                    B1.detach(),
                    h2k2.reshape(1),
                    loc,
                    loc_end - 1,
                    p1_init,
                    p2_init,
                )

                h2k2_layers[layer_idx] = h2k2
                p1_init_layers[layer_idx] = p1_init
                p2_init_layers[layer_idx] = p2_init
                p_history_layers[layer_idx] = p_history

                scale = 2.0 * h * rho
                f_cur = f_num[0] / scale
                g_cur = g_val[0]

            f_top_interior = f_cur
            g_top_interior = g_cur

            grad_f_cur = g_bc_top[m].detach()
            grad_g_cur = -f_bc_top[m].detach()
            dDelta_dx = (
                f_top_interior * dgdx_top[m].detach() - g_top_interior * dfdx_top[m].detach()
            )
            grad_B1_m = torch.zeros_like(B1)

            for layer_idx in range(n_layers):
                h = layer_h[layer_idx]
                rho = layer_rho[layer_idx]
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_end = loc + n_pts - 1
                g_in = g_in_layers[layer_idx]

                grad_f_num = (grad_f_cur / (2.0 * h * rho)).reshape(1)
                grad_g_val = grad_g_cur.reshape(1)

                grad_B1_layer, grad_h2k2_layer, grad_p1i_layer, grad_p2i_layer = (
                    _run_recurrence_bwd(
                        grad_f_num,
                        grad_g_val,
                        B1.detach(),
                        h2k2_layers[layer_idx].reshape(1),
                        p_history_layers[layer_idx],
                        loc,
                        loc_end - 1,
                        p1_init_layers[layer_idx],
                        p2_init_layers[layer_idx],
                        False,
                    )
                )

                grad_B1_m += grad_B1_layer
                grad_B1_m[loc_end] += grad_p2i_layer[0] * g_in

                grad_h2k2_total = grad_h2k2_layer[0] - grad_p2i_layer[0] * g_in
                dDelta_dx = dDelta_dx + grad_h2k2_total * (h * h)

                grad_f_cur = -2.0 * h * rho * grad_p2i_layer[0]
                grad_g_cur = (
                    -2.0 * grad_p1i_layer[0]
                    + (B1[loc_end].detach() - h2k2_layers[layer_idx]) * grad_p2i_layer[0]
                )

            dDelta_dx = dDelta_dx + grad_f_cur * dfdx_bot
            dDelta_dcb = grad_f_cur * dfdc_bot
            dDelta_drb = grad_g_cur

            ift_scale = -grad_x[m] * _lorentz_inv(dDelta_dx, eps)
            grad_B1_total = grad_B1_total + ift_scale * grad_B1_m
            grad_c_bot = grad_c_bot + ift_scale * dDelta_dcb
            grad_rho_bot = grad_rho_bot + ift_scale * dDelta_drb

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
            grad_c_bot,
            grad_rho_bot,
            None,
            None,
        )


def kraken_acoustic_bottom_ift(
    x_converged: torch.Tensor,
    B1: torch.Tensor,
    layer_h: torch.Tensor,
    layer_rho: torch.Tensor,
    layer_loc: torch.Tensor,
    layer_n: torch.Tensor,
    f_bc_top: torch.Tensor,
    g_bc_top: torch.Tensor,
    dfdx_top: torch.Tensor,
    dgdx_top: torch.Tensor,
    c_bot: torch.Tensor,
    rho_bot: torch.Tensor,
    omega2: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    return KrakenAcousticBottomIFT.apply(
        x_converged,
        B1,
        layer_h,
        layer_rho,
        layer_loc,
        layer_n,
        f_bc_top,
        g_bc_top,
        dfdx_top,
        dgdx_top,
        c_bot,
        rho_bot,
        omega2,
        eps,
    )


class KrakencVacuumAcousticBottomIFT(torch.autograd.Function):
    @staticmethod
    def forward(
        x_converged: torch.Tensor,
        B1: torch.Tensor,
        layer_h: torch.Tensor,
        layer_rho: torch.Tensor,
        layer_loc: torch.Tensor,
        layer_n: torch.Tensor,
        branch_signs: torch.Tensor,
        c_bot: torch.Tensor,
        rho_bot: torch.Tensor,
        omega2: float,
        c_imag: float,
        dc_imag_dc: float,
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
            branch_signs,
            c_bot,
            rho_bot,
            omega2,
            c_imag,
            dc_imag_dc,
            eps,
        ) = inputs
        ctx.omega2 = omega2
        ctx.c_imag = c_imag
        ctx.dc_imag_dc = dc_imag_dc
        ctx.eps = eps
        ctx.save_for_backward(
            x_converged,
            B1,
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
            branch_signs,
            c_bot,
            rho_bot,
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
            branch_signs,
            c_bot,
            rho_bot,
        ) = ctx.saved_tensors
        omega2 = ctx.omega2
        c_imag = ctx.c_imag
        dc_imag_dc = ctx.dc_imag_dc
        eps = ctx.eps

        M = x_converged.shape[0]
        n_layers = layer_h.shape[0]
        B1_c = B1.detach().to(torch.complex128)

        grad_B1_total = torch.zeros_like(B1)
        grad_c_bot = torch.zeros_like(c_bot)
        grad_rho_bot = torch.zeros_like(rho_bot)

        f_in_layers = [None] * n_layers
        g_in_layers = [None] * n_layers
        h2k2_layers = [None] * n_layers
        p1_init_layers = [None] * n_layers
        p2_init_layers = [None] * n_layers
        p_history_layers = [None] * n_layers

        cp_c = torch.complex(
            c_bot.detach(), torch.tensor(c_imag, dtype=c_bot.dtype, device=c_bot.device)
        )
        dcp_dc = torch.complex(
            torch.ones((), dtype=c_bot.dtype, device=c_bot.device),
            torch.tensor(dc_imag_dc, dtype=c_bot.dtype, device=c_bot.device),
        )

        for m in range(M):
            x_m = x_converged[m].detach().to(torch.complex128)
            branch_sign = branch_signs[m].detach().to(torch.complex128)
            gamma2 = x_m - omega2 / (cp_c * cp_c)
            f_sqrt = torch.sqrt(gamma2)
            f_bot = branch_sign * f_sqrt
            g_bot = rho_bot.detach().to(torch.complex128)
            dfdx_bot = branch_sign * 0.5 / f_sqrt
            dfdcp = branch_sign * omega2 / (cp_c * cp_c * cp_c * f_sqrt)
            dfdc_bot = dfdcp * dcp_dc

            f_cur = f_bot
            g_cur = g_bot

            for rev_idx in range(n_layers):
                layer_idx = n_layers - 1 - rev_idx
                h = layer_h[layer_idx]
                rho = layer_rho[layer_idx]
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_end = loc + n_pts - 1

                f_in_layers[layer_idx] = f_cur
                g_in_layers[layer_idx] = g_cur

                h2k2 = (h * h * x_m).reshape(1)
                p1_init = (-2.0 * g_cur).reshape(1)
                p2_init = ((B1_c[loc_end] - h2k2[0]) * g_cur - 2.0 * h * f_cur * rho).reshape(1)
                f_num, g_val, p_history = AcousticRecurrenceFn.apply(
                    B1_c,
                    h2k2,
                    loc,
                    loc_end - 1,
                    p1_init,
                    p2_init,
                )

                h2k2_layers[layer_idx] = h2k2
                p1_init_layers[layer_idx] = p1_init
                p2_init_layers[layer_idx] = p2_init
                p_history_layers[layer_idx] = p_history

                f_cur = f_num[0] / (2.0 * h * rho)
                g_cur = g_val[0]

            grad_f_cur = x_m.new_zeros(())
            grad_g_cur = -torch.ones((), dtype=torch.complex128, device=x_m.device)
            dDelta_dx = x_m.new_zeros(())
            grad_B1_delta = torch.zeros_like(B1_c)

            for layer_idx in range(n_layers):
                h = layer_h[layer_idx]
                rho = layer_rho[layer_idx]
                loc = int(layer_loc[layer_idx].item())
                n_pts = int(layer_n[layer_idx].item())
                loc_end = loc + n_pts - 1
                g_in = g_in_layers[layer_idx]

                grad_f_num = (grad_f_cur / (2.0 * h * rho)).reshape(1)
                grad_g_val = grad_g_cur.reshape(1)

                grad_B1_layer, grad_h2k2_layer, grad_p1i_layer, grad_p2i_layer = (
                    _run_recurrence_bwd(
                        grad_f_num,
                        grad_g_val,
                        B1_c,
                        h2k2_layers[layer_idx],
                        p_history_layers[layer_idx],
                        loc,
                        loc_end - 1,
                        p1_init_layers[layer_idx],
                        p2_init_layers[layer_idx],
                        True,
                    )
                )

                grad_B1_delta += grad_B1_layer
                grad_B1_delta[loc_end] += grad_p2i_layer[0] * g_in

                grad_h2k2_total = grad_h2k2_layer[0] - grad_p2i_layer[0] * g_in
                dDelta_dx = dDelta_dx + grad_h2k2_total * (h * h)

                grad_f_cur = -2.0 * h * rho * grad_p2i_layer[0]
                grad_g_cur = (
                    -2.0 * grad_p1i_layer[0]
                    + (B1_c[loc_end] - h2k2_layers[layer_idx][0]) * grad_p2i_layer[0]
                )

            dDelta_dx = dDelta_dx + grad_f_cur * dfdx_bot
            dDelta_dc = grad_f_cur * dfdc_bot
            dDelta_drho = grad_g_cur

            ift_scale = -grad_x[m].conj() * _lorentz_inv(dDelta_dx, eps)
            grad_B1_total = grad_B1_total + (ift_scale * grad_B1_delta).real.to(B1.dtype)
            grad_c_bot = grad_c_bot + (ift_scale * dDelta_dc).real.to(c_bot.dtype)
            grad_rho_bot = grad_rho_bot + (ift_scale * dDelta_drho).real.to(rho_bot.dtype)

        return (
            None,
            grad_B1_total,
            None,
            None,
            None,
            None,
            None,
            grad_c_bot,
            grad_rho_bot,
            None,
            None,
            None,
            None,
        )


def krakenc_vacuum_acoustic_bottom_ift(
    x_converged: torch.Tensor,
    B1: torch.Tensor,
    layer_h: torch.Tensor,
    layer_rho: torch.Tensor,
    layer_loc: torch.Tensor,
    layer_n: torch.Tensor,
    branch_signs: torch.Tensor,
    c_bot: torch.Tensor,
    rho_bot: torch.Tensor,
    omega2: float,
    c_imag: float,
    dc_imag_dc: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    return KrakencVacuumAcousticBottomIFT.apply(
        x_converged,
        B1,
        layer_h,
        layer_rho,
        layer_loc,
        layer_n,
        branch_signs,
        c_bot,
        rho_bot,
        omega2,
        c_imag,
        dc_imag_dc,
        eps,
    )


__all__ = [
    "KrakenEigenvalueIFT",
    "kraken_eigenvalue_ift",
    "kraken_multilayer_ift",
    "kraken_acoustic_bottom_ift",
    "krakenc_vacuum_acoustic_bottom_ift",
]
