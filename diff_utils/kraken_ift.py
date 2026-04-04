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


def _ift_forward_sweep(x_m, B1_c, n_layers, layer_h, layer_rho, layer_loc, layer_n, f_bot, g_bot):
    f_in = [None] * n_layers
    g_in = [None] * n_layers
    h2k2_l = [None] * n_layers
    p1i_l = [None] * n_layers
    p2i_l = [None] * n_layers
    ph_l = [None] * n_layers

    f_cur, g_cur = f_bot, g_bot
    for rev_idx in range(n_layers):
        li = n_layers - 1 - rev_idx
        h = layer_h[li]
        rho = layer_rho[li]
        loc = int(layer_loc[li].item())
        n_pts = int(layer_n[li].item())
        loc_end = loc + n_pts - 1

        f_in[li] = f_cur
        g_in[li] = g_cur

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
        h2k2_l[li] = h2k2
        p1i_l[li] = p1_init
        p2i_l[li] = p2_init
        ph_l[li] = p_history

        f_cur = f_num[0] / (2.0 * h * rho)
        g_cur = g_val[0]

    return f_cur, g_cur, f_in, g_in, h2k2_l, p1i_l, p2i_l, ph_l


def _ift_forward_sweep_batch(
    x_batch, B1_c, n_layers, layer_h, layer_rho, layer_loc, layer_n, f_bot, g_bot
):
    M = x_batch.shape[0]
    h2k2_l = [None] * n_layers
    p1i_l = [None] * n_layers
    p2i_l = [None] * n_layers
    ph_l = [None] * n_layers
    g_in = [None] * n_layers

    f_cur, g_cur = f_bot, g_bot  # [M] tensors

    for rev_idx in range(n_layers):
        li = n_layers - 1 - rev_idx
        h = layer_h[li]
        rho = layer_rho[li]
        loc = int(layer_loc[li].item())
        n_pts = int(layer_n[li].item())
        loc_end = loc + n_pts - 1

        g_in[li] = g_cur

        h2k2 = h * h * x_batch  # [M]
        p1_init = -2.0 * g_cur  # [M]
        p2_init = (B1_c[loc_end] - h2k2) * g_cur - 2.0 * h * f_cur * rho  # [M]
        f_num, g_val, p_history = AcousticRecurrenceFn.apply(
            B1_c,
            h2k2,
            loc,
            loc_end - 1,
            p1_init,
            p2_init,
        )
        h2k2_l[li] = h2k2
        p1i_l[li] = p1_init
        p2i_l[li] = p2_init
        ph_l[li] = p_history  # [M, sweep_len]

        f_cur = f_num / (2.0 * h * rho)  # [M]
        g_cur = g_val  # [M]

    return f_cur, g_cur, g_in, h2k2_l, p1i_l, p2i_l, ph_l


def _ift_backward_sweep(
    B1_c,
    n_layers,
    layer_h,
    layer_rho,
    layer_loc,
    layer_n,
    g_in,
    h2k2_l,
    p1i_l,
    p2i_l,
    ph_l,
    is_complex,
    f_top=None,
    g_top=None,
):
    device = B1_c.device
    dtype = torch.complex128 if is_complex else torch.float64
    # Delta = f * g_top - g * f_top → dDelta/df = g_top, dDelta/dg = -f_top
    if f_top is None:
        grad_f_cur = torch.zeros((), dtype=dtype, device=device)
        grad_g_cur = -torch.ones((), dtype=dtype, device=device)
    else:
        grad_f_cur = g_top.to(dtype)
        grad_g_cur = -f_top.to(dtype)
    dDelta_dx = torch.zeros((), dtype=dtype, device=device)
    grad_B1_delta = torch.zeros_like(B1_c)

    for li in range(n_layers):
        h = layer_h[li]
        rho = layer_rho[li]
        loc = int(layer_loc[li].item())
        n_pts = int(layer_n[li].item())
        loc_end = loc + n_pts - 1

        grad_f_num = (grad_f_cur / (2.0 * h * rho)).reshape(1)
        grad_g_val = grad_g_cur.reshape(1)

        grad_B1_layer, grad_h2k2_layer, grad_p1i_layer, grad_p2i_layer = _run_recurrence_bwd(
            grad_f_num,
            grad_g_val,
            B1_c,
            h2k2_l[li],
            ph_l[li],
            loc,
            loc_end - 1,
            p1i_l[li],
            p2i_l[li],
            False,
        )

        grad_B1_delta += grad_B1_layer
        grad_B1_delta[loc_end] += grad_p2i_layer[0] * g_in[li]

        grad_h2k2_total = grad_h2k2_layer[0] - grad_p2i_layer[0] * g_in[li]
        dDelta_dx = dDelta_dx + grad_h2k2_total * (h * h)

        grad_f_cur = -2.0 * h * rho * grad_p2i_layer[0]
        grad_g_cur = -2.0 * grad_p1i_layer[0] + (B1_c[loc_end] - h2k2_l[li][0]) * grad_p2i_layer[0]

    return grad_f_cur, grad_g_cur, dDelta_dx, grad_B1_delta


def _ift_backward_sweep_batch(
    B1_c,
    n_layers,
    layer_h,
    layer_rho,
    layer_loc,
    layer_n,
    g_in,
    h2k2_l,
    p1i_l,
    p2i_l,
    ph_l,
    M,
):
    device = B1_c.device
    dtype = torch.complex128
    grad_f_cur = torch.zeros(M, dtype=dtype, device=device)
    grad_g_cur = -torch.ones(M, dtype=dtype, device=device)
    dDelta_dx = torch.zeros(M, dtype=dtype, device=device)
    grad_B1_delta = torch.zeros_like(B1_c)

    for li in range(n_layers):
        h = layer_h[li]
        rho = layer_rho[li]
        loc = int(layer_loc[li].item())
        n_pts = int(layer_n[li].item())
        loc_end = loc + n_pts - 1

        grad_f_num = grad_f_cur / (2.0 * h * rho)  # [M]
        grad_g_val = grad_g_cur  # [M]

        grad_B1_layer, grad_h2k2_layer, grad_p1i_layer, grad_p2i_layer = _run_recurrence_bwd(
            grad_f_num,
            grad_g_val,
            B1_c,
            h2k2_l[li],
            ph_l[li],
            loc,
            loc_end - 1,
            p1i_l[li],
            p2i_l[li],
            False,
        )

        grad_B1_delta += grad_B1_layer
        grad_B1_delta[loc_end] += (grad_p2i_layer * g_in[li]).sum()

        grad_h2k2_total = grad_h2k2_layer - grad_p2i_layer * g_in[li]  # [M]
        dDelta_dx = dDelta_dx + grad_h2k2_total * (h * h)

        grad_f_cur = -2.0 * h * rho * grad_p2i_layer  # [M]
        grad_g_cur = -2.0 * grad_p1i_layer + (B1_c[loc_end] - h2k2_l[li]) * grad_p2i_layer

    return grad_f_cur, grad_g_cur, dDelta_dx, grad_B1_delta


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
        f_top_vals: torch.Tensor | None = None,
        g_top_vals: torch.Tensor | None = None,
        c_top: torch.Tensor | None = None,
        rho_top: torch.Tensor | None = None,
        cs_bot: torch.Tensor | None = None,
        c_imag_top: float = 0.0,
        dc_imag_dc_top: float = 0.0,
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
            f_top_vals,
            g_top_vals,
            c_top,
            rho_top,
            cs_bot,
            c_imag_top,
            dc_imag_dc_top,
        ) = inputs
        ctx.omega2 = omega2
        ctx.c_imag = c_imag
        ctx.dc_imag_dc = dc_imag_dc
        ctx.c_imag_top = c_imag_top
        ctx.dc_imag_dc_top = dc_imag_dc_top
        ctx.eps = eps
        ctx.has_top_bc = f_top_vals is not None
        ctx.has_c_top = c_top is not None
        ctx.has_rho_top = rho_top is not None
        ctx.has_cs_bot = cs_bot is not None
        tensors = [
            x_converged,
            B1,
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
            branch_signs,
            c_bot,
            rho_bot,
        ]
        if f_top_vals is not None:
            tensors.extend([f_top_vals, g_top_vals])
        if c_top is not None:
            tensors.append(c_top)
        if rho_top is not None:
            tensors.append(rho_top)
        if cs_bot is not None:
            tensors.append(cs_bot)
        ctx.save_for_backward(*tensors)

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
            *rest,
        ) = ctx.saved_tensors
        # Unpack optional saved tensors in order they were appended
        idx = 0
        f_top_vals = g_top_vals = c_top_t = rho_top_t = cs_bot_t = None
        if ctx.has_top_bc:
            f_top_vals, g_top_vals = rest[idx], rest[idx + 1]
            idx += 2
        if ctx.has_c_top:
            c_top_t = rest[idx]
            idx += 1
        if ctx.has_rho_top:
            rho_top_t = rest[idx]
            idx += 1
        if ctx.has_cs_bot:
            cs_bot_t = rest[idx]
            idx += 1
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
        grad_c_top = torch.zeros((), dtype=c_bot.dtype) if c_top_t is not None else None
        grad_rho_top = torch.zeros((), dtype=c_bot.dtype) if rho_top_t is not None else None
        grad_cs_bot = torch.zeros((), dtype=c_bot.dtype) if cs_bot_t is not None else None

        cp_c = torch.complex(
            c_bot.detach(), torch.tensor(c_imag, dtype=c_bot.dtype, device=c_bot.device)
        )
        dcp_dc = torch.complex(
            torch.ones((), dtype=c_bot.dtype, device=c_bot.device),
            torch.tensor(dc_imag_dc, dtype=c_bot.dtype, device=c_bot.device),
        )

        x_c = x_converged.detach().to(torch.complex128)  # [M]
        bs_c = branch_signs.detach().to(torch.complex128)  # [M]
        gamma2 = x_c - omega2 / (cp_c * cp_c)  # [M]
        f_sqrt = torch.sqrt(gamma2)  # [M]
        f_bot_all = bs_c * f_sqrt  # [M]
        g_bot_val = rho_bot.detach().to(torch.complex128)
        dfdx_bot = bs_c * 0.5 / f_sqrt  # [M]
        dfdc_bot = bs_c * omega2 / (cp_c**3 * f_sqrt) * dcp_dc  # [M]

        # Batched forward sweep — single C++ call per layer for all M modes
        f_at_top, g_at_top, g_in, h2k2_l, p1i_l, p2i_l, ph_l = _ift_forward_sweep_batch(
            x_c,
            B1_c,
            n_layers,
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
            f_bot_all,
            g_bot_val.expand(M),
        )

        # Per-mode backward (B1 gradient needs per-mode ift_scale)
        for m in range(M):
            g_in_m = [g[m] for g in g_in]
            h2k2_m = [h[m : m + 1] for h in h2k2_l]
            p1i_m = [p[m : m + 1] for p in p1i_l]
            p2i_m = [p[m : m + 1] for p in p2i_l]
            ph_m = [p[m : m + 1] for p in ph_l]

            ft_m = f_top_vals[m] if f_top_vals is not None else None
            gt_m = g_top_vals[m] if g_top_vals is not None else None
            grad_f_bot_m, grad_g_bot_m, dDelta_dx_m, grad_B1_delta = _ift_backward_sweep(
                B1_c,
                n_layers,
                layer_h,
                layer_rho,
                layer_loc,
                layer_n,
                g_in_m,
                h2k2_m,
                p1i_m,
                p2i_m,
                ph_m,
                True,
                f_top=ft_m,
                g_top=gt_m,
            )

            dDelta_dx_m = dDelta_dx_m + grad_f_bot_m * dfdx_bot[m]
            dDelta_dc = grad_f_bot_m * dfdc_bot[m]
            dDelta_drho = grad_g_bot_m

            # Top BC parameter gradients: dDelta/dc_top, dDelta/drho_top
            # Delta = f_top_recur * g_top_bc - g_top_recur * f_top_bc
            # dDelta/df_top_bc = -g_top_recur (at top of recurrence, before BC)
            # dDelta/dg_top_bc = f_top_recur
            # Wait: the forward sweep computes f,g from bottom to top.
            # f_at_top, g_at_top are BEFORE top BC application.
            # Delta = f_at_top * g_top_bc - g_at_top * f_top_bc
            # So: dDelta/df_top_bc = -g_at_top, dDelta/dg_top_bc = f_at_top
            if c_top_t is not None or rho_top_t is not None:
                f_t = f_at_top[m].detach()
                g_t = g_at_top[m].detach()
                # For acoustic top: f_top = sqrt(gamma2_top), g_top = -rho_top
                # df_top/dc_top = omega2/(cp_top_c^3 * sqrt(gamma2_top)) * dcp_dc_top
                if c_top_t is not None:
                    cp_top_c = torch.complex(
                        c_top_t.detach().to(torch.float64),
                        torch.tensor(ctx.c_imag_top, dtype=torch.float64),
                    )
                    dcp_dc_top = torch.complex(
                        torch.ones((), dtype=torch.float64),
                        torch.tensor(ctx.dc_imag_dc_top, dtype=torch.float64),
                    )
                    gamma2_top = x_c[m] - omega2 / (cp_top_c * cp_top_c)
                    f_sqrt_top = torch.sqrt(gamma2_top)
                    dfdc_top = omega2 / (cp_top_c**3 * f_sqrt_top) * dcp_dc_top
                    # dDelta/dc_top = dDelta/df_top * df_top/dc_top
                    # dDelta/df_top = -g_at_top (since Delta = f_at_top * g_top - g_at_top * f_top)
                    dDelta_dc_top_m = -g_t * dfdc_top
                    dDelta_dx_m = dDelta_dx_m + (-g_t) * (0.5 / f_sqrt_top)  # top dfdx contribution
                if rho_top_t is not None:
                    # g_top = -rho_top → dg_top/drho_top = -1
                    # dDelta/drho_top = dDelta/dg_top * dg_top/drho_top = f_at_top * (-1)
                    dDelta_drho_top_m = -f_t

            # cs_bot gradient via autograd
            if cs_bot_t is not None:
                with torch.enable_grad():
                    cs_ad = cs_bot_t.detach().to(torch.complex128).requires_grad_(True)
                    cp_ad = cp_c.clone().requires_grad_(False)
                    rho_ad = rho_bot.detach().to(torch.complex128)
                    f_el, g_el = _elastic_bc(x_c[m], omega2, cp_ad, cs_ad, rho_ad)
                    ones = torch.ones_like(f_el)
                    dfdcs = torch.autograd.grad(f_el, cs_ad, grad_outputs=ones, retain_graph=True)[
                        0
                    ]
                    dgdcs = torch.autograd.grad(g_el, cs_ad, grad_outputs=ones)[0]
                dDelta_dcs_m = grad_f_bot_m * dfdcs.detach() + grad_g_bot_m * dgdcs.detach()

            ift_scale = -grad_x[m].conj() * _lorentz_inv(dDelta_dx_m, eps)
            grad_B1_total = grad_B1_total + (ift_scale * grad_B1_delta).real.to(B1.dtype)
            grad_c_bot = grad_c_bot + (ift_scale * dDelta_dc).real.to(c_bot.dtype)
            grad_rho_bot = grad_rho_bot + (ift_scale * dDelta_drho).real.to(rho_bot.dtype)
            if grad_c_top is not None and c_top_t is not None:
                grad_c_top = grad_c_top + (ift_scale * dDelta_dc_top_m).real.to(c_bot.dtype)
            if grad_rho_top is not None and rho_top_t is not None:
                grad_rho_top = grad_rho_top + (ift_scale * dDelta_drho_top_m).real.to(c_bot.dtype)
            if grad_cs_bot is not None and cs_bot_t is not None:
                grad_cs_bot = grad_cs_bot + (ift_scale * dDelta_dcs_m).real.to(c_bot.dtype)

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
            None,
            None,  # f_top_vals, g_top_vals
            grad_c_top,
            grad_rho_top,
            grad_cs_bot,
            None,
            None,  # c_imag_top, dc_imag_dc_top
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
    f_top_vals: torch.Tensor | None = None,
    g_top_vals: torch.Tensor | None = None,
    c_top: torch.Tensor | None = None,
    rho_top: torch.Tensor | None = None,
    cs_bot: torch.Tensor | None = None,
    c_imag_top: float = 0.0,
    dc_imag_dc_top: float = 0.0,
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
        f_top_vals,
        g_top_vals,
        c_top,
        rho_top,
        cs_bot,
        c_imag_top,
        dc_imag_dc_top,
    )


def _elastic_bc(x, omega2, cp_c, cs_c, rho):
    """Elastic halfspace BC: returns (f, g) as differentiable torch scalars."""
    gammaS2 = x - omega2 / (cs_c * cs_c)
    gammaP2 = x - omega2 / (cp_c * cp_c)
    gammaS = torch.sqrt(gammaS2)
    gammaP = torch.sqrt(gammaP2)
    mu = rho * cs_c * cs_c
    f = omega2 * gammaP * (x - gammaS2)
    g = ((gammaS2 + x) ** 2 - 4.0 * gammaS * gammaP * x) * mu
    return f, g


class KrakencVacuumElasticBottomIFT(torch.autograd.Function):
    @staticmethod
    def forward(
        x_converged: torch.Tensor,
        B1: torch.Tensor,
        layer_h: torch.Tensor,
        layer_rho: torch.Tensor,
        layer_loc: torch.Tensor,
        layer_n: torch.Tensor,
        c_bot: torch.Tensor,
        cs_bot: torch.Tensor,
        rho_bot: torch.Tensor,
        omega2: float,
        c_imag: float,
        dc_imag_dc: float,
        cs_imag: float,
        dcs_imag_dcs: float,
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
            c_bot,
            cs_bot,
            rho_bot,
            omega2,
            c_imag,
            dc_imag_dc,
            cs_imag,
            dcs_imag_dcs,
            eps,
        ) = inputs
        ctx.omega2 = omega2
        ctx.c_imag = c_imag
        ctx.dc_imag_dc = dc_imag_dc
        ctx.cs_imag = cs_imag
        ctx.dcs_imag_dcs = dcs_imag_dcs
        ctx.eps = eps
        ctx.save_for_backward(
            x_converged,
            B1,
            layer_h,
            layer_rho,
            layer_loc,
            layer_n,
            c_bot,
            cs_bot,
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
            c_bot,
            cs_bot,
            rho_bot,
        ) = ctx.saved_tensors
        omega2 = ctx.omega2
        eps = ctx.eps

        M = x_converged.shape[0]
        n_layers = layer_h.shape[0]
        B1_c = B1.detach().to(torch.complex128)

        grad_B1_total = torch.zeros_like(B1)
        grad_c_bot = torch.zeros_like(c_bot)
        grad_cs_bot = torch.zeros_like(cs_bot)
        grad_rho_bot = torch.zeros_like(rho_bot)

        # Complex bottom BC parameters
        cp_c = torch.complex(
            c_bot.detach().to(torch.float64),
            torch.tensor(ctx.c_imag, dtype=torch.float64),
        )
        cs_c = torch.complex(
            cs_bot.detach().to(torch.float64),
            torch.tensor(ctx.cs_imag, dtype=torch.float64),
        )
        rho_c = rho_bot.detach().to(torch.complex128)

        # Derivatives of complex sound speeds w.r.t. real parameters
        dcp_dc = torch.complex(
            torch.ones((), dtype=torch.float64),
            torch.tensor(ctx.dc_imag_dc, dtype=torch.float64),
        )
        dcs_dcs = torch.complex(
            torch.ones((), dtype=torch.float64),
            torch.tensor(ctx.dcs_imag_dcs, dtype=torch.float64),
        )

        for m in range(M):
            x_m = x_converged[m].detach().to(torch.complex128)

            # Compute elastic BC and derivatives using autograd.
            # Need enable_grad() because we're inside backward() which disables grad.
            with torch.enable_grad():
                x_ad = x_m.clone().requires_grad_(True)
                cp_ad = cp_c.clone().requires_grad_(True)
                cs_ad = cs_c.clone().requires_grad_(True)
                rho_ad = rho_c.clone().requires_grad_(True)

                f_bot, g_bot = _elastic_bc(x_ad, omega2, cp_ad, cs_ad, rho_ad)
                ones = torch.ones_like(f_bot)

                # df/dx, dg/dx (holomorphic => use grad_outputs=1, not conjugate)
                dfdx = torch.autograd.grad(f_bot, x_ad, grad_outputs=ones, retain_graph=True)[0]
                dgdx = torch.autograd.grad(g_bot, x_ad, grad_outputs=ones, retain_graph=True)[0]

                # df/dcp, dg/dcp (chain rule: d/dc_bot = d/dcp_c * dcp_dc)
                dfdcp = torch.autograd.grad(f_bot, cp_ad, grad_outputs=ones, retain_graph=True)[0]
                dgdcp = torch.autograd.grad(g_bot, cp_ad, grad_outputs=ones, retain_graph=True)[0]
                dfdc_bot = dfdcp * dcp_dc
                dgdc_bot = dgdcp * dcp_dc

                # df/dcs, dg/dcs
                dfdcs = torch.autograd.grad(f_bot, cs_ad, grad_outputs=ones, retain_graph=True)[0]
                dgdcs = torch.autograd.grad(g_bot, cs_ad, grad_outputs=ones, retain_graph=True)[0]
                dfdcs_bot = dfdcs * dcs_dcs
                dgdcs_bot = dgdcs * dcs_dcs

                # df/drho, dg/drho (f doesn't depend on rho, so allow_unused)
                dfdrho_t = torch.autograd.grad(
                    f_bot, rho_ad, grad_outputs=ones, retain_graph=True, allow_unused=True
                )[0]
                dfdrho = dfdrho_t if dfdrho_t is not None else torch.zeros_like(rho_ad)
                dgdrho = torch.autograd.grad(g_bot, rho_ad, grad_outputs=ones)[0]

            _, _, f_in, g_in, h2k2_l, p1i_l, p2i_l, ph_l = _ift_forward_sweep(
                x_m,
                B1_c,
                n_layers,
                layer_h,
                layer_rho,
                layer_loc,
                layer_n,
                f_bot.detach(),
                g_bot.detach(),
            )
            grad_f_bot, grad_g_bot, dDelta_dx, grad_B1_delta = _ift_backward_sweep(
                B1_c,
                n_layers,
                layer_h,
                layer_rho,
                layer_loc,
                layer_n,
                g_in,
                h2k2_l,
                p1i_l,
                p2i_l,
                ph_l,
                True,
            )

            # Add BC derivatives: dDelta/dx from BC
            dDelta_dx = dDelta_dx + grad_f_bot * dfdx.detach() + grad_g_bot * dgdx.detach()

            # Parameter derivatives
            dDelta_dc = grad_f_bot * dfdc_bot.detach() + grad_g_bot * dgdc_bot.detach()
            dDelta_dcs = grad_f_bot * dfdcs_bot.detach() + grad_g_bot * dgdcs_bot.detach()
            dDelta_drho = grad_f_bot * dfdrho.detach() + grad_g_bot * dgdrho.detach()

            ift_scale = -grad_x[m].conj() * _lorentz_inv(dDelta_dx, eps)
            grad_B1_total = grad_B1_total + (ift_scale * grad_B1_delta).real.to(B1.dtype)
            grad_c_bot = grad_c_bot + (ift_scale * dDelta_dc).real.to(c_bot.dtype)
            grad_cs_bot = grad_cs_bot + (ift_scale * dDelta_dcs).real.to(cs_bot.dtype)
            grad_rho_bot = grad_rho_bot + (ift_scale * dDelta_drho).real.to(rho_bot.dtype)

        return (
            None,  # x_converged
            grad_B1_total,
            None,
            None,
            None,
            None,  # layer_h, layer_rho, layer_loc, layer_n
            grad_c_bot,
            grad_cs_bot,
            grad_rho_bot,
            None,
            None,
            None,
            None,
            None,
            None,  # omega2, c_imag, dc_imag_dc, cs_imag, dcs_imag_dcs, eps
        )


def krakenc_vacuum_elastic_bottom_ift(
    x_converged,
    B1,
    layer_h,
    layer_rho,
    layer_loc,
    layer_n,
    c_bot,
    cs_bot,
    rho_bot,
    omega2,
    c_imag,
    dc_imag_dc,
    cs_imag,
    dcs_imag_dcs,
    *,
    eps=1e-12,
):
    return KrakencVacuumElasticBottomIFT.apply(
        x_converged,
        B1,
        layer_h,
        layer_rho,
        layer_loc,
        layer_n,
        c_bot,
        cs_bot,
        rho_bot,
        omega2,
        c_imag,
        dc_imag_dc,
        cs_imag,
        dcs_imag_dcs,
        eps,
    )


# Backward-compatible aliases
eigenvalue_ift = None  # imported from eigenvalue_ift module

__all__ = [
    "KrakenEigenvalueIFT",
    "kraken_eigenvalue_ift",
    "kraken_multilayer_ift",
    "kraken_acoustic_bottom_ift",
    "krakenc_vacuum_acoustic_bottom_ift",
    "krakenc_vacuum_elastic_bottom_ift",
]
