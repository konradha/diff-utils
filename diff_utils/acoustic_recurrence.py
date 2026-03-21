import torch

from diff_utils._ext import _cpu_ext


class AcousticRecurrenceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        B1: torch.Tensor,
        h2k2: torch.Tensor,
        loc_start: int,
        loc_end: int,
        p1_init: torch.Tensor,
        p2_init: torch.Tensor,
    ):
        ext = _cpu_ext()
        if ext is None:
            raise RuntimeError("C++ extension required for acoustic_recurrence")
        f_num, g_val, p_history = ext.acoustic_recurrence_fwd(
            B1.detach().contiguous(),
            h2k2.detach().contiguous(),
            loc_start,
            loc_end,
            p1_init.detach().contiguous(),
            p2_init.detach().contiguous(),
        )
        return f_num, g_val, p_history

    @staticmethod
    def setup_context(ctx, inputs, output):
        B1, h2k2, loc_start, loc_end, p1_init, p2_init = inputs
        f_num, g_val, p_history = output
        ctx.loc_start = loc_start
        ctx.loc_end = loc_end
        ctx.save_for_backward(B1, h2k2, p_history, p1_init, p2_init)
        ctx.mark_non_differentiable(p_history)

    @staticmethod
    def backward(ctx, grad_f_num, grad_g_val, grad_p_history):
        B1, h2k2, p_history, p1_init, p2_init = ctx.saved_tensors
        loc_start = ctx.loc_start
        loc_end = ctx.loc_end
        is_complex = h2k2.is_complex()

        ext = _cpu_ext()
        if ext is None:
            raise RuntimeError("C++ extension required for acoustic_recurrence")
        grad_B1, grad_h2k2, grad_p1_init, grad_p2_init = ext.acoustic_recurrence_bwd(
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

        return grad_B1, grad_h2k2, None, None, grad_p1_init, grad_p2_init


def _backward_python(
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
    M = h2k2.shape[0]
    N = B1.shape[0]
    sweep_len = loc_end - loc_start + 1

    def mc(v):
        return v.conj() if is_complex else v

    grad_B1 = torch.zeros(N, dtype=B1.dtype, device=B1.device)
    grad_h2k2 = torch.zeros(M, dtype=h2k2.dtype, device=h2k2.device)
    grad_p1_init = torch.zeros(M, dtype=h2k2.dtype, device=h2k2.device)
    grad_p2_init = torch.zeros(M, dtype=h2k2.dtype, device=h2k2.device)

    for m in range(M):
        hk_m = mc(h2k2[m])  # conjugate fwd coefficient
        d_p2 = -grad_f_num[m]  # not conjugated grad
        d_p1 = -grad_g_val[m]
        d_p0 = grad_f_num[m]

        for s in range(sweep_len - 1, -1, -1):
            jj = loc_end - s
            p1_conj = mc(p_history[m, s])  # conj for grad accumulation
            coeff = hk_m - mc(B1[jj])  # conj coefficient

            d_p1_from_p2 = d_p2 * coeff
            d_p0_from_p2 = -d_p2

            grad_B1[jj] += -d_p2 * p1_conj
            grad_h2k2[m] += d_p2 * p1_conj

            new_d_p2 = d_p1 + d_p1_from_p2
            new_d_p1 = d_p0 + d_p0_from_p2
            d_p2 = new_d_p2
            d_p1 = new_d_p1
            d_p0 = h2k2.new_zeros(())

        grad_p1_init[m] = d_p1 + d_p0  # NOT conjugated
        grad_p2_init[m] = d_p2

    return grad_B1, grad_h2k2, grad_p1_init, grad_p2_init


def acoustic_recurrence(
    B1: torch.Tensor,
    h2k2: torch.Tensor,
    loc_start: int,
    loc_end: int,
    p1_init: torch.Tensor,
    p2_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_num, g_val, _ = AcousticRecurrenceFn.apply(B1, h2k2, loc_start, loc_end, p1_init, p2_init)
    return f_num, g_val


def acoustic_recurrence_nograd(
    B1: torch.Tensor,
    h2k2: torch.Tensor,
    loc_start: int,
    loc_end: int,
    p1_init: torch.Tensor,
    p2_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _cpu_ext()
    if ext is None:
        raise RuntimeError("C++ extension required for acoustic_recurrence")
    return ext.acoustic_recurrence_fwd(
        B1.contiguous(),
        h2k2.contiguous(),
        loc_start,
        loc_end,
        p1_init.contiguous(),
        p2_init.contiguous(),
    )


def acoustic_recurrence_scalar_counted(
    B1: torch.Tensor,
    h2k2: float,
    loc_start: int,
    loc_end: int,
    p1_init: float,
    p2_init: float,
) -> tuple[float, float, int]:
    ext = _cpu_ext()
    if ext is None:
        raise RuntimeError("C++ extension required")
    return ext.acoustic_recurrence_scalar_counted(
        B1.contiguous(),
        h2k2,
        loc_start,
        loc_end,
        p1_init,
        p2_init,
    )


__all__ = [
    "AcousticRecurrenceFn",
    "acoustic_recurrence",
    "acoustic_recurrence_nograd",
    "acoustic_recurrence_scalar_counted",
]
