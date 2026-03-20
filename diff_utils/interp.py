from __future__ import annotations

import torch

from diff_utils._ext import _cpu_ext


class SearchsortedLerpFn(torch.autograd.Function):
    @staticmethod
    def forward(z_knots: torch.Tensor, values: torch.Tensor, z_query: torch.Tensor):
        ext = _cpu_ext()
        if ext is None:
            raise RuntimeError("C++ extension required for searchsorted_lerp")
        out, idx, weights = ext.searchsorted_lerp_fwd(
            z_knots.detach().contiguous().to(torch.float64),
            values.detach().contiguous(),
            z_query.detach().contiguous().to(torch.float64),
        )
        return out, idx, weights

    @staticmethod
    def setup_context(ctx, inputs, output):
        z_knots, values, z_query = inputs
        out, idx, weights = output
        ctx.n_knots = z_knots.shape[0]
        ctx.save_for_backward(idx, weights)
        ctx.mark_non_differentiable(idx, weights)

    @staticmethod
    def backward(ctx, grad_out, grad_idx, grad_weights):
        idx, weights = ctx.saved_tensors
        ext = _cpu_ext()
        if ext is None:
            raise RuntimeError("C++ extension required for searchsorted_lerp")
        grad_values = ext.searchsorted_lerp_bwd(grad_out.contiguous(), idx, weights, ctx.n_knots)
        return None, grad_values, None


def searchsorted_lerp(
    z_knots: torch.Tensor,
    values: torch.Tensor,
    z_query: torch.Tensor,
) -> torch.Tensor:
    out, _, _ = SearchsortedLerpFn.apply(z_knots, values, z_query)
    return out


__all__ = ["SearchsortedLerpFn", "searchsorted_lerp"]
