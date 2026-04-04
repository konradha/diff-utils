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


class _InterpBatchFn(torch.autograd.Function):
    @staticmethod
    def forward(values_batch, idx_lo, idx_hi, weights):
        left = values_batch[:, idx_lo]
        right = values_batch[:, idx_hi]
        return left + (right - left) * weights.unsqueeze(0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        values_batch, idx_lo, idx_hi, weights = inputs
        ctx.N = values_batch.shape[1]
        ctx.save_for_backward(idx_lo, idx_hi, weights)

    @staticmethod
    def backward(ctx, grad_out):
        idx_lo, idx_hi, weights = ctx.saved_tensors
        M, Q = grad_out.shape
        w = weights.unsqueeze(0).to(grad_out.dtype)
        grad_left = grad_out * (1.0 - w)
        grad_right = grad_out * w
        grad_values = grad_out.new_zeros(M, ctx.N)
        grad_values.scatter_add_(1, idx_lo.unsqueeze(0).expand(M, -1), grad_left)
        grad_values.scatter_add_(1, idx_hi.unsqueeze(0).expand(M, -1), grad_right)
        return grad_values, None, None, None


def _interp_indices(z_knots, z_query):
    N = z_knots.shape[0]
    idx_hi = torch.searchsorted(z_knots, z_query, right=False).clamp(1, N - 1)
    idx_lo = idx_hi - 1
    z_lo = z_knots[idx_lo]
    z_hi = z_knots[idx_hi]
    denom = (z_hi - z_lo).clamp(min=torch.finfo(z_knots.dtype).eps)
    weights = (z_query - z_lo) / denom
    return idx_lo, idx_hi, weights


_interp_cache = {}


def interp_batch(
    z_knots: torch.Tensor,
    values_batch: torch.Tensor,
    z_query: torch.Tensor,
    *,
    cache_key: str | None = None,
) -> torch.Tensor:
    if values_batch.dim() != 2:
        raise ValueError("interp_batch expects values_batch with shape [M, N]")
    N = z_knots.shape[0]
    if N < 2:
        raise ValueError("interp_batch requires at least two knots")

    if cache_key is not None and cache_key in _interp_cache:
        idx_lo, idx_hi, weights = _interp_cache[cache_key]
    else:
        idx_lo, idx_hi, weights = _interp_indices(z_knots, z_query)
        if cache_key is not None:
            _interp_cache[cache_key] = (idx_lo, idx_hi, weights)

    return _InterpBatchFn.apply(values_batch, idx_lo, idx_hi, weights.to(values_batch.dtype))


__all__ = ["SearchsortedLerpFn", "searchsorted_lerp", "interp_batch"]
