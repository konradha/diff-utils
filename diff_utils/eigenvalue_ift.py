from typing import Callable

import torch


class _EigenvalueIFTFn(torch.autograd.Function):
    @staticmethod
    def forward(x_converged: torch.Tensor, dispersion_fn: Callable, *theta_tensors: torch.Tensor):
        return x_converged.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_converged = inputs[0]
        dispersion_fn = inputs[1]
        theta_tensors = inputs[2:]
        ctx.save_for_backward(x_converged, *theta_tensors)
        ctx.n_theta = len(theta_tensors)
        ctx.dispersion_fn = dispersion_fn

    @staticmethod
    def backward(ctx, grad_x):
        saved = ctx.saved_tensors
        x_converged = saved[0]
        theta_tensors = saved[1:]
        dispersion_fn = ctx.dispersion_fn

        M = x_converged.shape[0] if x_converged.dim() > 0 else 1
        x_flat = x_converged.reshape(-1)
        grad_flat = grad_x.reshape(-1)
        grad_thetas = [torch.zeros_like(t) for t in theta_tensors]

        with torch.enable_grad():
            for m in range(M):
                x_m = x_flat[m].detach().clone().requires_grad_(True)
                theta_detached = [t.detach().clone().requires_grad_(True) for t in theta_tensors]
                delta = dispersion_fn(x_m, *theta_detached)
                grad_ones = torch.ones_like(delta)
                dD_dx = torch.autograd.grad(
                    delta, x_m, grad_outputs=grad_ones, create_graph=False, retain_graph=True
                )[0]
                if dD_dx.abs() < 1e-30:
                    continue
                dD_dtheta = torch.autograd.grad(
                    delta,
                    theta_detached,
                    grad_outputs=grad_ones,
                    create_graph=False,
                    allow_unused=True,
                )
                ift_factor = -grad_flat[m] / dD_dx
                for k, dD_dt in enumerate(dD_dtheta):
                    if dD_dt is not None:
                        if x_converged.is_complex():
                            grad_thetas[k] = grad_thetas[k] + ift_factor * dD_dt
                        else:
                            grad_thetas[k] = grad_thetas[k] + float(ift_factor) * dD_dt

        return (None, None) + tuple(grad_thetas)


def eigenvalue_ift(
    x_converged: torch.Tensor,
    dispersion_fn: Callable,
    *theta_tensors: torch.Tensor,
) -> torch.Tensor:
    return _EigenvalueIFTFn.apply(x_converged, dispersion_fn, *theta_tensors)


__all__ = ["eigenvalue_ift"]
