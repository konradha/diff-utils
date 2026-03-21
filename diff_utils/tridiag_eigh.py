
import numpy as np
import torch
from scipy.linalg.lapack import dstev

class TridiagEigh(torch.autograd.Function):
    @staticmethod
    def forward(d: torch.Tensor, e: torch.Tensor, eps: float):
        N = d.shape[0]
        d_np = d.detach().contiguous().numpy().astype(np.float64, copy=True)
        e_np = e.detach().contiguous().numpy().astype(np.float64, copy=True)

        sigma, Q, info = dstev(d_np, e_np, compute_v=1)
        if info != 0:
            raise RuntimeError(f"dstev failed with info={info}")

        sigma_t = torch.from_numpy(sigma.copy()).to(d.dtype)
        Q_t = torch.from_numpy(Q.copy()).to(d.dtype)
        return sigma_t, Q_t

    @staticmethod
    def setup_context(ctx, inputs, output):
        d, e, eps = inputs
        sigma, Q = output
        ctx.eps = eps
        ctx.save_for_backward(sigma, Q)

    @staticmethod
    def backward(ctx, grad_sigma, grad_Q):
        sigma, Q = ctx.saved_tensors
        eps = ctx.eps
        N = sigma.shape[0]

        diff = sigma.unsqueeze(0) - sigma.unsqueeze(1)  # [N, N]
        F = diff / (diff * diff + eps * eps)
        F.diagonal().zero_()

        QT_grad_Q = Q.T @ grad_Q  # [N, N]

        sym = F * (QT_grad_Q + QT_grad_Q.T) / 2.0

        if grad_sigma is not None:
            sym = sym + torch.diag(grad_sigma)

        grad_T = Q @ sym @ Q.T

        grad_d = grad_T.diagonal()
        grad_e = grad_T.diagonal(1) + grad_T.diagonal(-1)

        return grad_d, grad_e, None

def tridiag_eigh(
    d: torch.Tensor,
    e: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    return TridiagEigh.apply(d, e, eps)

__all__ = ["tridiag_eigh"]
