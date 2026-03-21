from __future__ import annotations

import torch

def _trap_weights(z: torch.Tensor) -> torch.Tensor:
    N = z.shape[0]
    dz = z[1:] - z[:-1]
    w = torch.zeros(N, dtype=z.dtype, device=z.device)
    w[0] = dz[0] / 2.0
    w[-1] = dz[-1] / 2.0
    if N > 2:
        w[1:-1] = (dz[:-1] + dz[1:]) / 2.0
    return w

def weighted_depth_integral(
    f: torch.Tensor,
    z: torch.Tensor,
    rho: torch.Tensor = None,
) -> torch.Tensor:
    w = _trap_weights(z)
    if rho is not None:
        w = w / rho
    return (f * w).sum(dim=-1)

def weighted_depth_inner_product(
    f: torch.Tensor,
    g: torch.Tensor,
    z: torch.Tensor,
    rho: torch.Tensor = None,
) -> torch.Tensor:
    w = _trap_weights(z)
    if rho is not None:
        w = w / rho

    if f.dim() == 1 and g.dim() == 1:
        return (f * g * w).sum()
    elif f.dim() == 2 and g.dim() == 2:
        return f @ (w.unsqueeze(1) * g.T)
    elif f.dim() == 2 and g.dim() == 1:
        return (f * (g * w).unsqueeze(0)).sum(dim=-1)
    else:
        return (f * g * w).sum(dim=-1)

__all__ = ["weighted_depth_integral", "weighted_depth_inner_product"]
