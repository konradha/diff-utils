from __future__ import annotations

import torch

from diff_utils.interp import searchsorted_lerp

def mode_coupling(
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    z_common: torch.Tensor,
    rho_common: torch.Tensor,
) -> torch.Tensor:
    M_L = phi_left.shape[0]
    M_R = phi_right.shape[0]
    N_c = z_common.shape[0]

    phi_L_c = torch.zeros(M_L, N_c, dtype=torch.complex128)
    phi_R_c = torch.zeros(M_R, N_c, dtype=torch.complex128)
    for m in range(M_L):
        phi_L_c[m] = searchsorted_lerp(z_left, phi_left[m].to(torch.float64), z_common).to(
            torch.complex128
        )
    for m in range(M_R):
        phi_R_c[m] = searchsorted_lerp(z_right, phi_right[m].to(torch.float64), z_common).to(
            torch.complex128
        )

    dz = z_common[1:] - z_common[:-1]
    w = torch.zeros(N_c, dtype=torch.float64)
    w[0] = dz[0] / 2.0
    w[-1] = dz[-1] / 2.0
    if N_c > 2:
        w[1:-1] = (dz[:-1] + dz[1:]) / 2.0
    w_rho = (w / rho_common).to(torch.complex128)  # [N_c]

    C = phi_L_c @ (w_rho.unsqueeze(1) * phi_R_c.T)  # [M_L, M_R]
    return C

__all__ = ["mode_coupling"]
