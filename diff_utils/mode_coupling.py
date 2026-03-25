import torch

from diff_utils.interp import interp_batch


def mode_coupling(
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    z_common: torch.Tensor,
    rho_common: torch.Tensor,
) -> torch.Tensor:
    N_c = z_common.shape[0]

    phi_L_c = interp_batch(z_left, phi_left, z_common).to(torch.complex128)
    phi_R_c = interp_batch(z_right, phi_right, z_common).to(torch.complex128)

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
