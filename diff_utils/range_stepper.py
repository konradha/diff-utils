from __future__ import annotations

import torch


def range_stepper(
    A0: torch.Tensor,
    k_segments: list,
    dr_segments: list,
    C_interfaces: list = None,
) -> torch.Tensor:
    A = A0.clone()

    for i, (k, dr) in enumerate(zip(k_segments, dr_segments)):
        M = min(A.shape[0], k.shape[0])
        # within segment
        A = A[:M] * torch.exp(-1j * k[:M] * dr)

        # coupling next segment
        if C_interfaces is not None and i < len(C_interfaces):
            C = C_interfaces[i]  # [M_cur, M_next]
            A = C.T.to(torch.complex128) @ A.to(torch.complex128)

    return A


__all__ = ["range_stepper"]
