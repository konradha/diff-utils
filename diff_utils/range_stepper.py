from __future__ import annotations

from typing import List, Optional

import torch


def range_stepper(
    A0: torch.Tensor,
    k_segments: List[torch.Tensor],
    dr_segments: List[float],
    C_interfaces: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    A = A0.clone()
    for i, (k, dr) in enumerate(zip(k_segments, dr_segments)):
        M = min(A.shape[0], k.shape[0])
        A = A[:M] * torch.exp(-1j * k[:M] * dr)
        if C_interfaces is not None and i < len(C_interfaces):
            A = C_interfaces[i].T.to(torch.complex128) @ A.to(torch.complex128)
    return A


def range_stepper_batched(
    A0: torch.Tensor,
    k: torch.Tensor,
    r_receivers: torch.Tensor,
    r_boundaries: torch.Tensor,
    k_segments: Optional[List[torch.Tensor]] = None,
    C_interfaces: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    M = A0.shape[0]
    N_r = r_receivers.shape[0]
    N_b = r_boundaries.shape[0] if r_boundaries is not None else 0
    if N_b == 0:
        seg_ids = torch.zeros(N_r, dtype=torch.long)
    else:
        seg_ids = torch.searchsorted(r_boundaries, r_receivers, right=True)

    n_segments = int(seg_ids.max().item()) + 1 if N_r > 0 else 1

    A_out = torch.zeros(M, N_r, dtype=torch.complex128)
    A_cur = A0.to(torch.complex128).clone()
    k_cur = k.to(torch.complex128)
    M_cur = M
    r_prev = 0.0

    for seg in range(n_segments):
        # bdry crossing
        if seg > 0 and N_b > 0:
            r_bnd = r_boundaries[seg - 1].item()
            dr_to_boundary = r_bnd - r_prev
            if dr_to_boundary > 0:
                A_cur = A_cur[:M_cur] * torch.exp(-1j * k_cur[:M_cur] * dr_to_boundary)
            r_prev = r_bnd
            if C_interfaces is not None and seg - 1 < len(C_interfaces):
                C = C_interfaces[seg - 1]
                A_cur = C.T.to(torch.complex128) @ A_cur[:M_cur].to(torch.complex128)
                M_cur = A_cur.shape[0]

            if k_segments is not None and seg < len(k_segments):
                k_cur = k_segments[seg].to(torch.complex128)
                M_cur = min(M_cur, k_cur.shape[0])

        mask = seg_ids == seg
        if not mask.any():
            continue

        r_in_seg = r_receivers[mask]  # [N_seg]
        dr_from_prev = r_in_seg - r_prev  # [N_seg]

        phase = torch.exp(
            -1j * k_cur[:M_cur].unsqueeze(1) * dr_from_prev.unsqueeze(0)
        )  # [M_cur, N_seg]
        A_at_receivers = A_cur[:M_cur].unsqueeze(1) * phase  # [M_cur, N_seg]

        idx = mask.nonzero(as_tuple=True)[0]
        A_out[:M_cur, idx] = A_at_receivers

    return A_out


__all__ = ["range_stepper", "range_stepper_batched"]
