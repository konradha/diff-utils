from __future__ import annotations

from typing import Callable

import torch

from diff_utils.logdet import banded_logdet
from diff_utils.eigenvalue_gate import eigenvalue_gate


def krakel_eigenvalue_gate(
    x_converged: torch.Tensor,
    assemble_fn: Callable,
    *theta_tensors: torch.Tensor,
) -> torch.Tensor:
    def dispersion_fn(x_m, *theta):
        A_band, kl, ku = assemble_fn(x_m, *theta)
        _, logabsdet = banded_logdet(A_band, kl, ku)
        return logabsdet

    return eigenvalue_gate(x_converged, dispersion_fn, *theta_tensors)


__all__ = ["krakel_eigenvalue_gate"]
