from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.krakel_gate import krakel_eigenvalue_gate
from diff_utils.logdet import dense_to_lapack_band


def _simple_tridiag_assemble(x, d_vals):
    N = d_vals.shape[0]
    A = (
        torch.diag(d_vals - x)
        + torch.diag(torch.ones(N - 1, dtype=d_vals.dtype), 1)
        + torch.diag(torch.ones(N - 1, dtype=d_vals.dtype), -1)
    )
    kl, ku = 1, 1
    A_band = dense_to_lapack_band(A, kl, ku)
    return A_band, kl, ku


def test_krakel_eigenvalue_gate_runs():
    d = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)

    A = (
        torch.diag(d.detach())
        + torch.diag(torch.ones(2, dtype=torch.float64), 1)
        + torch.diag(torch.ones(2, dtype=torch.float64), -1)
    )
    evals = torch.linalg.eigvalsh(A)
    x_converged = evals[0:1]

    x_out = krakel_eigenvalue_gate(x_converged, _simple_tridiag_assemble, d)
    assert torch.allclose(x_out, x_converged)


def test_krakel_gradient_flows():
    d = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64, requires_grad=True)

    A = (
        torch.diag(d.detach())
        + torch.diag(torch.ones(2, dtype=torch.float64), 1)
        + torch.diag(torch.ones(2, dtype=torch.float64), -1)
    )
    evals = torch.linalg.eigvalsh(A)
    x_converged = evals[0:1]

    x_out = krakel_eigenvalue_gate(x_converged, _simple_tridiag_assemble, d)
    x_out.sum().backward()

    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
