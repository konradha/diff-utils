import torch

from diff_utils._ext import _cpu_ext


def solve_tridiag(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    ext = _cpu_ext()
    return ext.solve_tridiag(
        dl.contiguous(),
        d.contiguous(),
        du.contiguous(),
        b.contiguous(),
    )


def solve_tridiag_batch(
    dl: torch.Tensor,
    d_batch: torch.Tensor,
    du: torch.Tensor,
    b_batch: torch.Tensor,
) -> torch.Tensor:
    ext = _cpu_ext()
    return ext.solve_tridiag_batch(
        dl.contiguous(),
        d_batch.contiguous(),
        du.contiguous(),
        b_batch.contiguous(),
    )


def tridiag_inverse_iteration(
    d: torch.Tensor,
    e: torch.Tensor,
    n_iter: int = 3,
) -> torch.Tensor:
    N = d.shape[0]
    if N < 3:
        return torch.ones(N, dtype=d.dtype) / (N**0.5)

    phi = torch.ones(N, dtype=d.dtype) * 1e-10
    for _ in range(n_iter):
        phi = solve_tridiag(e, d, e, phi)
        norm = phi.abs().max()
        if norm > 0:
            phi = phi / norm

    if phi.is_complex():
        norm = torch.sqrt((phi.abs() ** 2).sum())
    else:
        norm = torch.sqrt(torch.dot(phi, phi))
    if norm.abs() > 0:
        phi = phi / norm
    return phi


def tridiag_inverse_iteration_batch(
    d_batch: torch.Tensor,
    e: torch.Tensor,
    n_iter: int = 3,
) -> torch.Tensor:
    M, N = d_batch.shape
    if N < 3:
        return torch.ones(M, N, dtype=d_batch.dtype) / (N**0.5)

    ext = _cpu_ext()
    return ext.tridiag_inverse_iteration_batch(
        d_batch.contiguous(),
        e.contiguous(),
        n_iter,
    )


__all__ = [
    "solve_tridiag",
    "solve_tridiag_batch",
    "tridiag_inverse_iteration",
    "tridiag_inverse_iteration_batch",
]
