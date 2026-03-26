import torch

from diff_utils._ext import _cpu_ext


def solve_tridiag(
    dl: torch.Tensor,
    d: torch.Tensor,
    du: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    ext = _cpu_ext()
    if ext is not None:
        return ext.solve_tridiag(
            dl.contiguous(),
            d.contiguous(),
            du.contiguous(),
            b.contiguous(),
        )
    return _solve_tridiag_python(dl, d, du, b)


def solve_tridiag_batch(
    dl: torch.Tensor,
    d_batch: torch.Tensor,
    du: torch.Tensor,
    b_batch: torch.Tensor,
) -> torch.Tensor:
    ext = _cpu_ext()
    if ext is not None:
        return ext.solve_tridiag_batch(
            dl.contiguous(),
            d_batch.contiguous(),
            du.contiguous(),
            b_batch.contiguous(),
        )
    return _solve_tridiag_batch_python(dl, d_batch, du, b_batch)


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
    if ext is not None and hasattr(ext, 'tridiag_inverse_iteration_batch'):
        return ext.tridiag_inverse_iteration_batch(
            d_batch.contiguous(), e.contiguous(), n_iter,
        )

    phi = torch.full((M, N), 1e-10, dtype=d_batch.dtype)
    for _ in range(n_iter):
        phi = solve_tridiag_batch(e, d_batch, e, phi)
        norms = phi.abs().amax(dim=1, keepdim=True)
        norms = norms.clamp(min=1e-30)
        phi = phi / norms

    if phi.is_complex():
        norms = torch.sqrt((phi.abs() ** 2).sum(dim=1, keepdim=True))
    else:
        norms = torch.sqrt((phi * phi).sum(dim=1, keepdim=True))
    norms = norms.abs().clamp(min=1e-30)
    phi = phi / norms
    return phi


def _solve_tridiag_python(dl, d, du, b):
    N = d.shape[0]
    dw = d.clone()
    x = b.clone()
    for i in range(1, N):
        w = dl[i - 1] / dw[i - 1]
        dw[i] = dw[i] - w * du[i - 1]
        x[i] = x[i] - w * x[i - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = (x[i] - du[i] * x[i + 1]) / dw[i]
    return x


def _solve_tridiag_batch_python(dl, d_batch, du, b_batch):
    M, N = d_batch.shape
    dw = d_batch.clone()
    x = b_batch.clone()
    for i in range(1, N):
        w = dl[i - 1] / dw[:, i - 1]
        dw[:, i] = dw[:, i] - w * du[i - 1]
        x[:, i] = x[:, i] - w * x[:, i - 1]
    x[:, N - 1] = x[:, N - 1] / dw[:, N - 1]
    for i in range(N - 2, -1, -1):
        x[:, i] = (x[:, i] - du[i] * x[:, i + 1]) / dw[:, i]
    return x


__all__ = [
    "solve_tridiag",
    "solve_tridiag_batch",
    "tridiag_inverse_iteration",
    "tridiag_inverse_iteration_batch",
]
