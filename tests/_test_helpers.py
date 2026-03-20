from __future__ import annotations

from typing import Dict

import torch


def random_banded_dense(
    n: int,
    kl: int,
    ku: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    diag_dominant: bool = True,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    is_complex = dtype in (torch.complex64, torch.complex128)
    real_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32

    A = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        lo = max(0, i - kl)
        hi = min(n - 1, i + ku)
        for j in range(lo, hi + 1):
            if i != j:
                re = 0.08 * torch.randn((), generator=g, dtype=real_dtype)
                if is_complex:
                    im = 0.08 * torch.randn((), generator=g, dtype=real_dtype)
                    A[i, j] = torch.complex(re, im)
                else:
                    A[i, j] = re

    if diag_dominant:
        row_abs = A.abs().sum(dim=1)
        diag_vals = row_abs + 1.5
        if is_complex:
            diag_vals = diag_vals.to(dtype)
        A[torch.arange(n), torch.arange(n)] = diag_vals

    return A


def random_banded_diags(
    n: int,
    kl: int,
    ku: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    is_complex = dtype in (torch.complex64, torch.complex128)
    real_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32

    diags: Dict[int, torch.Tensor] = {}
    abs_sum = torch.zeros(n, dtype=real_dtype)

    for d in range(-kl, ku + 1):
        if d == 0:
            continue
        length = n - abs(d)
        re = torch.randn(length, dtype=real_dtype, generator=g) * 0.05
        if is_complex:
            im = torch.randn(length, dtype=real_dtype, generator=g) * 0.05
            vals = (re + 1j * im).to(dtype)
        else:
            vals = re.to(dtype)
        diags[d] = vals
        rows = torch.arange(max(0, -d), min(n, n - d))
        abs_sum[rows] += vals.abs().to(real_dtype)

    main = abs_sum + 2.0
    if is_complex:
        main = main.to(dtype)
    diags[0] = main
    return diags


def pekeris_B1_rho_h(
    n_points: int = 100,
    c_water: float = 1500.0,
    c_bottom: float = 1700.0,
    rho_water: float = 1.0,
    rho_bottom: float = 1.5,
    freq: float = 100.0,
    depth: float = 100.0,
) -> tuple:
    omega = 2.0 * 3.141592653589793 * freq
    h = depth / n_points
    h2 = h * h
    k_water = omega / c_water
    B1 = torch.full((n_points + 1,), -2.0 + h2 * k_water**2, dtype=torch.float64)
    rho_arr = torch.full((n_points + 1,), rho_water, dtype=torch.float64)
    return B1, rho_arr, h, omega, k_water, c_water, c_bottom, rho_water, rho_bottom


def dense_to_band_complex(A: torch.Tensor, kl: int, ku: int) -> torch.Tensor:
    from banded.logdet import dense_to_lapack_band

    return dense_to_lapack_band(A, kl, ku)
