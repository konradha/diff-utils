import torch

from diff_utils._ext import _cpu_ext

# BC type constants just like at/ fortran code
BC_VACUUM = 0
BC_RIGID = 1
BC_ACOUSTIC = 2
BC_ELASTIC = 3


def acoustic_solve1(
    B1: torch.Tensor,
    layer_loc: torch.Tensor,
    layer_n: torch.Tensor,
    layer_h: torch.Tensor,
    layer_rho: torch.Tensor,
    omega2: float,
    bc_bot_type: int,
    bc_bot_cp: float,
    bc_bot_cs: float,
    bc_bot_rho: float,
    bc_top_type: int,
    bc_top_cp: float,
    bc_top_cs: float,
    bc_top_rho: float,
    x_min: float,
    x_max: float,
    precision: float = 15.0,
    max_modes: int = 5000,
) -> tuple[torch.Tensor, int]:
    ext = _cpu_ext()
    if ext is None:
        raise RuntimeError("C++ extension required")
    return ext.acoustic_solve1(
        B1.contiguous(),
        layer_loc.contiguous().to(torch.int64),
        layer_n.contiguous().to(torch.int64),
        layer_h.contiguous().to(torch.float64),
        layer_rho.contiguous().to(torch.float64),
        omega2,
        bc_bot_type,
        bc_bot_cp,
        bc_bot_cs,
        bc_bot_rho,
        bc_top_type,
        bc_top_cp,
        bc_top_cs,
        bc_top_rho,
        x_min,
        x_max,
        precision,
        max_modes,
    )


def acoustic_solve2(
    B1: torch.Tensor,
    layer_loc: torch.Tensor,
    layer_n: torch.Tensor,
    layer_h: torch.Tensor,
    layer_rho: torch.Tensor,
    omega2: float,
    bc_bot_type: int,
    bc_bot_cp: float,
    bc_bot_cs: float,
    bc_bot_rho: float,
    bc_top_type: int,
    bc_top_cp: float,
    bc_top_cs: float,
    bc_top_rho: float,
    prev_eigenvalues: torch.Tensor,
    M: int,
    precision: float = 15.0,
    c_high: float = 1e30,
) -> tuple[torch.Tensor, int]:
    ext = _cpu_ext()
    if ext is None:
        raise RuntimeError("C++ extension required")
    return ext.acoustic_solve2(
        B1.contiguous(),
        layer_loc.contiguous().to(torch.int64),
        layer_n.contiguous().to(torch.int64),
        layer_h.contiguous().to(torch.float64),
        layer_rho.contiguous().to(torch.float64),
        omega2,
        bc_bot_type,
        bc_bot_cp,
        bc_bot_cs,
        bc_bot_rho,
        bc_top_type,
        bc_top_cp,
        bc_top_cs,
        bc_top_rho,
        prev_eigenvalues.contiguous().to(torch.float64),
        M,
        precision,
        c_high,
    )


__all__ = [
    "acoustic_solve1",
    "acoustic_solve2",
    "BC_VACUUM",
    "BC_RIGID",
    "BC_ACOUSTIC",
    "BC_ELASTIC",
]
