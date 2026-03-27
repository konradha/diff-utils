import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "diff-utils") not in sys.path:
    sys.path.insert(0, str(ROOT / "diff-utils"))

from diff_utils._ext import _cpu_ext
from krakmod.env import Environment
from krakmod.io import read_env
from krakmod.kraken import kraken, krakenc
from krakmod.mesh import discretize
from krakmod.operators import (
    _complex_bc_params,
    _dispersion_layer_tensors,
    dispersion_complex,
    select_supported_krakenc_bottom_branch_sign,
)

FIELD_FIXTURES = ROOT / "tests" / "fixtures" / "field"


def _run_refine_kernel(name: str):
    ext = _cpu_ext()
    assert ext is not None, "CPU extension is required for refine_roots kernel tests"

    ef = read_env(str(FIELD_FIXTURES / f"{name}.env"))
    env = Environment.from_envfile(ef)

    fwd = kraken(env)
    ws = discretize(env, nv=16)
    omega = ws.omega2**0.5
    real_k_min = omega / ws.c_high

    ll, ln, lh, lr = _dispersion_layer_tensors(ws)
    bot_type, bot_cp, bot_ci, bot_cs, bot_csi, bot_rho = _complex_bc_params(env, ws, top=False)
    top_type, top_cp, top_ci, top_cs, top_csi, top_rho = _complex_bc_params(env, ws, top=True)

    x_refined, signs = ext.refine_roots(
        fwd.x.detach().to(torch.complex128).contiguous(),
        fwd.k.detach().to(torch.complex128).contiguous(),
        ws.B1.contiguous(),
        ll,
        ln,
        lh,
        lr,
        ws.omega2,
        bot_type,
        bot_cp,
        bot_ci,
        bot_cs,
        bot_csi,
        bot_rho,
        top_type,
        top_cp,
        top_ci,
        top_cs,
        top_csi,
        top_rho,
        real_k_min,
        80,
    )
    return env, ws, fwd, x_refined, signs


def _best_initial_residuals(env, ws, x_init: torch.Tensor) -> torch.Tensor:
    vals = []
    for x in x_init:
        x_c = complex(x.item())
        resid_pos = abs(dispersion_complex(x_c, ws, env, bottom_branch_sign=1.0))
        resid_neg = abs(dispersion_complex(x_c, ws, env, bottom_branch_sign=-1.0))
        vals.append(min(resid_pos, resid_neg))
    return torch.tensor(vals, dtype=torch.float64)


@pytest.mark.parametrize("name", ["calibK", "calibKgrad"])
def test_refine_roots_kernel_reduces_dispersion_residual(name: str):
    env, ws, fwd, x_refined, signs = _run_refine_kernel(name)

    init_best = _best_initial_residuals(env, ws, fwd.x.detach())
    refined = torch.tensor(
        [
            abs(dispersion_complex(complex(x.item()), ws, env, bottom_branch_sign=float(s.item())))
            for x, s in zip(x_refined, signs)
        ],
        dtype=torch.float64,
    )

    # Refinement should not make the supported subset materially worse.
    assert torch.all(refined <= init_best * 1.05 + 1e-12)

    # Tail modes are where refinement/branch selection should matter the most.
    tail = slice(max(0, fwd.M - 8), fwd.M)
    assert torch.any(refined[tail] < init_best[tail] * 0.5)


@pytest.mark.parametrize(
    ("name", "mode_index", "expected_sign"),
    [
        ("calibK", 28, +1.0),
        ("calibK", 33, -1.0),
        ("calibKgrad", 29, +1.0),
        ("calibKgrad", 33, -1.0),
    ],
)
def test_refine_roots_kernel_returns_expected_tail_branch(name: str, mode_index: int, expected_sign: float):
    env, ws, _fwd, x_refined, signs = _run_refine_kernel(name)

    idx = mode_index - 1
    assert idx < x_refined.shape[0]
    x_mode = complex(x_refined[idx].item())

    resid_pos = abs(dispersion_complex(x_mode, ws, env, bottom_branch_sign=1.0))
    resid_neg = abs(dispersion_complex(x_mode, ws, env, bottom_branch_sign=-1.0))
    assert abs(resid_pos - resid_neg) > 1e-12, "branch preference is too ambiguous for this regression"

    python_sign = select_supported_krakenc_bottom_branch_sign(x_mode, ws, env)
    assert python_sign == expected_sign
    assert float(signs[idx].item()) == expected_sign


@pytest.mark.parametrize("name", ["calibK", "calibKgrad"])
def test_refine_roots_kernel_matches_high_level_krakenc_roots(name: str):
    env, ws, _fwd, x_refined, signs = _run_refine_kernel(name)
    result = krakenc(env)

    refined_k = torch.sqrt(x_refined)
    refined_k = torch.where(refined_k.real < 0, -refined_k, refined_k)
    order = torch.argsort(refined_k.real, descending=True)

    # krakenc uses krakenc_fused (warm-start, no Newton) while refine_roots uses
    # full secant+Newton+conjugate. They converge to the same roots for well-behaved
    # modes but may differ for near-cutoff modes. Accept 1% relative tolerance.
    torch.testing.assert_close(x_refined[order], result.x, rtol=1e-2, atol=1e-6)
    selected_signs = torch.tensor(
        [
            select_supported_krakenc_bottom_branch_sign(complex(x.item()), ws, env)
            for x in result.x
        ],
        dtype=torch.float64,
    )
    # Branch signs may differ for near-cutoff modes where roots diverge
    x_diff = (x_refined[order] - result.x).abs()
    well_converged = x_diff < 1e-6
    if well_converged.any():
        torch.testing.assert_close(
            signs[order][well_converged], selected_signs[well_converged], rtol=0.0, atol=0.0
        )
