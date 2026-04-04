import sys
from pathlib import Path

import torch
from torch.autograd import gradcheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_utils.mode_coupling import mode_coupling
from diff_utils.range_stepper import range_stepper, range_stepper_batched
from diff_utils.interp import interp_batch, searchsorted_lerp


def test_mode_coupling_self():
    N = 50
    z = torch.linspace(0, 100, N, dtype=torch.float64)
    phi = torch.stack([torch.sin(k * 3.14159 * z / 100) for k in range(1, 4)])
    phi = phi / phi.norm(dim=1, keepdim=True)
    rho = torch.ones(N, dtype=torch.float64)

    C = mode_coupling(phi, phi, z, z, z, rho)
    assert C.shape == (3, 3)
    diag = C.real.diagonal()
    assert torch.all(diag > 0.9)
    off_diag = C.real - torch.diag(diag)
    assert off_diag.abs().max() < 0.1


def test_mode_coupling_shape():
    z_l = torch.linspace(0, 100, 30, dtype=torch.float64)
    z_r = torch.linspace(0, 100, 40, dtype=torch.float64)
    z_c = torch.linspace(0, 100, 50, dtype=torch.float64)
    phi_l = torch.randn(3, 30, dtype=torch.float64)
    phi_r = torch.randn(5, 40, dtype=torch.float64)
    rho = torch.ones(50, dtype=torch.float64)
    C = mode_coupling(phi_l, phi_r, z_l, z_r, z_c, rho)
    assert C.shape == (3, 5)


def test_mode_coupling_preserves_complex_values():
    z = torch.linspace(0, 100, 40, dtype=torch.float64)
    rho = torch.ones(40, dtype=torch.float64)
    phi_l = torch.stack([torch.exp(1j * 0.01 * z), torch.exp(1j * 0.02 * z)])
    phi_r = torch.stack([torch.exp(-1j * 0.015 * z), torch.exp(1j * 0.005 * z)])

    C = mode_coupling(phi_l, phi_r, z, z, z, rho)

    assert C.dtype == torch.complex128
    assert C.abs().max() > 0.0


def test_range_stepper_single_segment():
    M = 5
    k = torch.randn(M, dtype=torch.complex128) * 0.01 + 0.04
    A0 = torch.ones(M, dtype=torch.complex128)
    dr = 1000.0

    A = range_stepper(A0, [k], [dr])
    expected = A0 * torch.exp(-1j * k * dr)
    assert torch.allclose(A, expected, atol=1e-12)


def test_range_stepper_multi_segment():
    M = 3
    k1 = torch.tensor([0.04, 0.03, 0.02], dtype=torch.complex128)
    k2 = torch.tensor([0.041, 0.031, 0.021], dtype=torch.complex128)
    A0 = torch.ones(M, dtype=torch.complex128)
    C = torch.eye(M, dtype=torch.complex128)

    A = range_stepper(A0, [k1, k2], [1000.0, 2000.0], [C])
    assert A.shape == (M,)
    assert torch.isfinite(A.real).all()


def test_range_stepper_gradcheck_k():
    M = 3
    k = torch.randn(M, dtype=torch.complex128, requires_grad=True) * 0.01 + 0.04
    A0 = torch.ones(M, dtype=torch.complex128)

    def fn(kk):
        return range_stepper(A0, [kk], [1000.0]).abs().sum()

    assert gradcheck(fn, (k,), eps=1e-7, atol=1e-4)


def test_range_stepper_gradcheck_A0():
    M = 3
    k = torch.tensor([0.04, 0.03, 0.02], dtype=torch.complex128)
    A0 = torch.ones(M, dtype=torch.complex128, requires_grad=True)

    def fn(a):
        return range_stepper(a, [k], [1000.0]).abs().sum()

    assert gradcheck(fn, (A0,), eps=1e-7, atol=1e-4)


def test_interp_batch_parity():
    z = torch.linspace(0, 10, 20, dtype=torch.float64)
    values = torch.stack([torch.sin(z), torch.cos(z), z**2])
    q = torch.tensor([1.5, 5.5, 8.3], dtype=torch.float64)

    batch_out = interp_batch(z, values, q)
    ref = torch.stack([searchsorted_lerp(z, values[m], q) for m in range(3)])
    assert torch.allclose(batch_out, ref, atol=1e-14)


def test_interp_batch_shape():
    z = torch.linspace(0, 100, 50, dtype=torch.float64)
    values = torch.randn(10, 50, dtype=torch.float64)
    q = torch.linspace(5, 95, 30, dtype=torch.float64)
    out = interp_batch(z, values, q)
    assert out.shape == (10, 30)


def test_interp_batch_complex_shape_and_dtype():
    z = torch.linspace(0, 100, 50, dtype=torch.float64)
    values = torch.randn(4, 50, dtype=torch.float64) + 1j * torch.randn(4, 50, dtype=torch.float64)
    q = torch.linspace(5, 95, 30, dtype=torch.float64)
    out = interp_batch(z, values.to(torch.complex128), q)
    assert out.shape == (4, 30)
    assert out.dtype == torch.complex128


def test_interp_batch_gradcheck():
    z = torch.linspace(0, 10, 15, dtype=torch.float64)
    values = torch.randn(3, 15, dtype=torch.float64, requires_grad=True)
    q = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float64)

    def fn(v):
        return interp_batch(z, v, q).sum()

    assert gradcheck(fn, (values,), eps=1e-7, atol=1e-5)


def test_batched_single_segment_parity():
    M = 5
    k = torch.randn(M, dtype=torch.complex128) * 0.01 + 0.04
    A0 = torch.ones(M, dtype=torch.complex128)
    r = torch.tensor([1000.0, 3000.0, 5000.0, 10000.0], dtype=torch.float64)

    A_batched = range_stepper_batched(A0, k, r, torch.tensor([], dtype=torch.float64))

    for i, ri in enumerate(r):
        A_ref = A0 * torch.exp(-1j * k * ri)
        assert torch.allclose(A_batched[:, i], A_ref, atol=1e-12), f"r={ri}"


def test_batched_multi_segment_parity():
    M = 3
    k1 = torch.tensor([0.04, 0.03, 0.02], dtype=torch.complex128)
    k2 = torch.tensor([0.041, 0.031, 0.021], dtype=torch.complex128)
    A0 = torch.ones(M, dtype=torch.complex128)
    C = torch.eye(M, dtype=torch.complex128) * 0.95

    r_bnd = torch.tensor([5000.0], dtype=torch.float64)
    r = torch.tensor([2000.0, 4000.0, 7000.0, 12000.0], dtype=torch.float64)

    A_batched = range_stepper_batched(
        A0,
        k1,
        r,
        r_bnd,
        k_segments=[k1, k2],
        C_interfaces=[C],
    )

    assert A_batched.shape == (M, 4)
    assert torch.isfinite(A_batched.real).all()
    A_r0 = A0 * torch.exp(-1j * k1 * 2000.0)
    assert torch.allclose(A_batched[:, 0], A_r0, atol=1e-12)


def test_batched_gradcheck_k():
    M = 3
    k = (torch.randn(M, dtype=torch.complex128) * 0.001 + 0.01).requires_grad_(True)
    A0 = torch.ones(M, dtype=torch.complex128)
    r = torch.tensor([10.0, 50.0], dtype=torch.float64)

    def fn(kk):
        return range_stepper_batched(A0, kk, r, torch.tensor([], dtype=torch.float64)).abs().sum()

    assert gradcheck(fn, (k,), eps=1e-7, atol=1e-4)


def test_batched_gradcheck_A0():
    M = 3
    k = torch.tensor([0.04, 0.03, 0.02], dtype=torch.complex128)
    A0 = torch.ones(M, dtype=torch.complex128, requires_grad=True)
    r = torch.tensor([3000.0, 7000.0], dtype=torch.float64)

    def fn(a):
        return range_stepper_batched(a, k, r, torch.tensor([], dtype=torch.float64)).abs().sum()

    assert gradcheck(fn, (A0,), eps=1e-7, atol=1e-4)


def test_batched_many_receivers():
    M = 5
    k_real = torch.linspace(0.03, 0.05, M, dtype=torch.float64)
    k_imag = -torch.linspace(0.0, 2e-5, M, dtype=torch.float64)
    k = k_real.to(torch.complex128) + 1j * k_imag.to(torch.complex128)
    A0 = torch.ones(M, dtype=torch.complex128)
    r = torch.linspace(100.0, 50000.0, 200, dtype=torch.float64)

    A_batched = range_stepper_batched(A0, k, r, torch.tensor([], dtype=torch.float64))
    assert A_batched.shape == (M, 200)
    assert torch.isfinite(A_batched.real).all()
