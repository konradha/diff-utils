# banded

Differentiable single-core CPU banded linear algebra for PyTorch.

This repository focuses on inverse problems where banded operators are evaluated many times in optimization loops and root-finding loops. The code is designed for correctness, custom autograd behavior, and practical CPU performance.

## Core Goal

Build production-usable differentiable operators for banded systems that avoid dense formulations and integrate with PyTorch autograd.

## Implemented Operators

## 1) `solve_banded` (CSR storage)

- Solves `A x = b` for fixed-bandwidth sparse CSR matrices.
- Supports real/complex, single and batched RHS.
- Has custom backward and vmap support.

Public API:
- `banded.make_banded_csr`
- `banded.solve_banded`
- `banded.solve_banded_csr_values`

## 2) `BandedLogDet` (LAPACK GB storage)

- Computes signed log-determinant of real banded matrices.
- Uses LAPACK `dgbtrf` in forward.
- Uses custom backward based on
  $$
  \frac{\partial \log|\det A|}{\partial A_{ij}} = (A^{-T})_{ij},
  $$
  evaluated via `dgbtrs(..., trans='T')`.

Public API:
- `banded.BandedLogDet`
- `banded.banded_logdet`
- `banded.dense_to_lapack_band`
- `banded.lapack_band_to_dense`

## Performance Focus

- Single-core CPU first.
- Low Python overhead in tight loops.
- LAPACK-backed critical paths for factorization/solve.
- Benchmark harnesses under `benchmarks/`.

## Testing

Run all tests:
```bash
pytest -q
```

Run only logdet tests:
```bash
pytest -q tests/test_banded_logdet.py
```

Run perf-marked tests:
```bash
RUN_PERF_TESTS=1 pytest -q tests/test_banded_logdet.py -m performance
```

## Benchmarks

`BandedLogDet` benchmark:
```bash
python benchmarks/benchmark_banded_logdet.py --sizes 100 500 1000 5000 10000
```

## Demos

Demo descriptions and commands are in:
- `demo/README.md`

Most demos write:
- `*.mp4`
- `*_final.png`
- `*_history.npz`

## Project Layout

- `banded/`: library code
- `tests/`: correctness and gradient tests
- `benchmarks/`: microbench and scaling harnesses
- `demo/`: inverse-problem demos and animations
- `docs/`: writeup and PR notes
