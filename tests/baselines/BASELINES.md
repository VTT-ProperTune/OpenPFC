<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Scientific & performance baselines (Pre-M0 / audit §16)

This directory holds the regression contract for the OpenPFC 0.2 refactor: the
reference results the big-bang refactor (M0→M12) must preserve. Each baseline is
classified **bitwise** (must reproduce exactly) or **tolerance** (must reproduce
within a stated numeric tolerance, per backend).

> Status: this file is the **framework and capture plan**. The correctness
> defects (Pre-M0 PA–PL) are fixed and the suite is green on CPU with CUDA/HIP
> compiling; capturing the golden *data* and the performance numbers requires
> runs on the reference machines (tohtori = CUDA, LUMI = HIP) and is the safety
> net to complete **before** M0 begins. Items marked ☐ are not yet captured.

## Classification

| Baseline | Type | Tolerance | Where it runs |
|---|---|---|---|
| Tungsten CPU↔CUDA parity (`test_tungsten_cpu_vs_cuda`) | tolerance | ≤ 1e-10 | tohtori (GPU) |
| Tungsten CPU↔HIP parity (`test_tungsten_cpu_vs_hip`) | tolerance | ≤ 1e-10 | LUMI (GPU) |
| allen_cahn / wave2d CPU↔GPU parity | tolerance | ≤ 1e-10 | tohtori / LUMI |
| ETD weight provenance (`spectral_exp_cache_matches_legacy_etd_weights`) | tolerance | test-defined | CI (CPU) |
| RK2/RK3/RK4 convergence-order windows | tolerance | ratio windows in-test | CI (CPU) |
| aluminum 5-step field norms (`aluminumTest`) | tolerance | ±0.1 (in-test) | CI (CPU) |
| kobayashi `KOBAYASHI_VERIFY_HEX` checksums | **bitwise** (CPU; CUDA pin) | exact hexfloat; CUDA `sum_T` 1 ULP vs CPU | cluster; CUDA CTest `kobayashi-cuda-hex-smoke` |
| kobayashi OpenMP thread-count parity | **bitwise** | exact | CI/cluster |
| heat3d manual-vs-stack L2 equality; wave2d manual-vs-separated | tolerance | in-test | CI (CPU) |
| Spectral first derivative of a Nyquist mode (`test_spectral_gradient`) | tolerance | ≤ 1e-12 (must be ~0) | CI (CPU) |
| Tungsten 0.2 vs Gen-1 trajectory (`test_tungsten` `[golden]`) | tolerance | ≤1e-10 local; Σψ² 1e-12 relative | CI 1-rank 8³/100 steps; MPI 4-rank 16³/20 steps (`tungsten-golden-4rank`). Pre-M0 field dump was never captured — this is the living A/B golden. |
| ☐ aluminumNew multi-rank golden trajectory | tolerance | TBD | tohtori/CPU (MPI) |
| ☐ CPU-side goldens for each CPU-vs-GPU parity test | tolerance | 1e-10 | CI (CPU) |
| Restart-equivalence (`CheckpointService`) | bitwise (1 rank) | exact owned cells | CI: kernel `[checkpoint][service]`; heat3d FD Euler; tungsten ETD JSON session. 2-rank kernel `[checkpoint][MPI]`. |

## Performance baselines (☐ — capture on the reference machines)

Capture machine-tagged JSON via the profiling schema-v2/v3 exporter into
`tests/baselines/perf/` and compare with `scripts/compare_perf_baseline.py`
(pass ≤5% regression / warn >5% / fail >15%; speedups pass):

```
python3 scripts/compare_perf_baseline.py \
  tests/baselines/perf/<baseline>.json path/to/new_profile.json
```

Canary input: `tests/baselines/perf/inputs/tungsten_canary.json` (64³, 20 steps,
`saveat=-1`, profiling JSON). Capture on tohtori with `tungsten` / `tungsten_cuda`
from a Release or Debug tree; copy the exported `*.json` next to this file with a
machine tag in the name (`tohtori-g0005-tungsten-cuda-1rank.json`).

- ☐ Tungsten strong scaling, CPU, 1/4/16 ranks (tohtori; 1-rank Debug canary exists)
- ☑ Tungsten CUDA 1-rank Debug canary (tohtori `g0005`, 64³ / 20 steps, `tungsten_cuda`, 2026-09-01): `tests/baselines/perf/tohtori-g0005-tungsten-cuda-1rank-debug-canary.json`. Compare with `--warmup-frames=1` (first step includes CUDA context setup). This is a plumbing pin, not a production scaling number.
- ☑ Tungsten CPU 1-rank Debug canary (same grid/steps, `tungsten`): `tests/baselines/perf/tohtori-g0005-tungsten-cpu-1rank-debug-canary.json`.
- ☐ Tungsten CUDA single node Release (tohtori GPU)
- ☐ Kobayashi HIP single node (LUMI)
- ☐ Halo-exchange microtimings, host and device, 2–8 ranks

## How to capture (reference commands)

CPU golden trajectory (Gen-1 `Tungsten` vs `tungsten_etd` / `TungstenETDSession`):

```
# 1-rank (CI): ctest -R tungsten-all-tests
# 4-rank MPI suite:
mpiexec -n 4 ./apps/tungsten/test_tungsten "[golden][MPI]"
```

A 4-rank / 100-step capture of `tungsten_etd` on tohtori is still optional for archival binaries under `tests/baselines/`; the Catch2 A/B is the gate.

GPU compile + run parity (cluster only):

```
scripts/build.sh --with-cuda  --build-dir=/WRK/<user>/openpfc/builds/cuda-release   # tohtori
scripts/build.sh --with-rocm  --build-dir=/WRK/<user>/openpfc/builds/rocm-release   # LUMI
# then run test_tungsten_cpu_vs_cuda / _hip and HIP_TungstenETD / CUDA_TungstenETD
# (App-GPU-IC is the [ic] case on TungstenETDGPUSession)
# Kobayashi CUDA HEX smoke (32² / 4 steps): ctest -R kobayashi-cuda-hex
```

Kobayashi CUDA HEX (Tohtori `g0005`, 2026-09-01, gcc 15.2 / CUDA 13.1 / CUDA-aware
Open MPI, `HaloExchange<CUDASpace>` library path, 1-rank and 2-rank identical):

```
KOBAYASHI_VERIFY_HEX sum_phi=0x1.b96bf451009d9p+3 sumsq_phi=0x1.4e770b1504ae4p+3 sum_T=0x1.6e128af4d5ac5p+0 sumsq_T=0x1.6546ee0a021fp-2
```

CPU OpenMP/MPI pin for the same `(Nx, Ny, steps, dt, dx)` has `sum_T=0x1.6e128af4d5ac6p+0`
(1 ULP). Cross-backend last-bit drift is expected; the CUDA pin is bitwise for the
CUDA driver so library-halo changes fail closed.

## Why bitwise vs tolerance

Consolidating kernels during the refactor (reduction order, FMA contraction,
CPU↔GPU math) perturbs the last bits, so cross-backend and post-refactor checks
are **tolerance**-based. Only same-binary, same-decomposition checks
(kobayashi checksums, OpenMP thread parity, single-rank restart round-trip) are
**bitwise**. Any tolerance that must be widened during the refactor requires a
one-line justification appended to this file with the commit that widens it.
