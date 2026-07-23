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
| kobayashi `KOBAYASHI_VERIFY_HEX` checksums | **bitwise** | exact hexfloat | cluster |
| kobayashi OpenMP thread-count parity | **bitwise** | exact | CI/cluster |
| heat3d manual-vs-stack L2 equality; wave2d manual-vs-separated | tolerance | in-test | CI (CPU) |
| ☐ Tungsten multi-rank golden trajectory (4 ranks, ≥100 ETD steps, CPU) | tolerance | TBD (propose 1e-12 per save-point checksum) | tohtori/CPU (MPI) |
| ☐ aluminumNew multi-rank golden trajectory | tolerance | TBD | tohtori/CPU (MPI) |
| ☐ CPU-side goldens for each CPU-vs-GPU parity test | tolerance | 1e-10 | CI (CPU) |
| ☐ Restart-equivalence (lands in M11 when a loader exists) | bitwise (1 rank) / tolerance (N rank) | — | — |

## Performance baselines (☐ — capture on the reference machines)

Capture machine-tagged JSON via the profiling schema-v2 exporter into
`tests/baselines/perf/` and compare with `scripts/compare_perf_baseline.py`
(pass/warn >5% / fail >15%):

- ☐ Tungsten strong scaling, CPU, 1/4/16 ranks (tohtori)
- ☐ Tungsten CUDA single node (tohtori GPU)
- ☐ Kobayashi HIP single node (LUMI)
- ☐ Halo-exchange microtimings, host and device, 2–8 ranks

## How to capture (reference commands)

CPU golden trajectory (example; adapt input + step count):

```
scripts/build.sh --build-type=Release --cpu --build-dir=/WRK/<user>/openpfc/builds/cpu-release
# run tungsten CPU on 4 ranks for >=100 steps, record per-save-point checksums
```

GPU compile + run parity (cluster only):

```
scripts/build.sh --with-cuda  --build-dir=/WRK/<user>/openpfc/builds/cuda-release   # tohtori
scripts/build.sh --with-rocm  --build-dir=/WRK/<user>/openpfc/builds/rocm-release   # LUMI
# then run test_tungsten_cpu_vs_cuda / _hip and test_tungsten_app_gpu_ic on a GPU node
```

## Why bitwise vs tolerance

Consolidating kernels during the refactor (reduction order, FMA contraction,
CPU↔GPU math) perturbs the last bits, so cross-backend and post-refactor checks
are **tolerance**-based. Only same-binary, same-decomposition checks
(kobayashi checksums, OpenMP thread parity, single-rank restart round-trip) are
**bitwise**. Any tolerance that must be widened during the refactor requires a
one-line justification appended to this file with the commit that widens it.
