<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Scientific & performance baselines (Pre-M0 / audit §16)

This directory holds the regression contract for the OpenPFC 0.2 refactor: the
reference results the big-bang refactor (M0→M12) must preserve. Each baseline is
classified **bitwise** (must reproduce exactly) or **tolerance** (must reproduce
within a stated numeric tolerance, per backend).

> Status: tohtori CUDA/host goldens and perf JSON are captured (see tables).
> LUMI HIP execution is green: job 21683330 (`standard-g`, 2026-09-02) and
> job 21685558 (`dev-g`, 2026-09-03, after Gen-1 Model/Simulator deletion):
> 58/58 CTest batches passed (1 skip: `HIP_ExchangeFailClosed` with
> `OpenPFC_MPI_HIP_AWARE=ON`). Includes `HIP_TungstenETD`, `HIP_AluminumETD`,
> `session-matrix-hip`, allen_cahn/wave2d CPU-vs-HIP, `kobayashi-hip-hex-smoke`
> / `kobayashi-hip-hex-2rank`, and OpenMP HEX with 2 ULP on T checksums.
> Remaining ☐: none for LUMI HIP science/perf JSON pins (Kobayashi HIP
> schema-v2 captured on job 21689652). Host-staged Kobayashi HIP driver is
> gone; the JSON is the 0.2 device-path pin, not a vs-host-staged delta.
> Halo HIP microtiming JSON captured (job 21685573). Cray GPU-aware MPI
> log-assert: job 21685573 `MPICH_GPU_SUPPORT_ENABLED=1` and
> `[verify_gpu_aware_mpi] OK: device-buffer MPI_Send/Recv succeeded.`
> CTest `HIP_VerifyGpuAwareMpi` added (needs a follow-up HIP rebuild).

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
| Aluminum 0.2 vs Gen-1 4-rank trajectory (`aluminumTest` `[golden][MPI]`) | tolerance | ≤1e-10 local; Σψ² 1e-12 relative | MPI 4-rank 16³/20 steps (`aluminum-golden-4rank`) |
| CPU-side goldens for CPU-vs-GPU parity configs | tolerance | 1e-10 relative (checksum) | CI (CPU): `tungsten-cpu-golden`, `allen-cahn-cpu-golden`, `wave2d-cpu-golden` |
| Restart-equivalence (`CheckpointService`) | bitwise (1 rank) | exact owned cells | CI: kernel `[checkpoint][service]`; heat3d FD Euler; tungsten ETD JSON session. 2-rank kernel `[checkpoint][MPI]`. |

## Performance baselines

In-tree pins under `tests/baselines/perf/` are **schema v4 summaries** (mean /
median / min / max of frame scalars after warmup, plus top-level region
exclusive means). They are not full per-frame traces. Cluster jobs still
export schema v2/v3 traces (write those under `results/`); collapse a new pin
with:

```
python3 scripts/compare_perf_baseline.py --summarize path/to/new_profile.json \
  --warmup-frames=N -o tests/baselines/perf/<baseline>.json
```

Compare a stored pin to a new full export with `scripts/compare_perf_baseline.py`
(pass ≤5% regression / warn >5% / fail >15%; speedups pass). `--warmup-frames`
applies to the v2/v3 *current* file; the pin already baked that skip into
`metrics.*.mean`:

```
python3 scripts/compare_perf_baseline.py \
  tests/baselines/perf/<baseline>.json path/to/new_profile.json \
  --warmup-frames=N
```

Canary input: `tests/baselines/perf/inputs/tungsten_canary.json` (64³, 20 steps,
`saveat=-1`, profiling JSON). Capture on tohtori with `tungsten` / `tungsten_cuda`
from a Release or Debug tree; copy the exported `*.json` next to this file with a
machine tag in the name (`tohtori-g0005-tungsten-cuda-1rank.json`).

- ☑ Tungsten strong scaling, CPU, 1/4/16 ranks (tohtori `g0005` Release, 64³ / 20 steps, `tungsten`, 2026-09-01): `tests/baselines/perf/tohtori-g0005-tungsten-cpu-{1,4,16}rank-release-64.json`. Mean `wall_step` after `--warmup-frames=1`: 0.00630 s / 0.00177 s / 0.000910 s. Input: `tests/baselines/perf/inputs/tungsten_canary.json`. 64³ is small; 16-rank is communication-heavy.
- ☑ Tungsten CUDA 1-rank Debug canary (tohtori `g0005`, 64³ / 20 steps, `tungsten_cuda`, 2026-09-01): `tests/baselines/perf/tohtori-g0005-tungsten-cuda-1rank-debug-canary.json`. Compare with `--warmup-frames=1` (first step includes CUDA context setup). This is a plumbing pin, not a production scaling number.
- ☑ Tungsten CPU 1-rank Debug canary (same grid/steps, `tungsten`): `tests/baselines/perf/tohtori-g0005-tungsten-cpu-1rank-debug-canary.json`.
- ☑ Tungsten CUDA single node Release (tohtori `g0005`, 256³ / 20 steps, `tungsten_cuda`, 2026-09-01): `tests/baselines/perf/tohtori-g0005-tungsten-cuda-1rank-release-256.json` (mean `wall_step` 0.0253 s after `--warmup-frames=1`) and 8-rank `tohtori-g0005-tungsten-cuda-8rank-release-256.json` (0.0240 s, 8×H100, one GPU per rank). Input: `tests/baselines/perf/inputs/tungsten_release_256.json`. 8-rank step 1 is ~6 s of setup; compare with `--warmup-frames=1`. This size is still latency-bound on H100 (8-rank does not speed up vs 1-rank).
- ☑ 0.2 ETD vs Gen-1 (tohtori `g0005` Release). Compare with `--warmup-frames=1`. Remeasured 2026-09-02 after JSON HeFFTe `plan_options` overlay on `GPUSpectralStack` and per-rank `cudaSetDevice`.

| Case | Gen-1 `wall_step` | 0.2 ETD | vs Gen-1 | JSON |
|------|-------------------|---------|----------|------|
| CPU 1-rank 64³ / 20 | 0.00630 s | 0.00850 s | +35% | `tohtori-g0005-tungsten-etd-cpu-1rank-release-64.json` |
| CUDA 1-rank 256³ / 20 | 0.0253 s | 0.00543 s | −79% (faster) | `tohtori-g0005-tungsten-etd-cuda-1rank-release-256.json` |
| CUDA 8-rank 256³ / 20 | 0.0240 s | 0.00253 s | −89% (faster) | `tohtori-g0005-tungsten-etd-cuda-8rank-release-256.json` |

Science A/B (32³/10-step sine) holds. CPU 64³ is a tiny grid; the extra session/profiling frame cost is ~2 ms/step. CUDA 1-rank and 8-rank are both faster than Gen-1. Before this remasure, 8-rank was 0.0864 s/step because every rank used GPU 0 and HeFFTe defaults (no `use_gpu_aware` / `p2p_plined`). Input: `tests/baselines/perf/inputs/tungsten_release_256.json`.

Gen-1 tungsten sources deleted on g0005 (2026-09-02). DoD greps: no `public pfc::Model` under `apps/tungsten`, no `*model*` headers, no app `.cu`/`.hip`, no `dummy_fft`. Non-test line count under `apps/tungsten/` (exclude `tests/`, `inputs_*`) is 2335 — above the 1500 sketch; leftover is one physics/session/IO stack plus CLI, not a second model.
- ☑ Halo-exchange microtimings, host and CUDA, 2/4/8 ranks (tohtori `g0005` Release, 128³ Faces, 50 timed exchanges, `examples/23_halo_microtiming`, 2026-09-01): `tests/baselines/perf/tohtori-g0005-halo-{host,cuda}-{2,4,8}rank-release-128.json`. Mean `wall_step` after `--warmup-frames=5`: host 78.4 / 118 / 67.9 µs; CUDA 5.78 / 7.13 / 3.93 ms (GPU-aware MPI).
- ☑ Halo-exchange microtimings, HIP, 2/4/8 ranks (LUMI `dev-g` nid005008, job 21685573, 2026-09-03, 128³ Faces, 50 timed exchanges, `examples/23_halo_microtiming --hip`): `tests/baselines/perf/lumi-dev-g-halo-hip-{2,4,8}rank-release-128.json`. Mean `wall_step` after warmup 5: 105 / 127 / 132 µs (`OpenPFC_MPI_HIP_AWARE=ON`, `MPICH_GPU_SUPPORT_ENABLED=1`).
- ☑ Cray GPU-aware MPI log-assert (LUMI `dev-g` job 21685573): `verify_gpu_aware_mpi` 2-rank device-buffer `MPI_Send`/`MPI_Recv` printed `MPICH_GPU_SUPPORT_ENABLED=1` and `OK: device-buffer MPI_Send/Recv succeeded.`
- ☑ Kobayashi HIP single-node perf JSON (LUMI `dev-g` job 21689652, 2026-09-03, `kobayashi_fd_hip` device path, GPU-aware MPI, 1 rank, 256×256 / 200 steps, `--warmup 20`): `tests/baselines/perf/lumi-dev-g-kobayashi-hip-1rank-release-256.json`. Mean `wall_step` ~0.30 ms (180 timed frames). Whole-loop `wall_loop_max_s=0.0623`. HEX CTest still pins 32²/4-step (`kobayashi-hip-hex-smoke` / `-2rank`). The Pre-M0 host-staged HIP driver no longer exists, so this is the 0.2 device-path pin rather than a vs-host-staged speedup table.

## How to capture (reference commands)

CPU golden trajectory (Gen-1 `Tungsten` vs `tungsten_etd` / `TungstenSession`, i.e. `pfc::ui::SpectralETDSession<TungstenPhysics, SpectralCPUStack>`):

> **2026-09-03 consolidation:** the six ETD drivers and four app sessions were folded into `SpectralETDSystem` + `SpectralETDSession`. Re-confirmed on the consolidated tree (LUMI, Cray CC): `tungsten-etd-cpu-golden` 32³/10-step sine IC `sum=-13107.200000000043`, `sumsq=5406.3450894885691` (pin 5406.3450894885682, within 1e-10 rel); `aluminum-etd-cpu-golden` seed_grid_fcc 32³/5-step `sum=-263.63079658808613`, `sumsq=1111.6016268617182` (pins within 1e-10 rel). CPU Debug job 21690851: 39/39 CTest incl. `tungsten-golden-4rank`, `aluminum-golden-4rank`. HIP Release job 21690849: 57/57 CTest incl. `HIP_SpectralETD` (plain, mean-field, moving-frame toys device vs host ≤1e-10), `HIP_TungstenETD`, `HIP_AluminumETD` (device nonlinearity, G_grid case). CUDA not re-run (Tohtori).

```
# 1-rank (CI): ctest -R tungsten-all-tests
# 4-rank MPI suite:
mpiexec -n 4 ./apps/tungsten/test_tungsten "[golden][MPI]"
```

A 4-rank / 100-step capture of `tungsten_etd` on tohtori is still optional for archival binaries under `tests/baselines/`; the Catch2 A/B is the gate.

CPU-side checksums of the CPU-vs-GPU parity configs (Tohtori `g0005`, gcc 15.2
Debug, 2026-09-02). CI can run these without a GPU:

```
ctest -R 'tungsten-cpu-golden|allen-cahn-cpu-golden|wave2d-cpu-golden'
```

Captured `sum` / `sumsq` (and wave2d `u`/`v`):

```
tungsten  32³ / 10 steps / dt=0.01   n=32768  sum=-13107.200000000043  sumsq=5406.3450894885682
tungsten_etd (0.2 session, same IC)  n=32768  sum=-13107.200000000043  sumsq=5406.3450894885682  (g0005 Debug; CTest tungsten-etd-cpu-golden / tungsten-etd-cpu-vs-cuda)
aluminum_etd 32³ / 5-step SeedGridFCC (rseed=42)  n=32768  sum=-263.63079658808601  sumsq=1111.6016268617182  (g0005 Debug; CTest aluminum-etd-cpu-golden)
allen_cahn 32² / 20 steps / dt=0.002 n=1024   sum=-967.34722270794146  sumsq=961.14566882919667
wave2d    24² / 8 steps / dt=0.01    n=576    sum_u=56.542106624911966 sumsq_u=28.256988690744471
                                              sum_v=0.0018563899833072017 sumsq_v=0.0042855033181676445
```

GPU compile + run parity (cluster only):

```
scripts/build.sh --with-cuda  --build-dir=/WRK/<user>/openpfc/builds/cuda-release   # tohtori
scripts/build.sh --with-rocm  --build-dir=/WRK/<user>/openpfc/builds/rocm-release   # LUMI
# then run test_tungsten_cpu_vs_cuda / _hip and HIP_TungstenETD / CUDA_TungstenETD
# (App-GPU-IC is the [ic] case on TungstenCUDASession / TungstenHIPSession)
# Kobayashi CUDA HEX smoke (32² / 4 steps): ctest -R kobayashi-cuda-hex
```

Kobayashi CUDA HEX (Tohtori `g0005`, 2026-09-01, gcc 15.2 / CUDA 13.1 / CUDA-aware
Open MPI, `HaloExchange<CUDASpace>` library path, 1-rank and 2-rank identical):

```
KOBAYASHI_VERIFY_HEX sum_phi=0x1.b96bf451009d9p+3 sumsq_phi=0x1.4e770b1504ae4p+3 sum_T=0x1.6e128af4d5ac5p+0 sumsq_T=0x1.6546ee0a021fp-2
```

CPU OpenMP/MPI pin for the same `(Nx, Ny, steps, dt, dx)` has `sum_T=0x1.6e128af4d5ac6p+0`
on Tohtori gcc. LUMI Cray GNU OpenMP (job 21683102) matches phi bitwise and is
2 ULP on `sumsq_T` (1 ULP on `sum_T`). The OpenMP Catch2 case allows 2 ULP on
those two. Cross-backend last-bit drift is expected; CUDA and HIP pins are bitwise for
their own drivers so library-halo changes fail closed.

Kobayashi HIP HEX (LUMI-G MI250X, job 21682844, 1-rank and 2-rank identical):

```
KOBAYASHI_VERIFY_HEX sum_phi=0x1.b96bf451009d9p+3 sumsq_phi=0x1.4e770b1504ae4p+3 sum_T=0x1.6e128af4d5ac8p+0 sumsq_T=0x1.6546ee0a021efp-2
```

## Why bitwise vs tolerance

Consolidating kernels during the refactor (reduction order, FMA contraction,
CPU↔GPU math) perturbs the last bits, so cross-backend and post-refactor checks
are **tolerance**-based. Only same-binary, same-decomposition checks
(kobayashi checksums, OpenMP thread parity, single-rank restart round-trip) are
**bitwise**. Any tolerance that must be widened during the refactor requires a
one-line justification appended to this file with the commit that widens it.
