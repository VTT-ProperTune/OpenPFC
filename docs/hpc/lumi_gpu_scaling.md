<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI GPU scaling campaign (`tungsten_hip`)

This page is the first slice of issue `#87`: make `tungsten_hip` busy on one
LUMI-G GCD, then strong-scale that grid across 1, 2, 4, and 8 GCDs on one
node. The one-node spectral curve below was measured on 2026-09-06. Raw
profiles stay under scratch; schema-v4 summaries are in
`tests/baselines/perf/`.

The reporting contract is [Scalability analysis plan](scalability_analysis_plan.md).
Install and GPU-aware MPI notes are in [INSTALL.LUMI.md](INSTALL.LUMI.md).

## What this slice answers

For the shipped 3D spectral ETD app `tungsten_hip`, on LUMI-G (MI250X, HIP /
rocFFT, one MPI rank per GCD):

1. Which cubic grid makes one GCD compute-bound (`wall_step` clearly above
   launch and sync noise, with memory recorded)?
2. How does time per accepted step change from 1 to 8 GCDs on that global
   grid (strong scaling, I/O off)?

In-tree pins (256³ CUDA on Tohtori, 64³ CPU) are too small for this question.
The older `docs/lumi_slurm/tungsten_gpu.sbatch` path is a 0.1.4 full-node
1024³ example on `project_462001245` scratch; do not use it for this campaign.

## What this slice does not answer

Keep these comparisons separate, as `#87` states:

| Comparison | Status in this slice |
|------------|----------------------|
| Same PDE, spectral vs finite difference (Heat3D HIP twins) | Both HIP drivers exist (`heat3d_fd_hip`, `heat3d_spectral_hip`). A published science figure is still later. |
| Production FD envelope (`kobayashi_fd_hip`, 2D) | Later `#87` slice. Different PDE; report cells/s, not “FD is faster”. |
| Multi-node (>8 GCD) and LUMI-C CPU control | Multi-node GPU jobs are in the submit helper (16/24/32 GCD). LUMI-C CPU control is still later. |
| Float GPU path | `#11`, not this campaign. Precision is double. |

## How to run (LUMI login node)

Build a 0.2 HIP tree with [`scripts/build.sh`](../../scripts/build.sh)
(`--machine=lumi --with-rocm`). Trees go under
`/flash/project_462001519/juaho/build/`. Point `TUNGSTEN_HIP_BIN` at that
`tungsten_hip`.

Account is `project_462001519`. Job logs go to
`/scratch/project_462001519/juaho/logs/`. Per-job working directories go to
`/scratch/project_462001519/juaho/openpfc-scaling/runs/` (override with
`OPENPFC_SCALING_ROOT`).

```bash
export TUNGSTEN_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/tungsten/tungsten_hip

# 1. Size one GCD: 256³ … 768³, 20 steps, I/O off, profiling on.
./docs/lumi_slurm/submit_tungsten_hip_scaling.sh size

# 2. After picking Lx (wall_step busy, memory fits), strong-scale 1/2/4/8 GCDs.
TUNGSTEN_LX=768 ./docs/lumi_slurm/submit_tungsten_hip_scaling.sh strong

# 3. Same grid, 2/3/4 nodes (16/24/32 GCDs). Off-node uses 1D z-slabs
#    in real space. The r2c complex outbox is chosen by HeFFTe reshape
#    cost (y-slabs when Ny divides nproc; otherwise a legal gz=1 grid).
TUNGSTEN_LX=768 PARTITION=standard-g ./docs/lumi_slurm/submit_tungsten_hip_scaling.sh multinode

# 4. 3D FD HIP twin (device halo + stencil), same node counts.
export HEAT3D_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/heat3d/heat3d_fd_hip
./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh size
HEAT3D_N=256 PARTITION=standard-g ./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh strong
HEAT3D_N=256 PARTITION=standard-g ./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh multinode

# 5. 3D spectral HIP twin (implicit Euler, 2 FFTs/step), same node counts.
export HEAT3D_SPECTRAL_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/heat3d/heat3d_spectral_hip
./docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh size
HEAT3D_N=768 PARTITION=standard-g ./docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh strong
HEAT3D_N=768 PARTITION=standard-g ./docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh multinode
```

`PARTITION` defaults to `small-g`. Use `dev-g` for bring-up. `standard-g` is
the full-node queue; it is not required for a 1–8 GCD single-node curve.

The scaling sbatch does not mask GCDs with `ROCR_VISIBLE_DEVICES`.
`bind_local_device()` picks `local_rank % n_devices` while every GCD stays
visible so GPU-aware HeFFTe can use intra-node IPC. Set
`OPENPFC_FFT_SLAB_AXIS` to `x`, `y`, or `z` to force the 1D slab split.

Each job writes `input.toml`, a copy of the sbatch script, `run_meta.txt`,
and `timing_profile.json` in its run directory. Keep the Slurm job id with
those files. Do not commit profiles.

Inputs: [`docs/lumi_slurm/tungsten_hip_scaling.toml`](../lumi_slurm/tungsten_hip_scaling.toml)
(`saveat = -1`, no `[[fields]]`). The wrapper substitutes `__LX__` and
`__T1__`. The 8-GCD CPU map is used only when `ntasks-per-node` is 8; 1–4 GCD
allocations skip it. GPU-aware MPI is `MPICH_GPU_SUPPORT_ENABLED=1`.

## Measured one-node curve (2026-09-06)

HIP Release `tungsten_hip` from
`/flash/project_462001519/juaho/build/openpfc-lumi-rocm-0.2` (commit
`ce2060db`, OpenPFC 0.2.0). Double precision. I/O off (no `fields[]`). 10
accepted steps, `dt = 1`. Median `wall_step` after dropping step 1 (plan /
first-touch). Speedup and efficiency vs 1 GCD. `p` is GCD count (= MPI ranks,
one per GCD).

1-GCD sizing (same binary, `dev-g`):

| Lx | Job | Median `wall_step` |
|----|-----|--------------------|
| 256 | 21759671 | 20 ms |
| 384 | 21759672 | 79 ms |
| 512 | 21759709 | 232 ms |
| 640 | 21759710 | 466 ms |
| 768 | 21759720 | 847 ms |

256³ is still launch-noise class. 768³ is the strong-scaling grid.

| GCDs | Nodes | Partition | Job | Median `wall_step` | Speedup | Efficiency |
|------|-------|-----------|-----|--------------------|---------|------------|
| 1 | 1 | `dev-g` | 21759720 | 847 ms | 1.00 | 100% |
| 2 | 1 | `dev-g` | 21759944 | 455 ms | 1.86 | 93% |
| 4 | 1 | `dev-g` | 21759945 | 315 ms | 2.69 | 67% |
| 8 | 1 | `standard-g` | 21759946 | 215 ms | 3.94 | 49% |
| 16 | 2 | `standard-g` | 21760377 | 222 ms | 3.82 | 24% |
| 32 | 4 | `standard-g` | 21760378 | 103 ms | 8.19 | 26% |

Schema-v4 summaries: `tests/baselines/perf/lumi-dev-g-tungsten-hip-{1,2,4}gcd-release-768.json` and
`lumi-standard-g-tungsten-hip-{8,16,32}gcd-release-768.json`. Compare with
`--warmup-frames=1`. Use median `wall_step`: the 4-GCD run had one
collective stall (step 3 ≈ 11.4 s on every rank), so the mean is not a
steady-state number.

HIP `fft` region timers after step 1 are under-counted on the 1-GCD path
(sub-millisecond) and should not be used to explain the curve. `wall_step` is
the metric.

Pencil `p2p_plined` on a min-surface 2×2×4 grid was slower at 16 GCDs than
at 8. Real-space 1D z-slabs with a **y-slab complex outbox** (full z per
rank, used when `Ny` divides `nproc`) drop HeFFTe's extra
pencils-back-to-z-slabs reshape. When `Ny` does not divide, the
production selector keeps that hop skip by choosing another legal
`gz=1` complex grid (issue #45). Combined with
leaving every GCD visible (no `ROCR_VISIBLE_DEVICES`) and pinning the
device before `MPI_Init`, 16-GCD 768³ is 84 ms (64% vs 1 GCD). 1×8×N
pencils and `alltoall` / `alltoallv` were slower. HIP `fft` exclusive
tracks `wall_step` on multi-GCD runs. 1-GCD `fft` timers remain untrusted.

Slabs 768³, 10 steps, I/O off, median `wall_step` after warmup, HIP tree
`openpfc-lumi-rocm-prebind` (2026-09-06). Efficiency vs 1 GCD job
21761281 (851.215 ms; 1-GCD layout is unchanged):

| GCDs | Nodes | Partition | Job | Median `wall_step` | Speedup | Efficiency |
|------|-------|-----------|-----|--------------------|---------|------------|
| 1 | 1 | `standard-g` | 21761281 | 851 ms | 1.00 | 100% |
| 8 | 1 | `standard-g` | 21764313 | 177 ms | 4.81 | 60% |
| 16 | 2 | `standard-g` | 21764315 | 84 ms | 10.2 | 64% |
| 24 | 3 | `standard-g` | 21764316 | 72 ms | 11.9 | 50% |
| 32 | 4 | `standard-g` | 21764317 | 63 ms | 13.6 | 42% |

Repeat 16 GCD job 21764314 was 85 ms. Pins:
`tests/baselines/perf/lumi-standard-g-tungsten-hip-slabs-{1,8,16,24,32}gcd-release-768.json`.
`SPECTRAL_CHECKSUM` 1 vs 16 GCD agrees to ~1e-12 relative.

`submit_tungsten_hip_scaling.sh multinode` launches 16/24/32 GCD jobs on
`standard-g` (8 ranks per node). 24 GCDs (3 nodes) starts.

### 3D FD HIP (`heat3d_fd_hip`)

Same node counts, device `HaloExchange` + stencil, I/O off, 512³ / 20
steps / `dt=0.01` / `fd_order=2`. GPU-aware MPI. Median `wall_step` after
warmup:

| GCDs | Nodes | Partition | Job | Median `wall_step` | Speedup | Efficiency |
|------|-------|-----------|-----|--------------------|---------|------------|
| 1 | 1 | `dev-g` | 21761036 | 5.91 ms | 1.00 | 100% |
| 8 | 1 | `standard-g` | 21761037 | 0.951 ms | 6.22 | 78% |
| 16 | 2 | `standard-g` | 21761038 | 0.567 ms | 10.4 | 65% |
| 24 | 3 | `standard-g` | 21761039 | 0.476 ms | 12.4 | 52% |
| 32 | 4 | `standard-g` | 21761042 | 0.394 ms | 15.0 | 47% |

Pins: `tests/baselines/perf/lumi-dev-g-heat3d-fd-hip-1gcd-release-512.json` and
`lumi-standard-g-heat3d-fd-hip-{8,16,24,32}gcd-release-512.json`.
`HEAT3D_HIP_CHECKSUM` 1 vs 16 GCD agrees to ~5e-15 relative. Submit with
`submit_heat3d_fd_hip_scaling.sh`.

Two-stream overlap A/B on the same 512³ protocol (issue #48, binary
`c01edbba…`, 20 steps, warmup 1). Do not replace the table above.

| GCDs | ov=0 | ov=1 | jobs |
| ---: | ---: | ---: | ---- |
|    1 | 5.955 ms | 6.418 ms | 22162498 / 22162499 |
|    8 | 0.934 ms | **0.856 ms** | 22162500 / 22162501 |
|   16 | 0.558 ms | **0.473 ms** | 22162547 / 22162548 |
|   32 | 0.374 ms | **0.344 ms** | 22162549 / 22162550 |

1 GCD has no MPI halo: forcing the inner/border split is 8% slower, so
production default overlap falls back to blocking on a single rank.
8–32 GCD overlap is faster. Strong efficiency vs blocking
$T_1=5.955$ ms:

| GCDs | ov=0 | ov=1 |
| ---: | ---: | ---: |
|    8 | 80% | 87% |
|   16 | 67% | 79% |
|   32 | 50% | 54% |

512³ saturates near 32 GCD even with overlap (local brick 256×128×128).
That is the meaningful strong-scaling endpoint for this grid: interior
work is no longer enough to hide halo plus launch overhead. Do not
replace the admitted blocking table.

### Spectral HIP (`heat3d_spectral_hip`)

HIP twin of `heat3d_spectral`: implicit Euler in Fourier space, 2 FFTs per
step (`HIPSpectralStack` + `1/(1-\Delta t D k_\mathrm{lap})`). Same CLI as
the CPU spectral driver (`<N> <n_steps> <dt>`). I/O off via
`HEAT3D_PROFILE_JSON`. Same bind/visibility as `tungsten_hip` /
`heat3d_fd_hip`. The driver uses HeFFTe slabs (`use_pencils=false`,
GPU-aware `p2p_plined`) on every rank count, matching
`tungsten_hip_scaling.toml`. Compare FFT time to the FD HIP halo+stencil
on the **same PDE**; do not claim “FD is faster than tungsten.”

768³ / 20 steps / `dt=0.01`, I/O off, median `wall_step` after warmup 1.
Efficiency vs 1 GCD job 21773962 (416 ms):

| GCDs | Nodes | Partition | Job | Median `wall_step` | Speedup | Efficiency |
|------|-------|-----------|-----|--------------------|---------|------------|
| 1 | 1 | `standard-g` | 21773962 | 416 ms | 1.00 | 100% |
| 2 | 1 | `standard-g` | 21774367 | 204 ms | 2.04 | 102% |
| 4 | 1 | `standard-g` | 21774368 | 127 ms | 3.29 | 82% |
| 8 | 1 | `standard-g` | 21774369 | 88 ms | 4.74 | 59% |
| 16 | 2 | `standard-g` | 21773966 | 50 ms | 8.34 | 52% |
| 24 | 3 | `standard-g` | 21773967 | 39 ms | 10.6 | 44% |
| 32 | 4 | `standard-g` | 21773968 | 34 ms | 12.2 | 38% |

2/4/8 jobs 21773963–21773965 used default rocFFT pencils and were slower
than 1 GCD; they are not the pin. Pins:
`tests/baselines/perf/lumi-standard-g-heat3d-spectral-hip-{1,2,4,8,16,24,32}gcd-release-768.json`.
`HEAT3D_SPECTRAL_HIP_CHECKSUM` 1 vs 16 GCD agrees to ~3e-16 relative.
Submit with `submit_heat3d_spectral_hip_scaling.sh`.

### Spectral HIP after min-reshape (issue #51; not admitted)

Current-master `heat3d_spectral_hip` (min-reshape PR #46 in the binary;
FFT path unchanged vs `origin/master`). Same 768³ / 20 / `dt=0.01` /
I/O off protocol. Efficiency vs 1 GCD job 22162399 (416 ms). Do not
replace the pre-#46 table above; that remains the historical pin.

1 GCD and 2 GCD `standard-g` match the old pins (416 / 204 ms). 4 GCD
and 8 GCD are faster (103 / 77 ms vs 127 / 88 ms). The 2 GCD `dev-g`
227 ms point was a shared-partition effect, not a min-reshape
regression. 16 GCD (first off-node, job 22162465) is 48.6 ms vs the
old 50 ms pin (53% efficiency), so the remaining drop is the
distributed transpose, not a leftover extra reshape.

| GCDs | Nodes | Partition | bind | Job | Median `wall_step` | Speedup | Efficiency |
|------|-------|-----------|------|-----|--------------------|---------|------------|
| 1 | 1 | `dev-g` | none | 22162399 | 416 ms | 1.00 | 100% |
| 2 | 1 | `standard-g` | none | 22162446 | 204 ms | 2.04 | 102% |
| 4 | 1 | `standard-g` | none | 22162447 | 103 ms | 4.03 | 101% |
| 8 | 1 | `standard-g` | CCD | 22162433 | 76.9 ms | 5.41 | 68% |
| 16 | 2 | `standard-g` | CCD | 22162465 | 48.6 ms | 8.55 | 53% |
| 2 | 1 | `dev-g` | none | 22162431 | 227 ms | 1.83 | 92% |
| 4 | 1 | `dev-g` | none | 22162432 | 105 ms | 3.95 | 99% |
| 8 | 1 | `dev-g` | none | 22162416 | 74.5 ms | 5.58 | 70% |

Checksum HEX 1 GCD `0x1.778597062215fp+3` vs 8 GCD
`0x1.77859706219b1p+3` (relative ~3e-13). `use_pencils=0`,
`gpu_aware=1`, real slabs. Binary SHA256
`0495828f6ba30e60d8d642e3dc0764083909efae0fc591a0f015ddf90662fb3a`.
Scratch:
`/scratch/project_462001519/juaho/openpfc-scaling/heat3d-spectral-fft-51`.

Job 22162400 failed: 8-GPU `dev-g` allocations do not own the exclusive
CCD CPU mask. The submit script now applies that mask only on
`standard-g`, matching `heat3d_fd_hip_weak.sbatch`.

Real process-grid A/B at 16 GCD (same 768³, `standard-g`, CCD bind,
`use_pencils=0`). Default off-node policy is `slab_proc_grid` →
`1×1×16`. Held-out general factorizations:

| grid | job | median `wall_step` |
|------|-----|-------------------:|
| `1×1×16` (default) | 22162465 | 48.6 ms |
| `1×4×4` | 22162594 | 49.4 ms |
| `2×2×4` | 22162593 | 76.7 ms |

`2×2×4` is 58% slower. `1×4×4` is a tie within noise. Do not change
`spectral_fft_proc_grid` and do not resurrect `1×8×N`. A general cost
model is not justified until it beats slabs on a held-out rank count.

Off-node `use_pencils=1` at the same 16 GCD 768³ point (job 22162734,
`HEAT3D_USE_PENCILS=1` after the slab overlay) is 48.4 ms vs slabs
48.6 ms. A tie. Keep production `use_pencils=false`. Checksum HEX
matches the slab 16 GCD run (`0x1.7785970621a73p+3`).

On-node `HEAT3D_RESHAPE_ALG=alltoall` at 8 GCD 768³ (`dev-g`, new
binary `f77bb73a…`, jobs **22165163** / **22165164**): `p2p_plined`
74.1 ms vs `alltoall` 92.8 ms (**+25%**). Checksum HEX identical
(`0x1.77859706219b1p+3`, matches the pre-override 8 GCD pin
22162433). Keep production `p2p_plined`.

Job **22166458** is **not** an off-node reshape A/B. A leaked
`HEAT3D_SPECTRAL_HIP_BIN` pointed at `c4e2f667…`, which does not
parse `HEAT3D_RESHAPE_ALG`; the four sequential 16 GCD 768³ runs are
repeat `p2p_plined` noise (~48 ms ±2%).

Held-out off-node reshape A/B (job **22166715**, `dev-g`, 2 nodes,
hardcoded binary `fba65f33…`, banner confirms `reshape=`). Same 768³
16 GCD strong protocol as 22162465. Barriered median `wall_step`:

| algorithm | median `wall_step` |
|-----------|-------------------:|
| `p2p_plined` (production) | 48.2 ms |
| `alltoall` | 53.8 ms (**+12%**) |
| `alltoallv` | 50.4 ms (+5%) |
| `p2p` | 63.2 ms (**+31%**) |

Checksum HEX `0x1.7785970621a73p+3` on all four. Production
`p2p_plined` wins on-node *and* off-node at this 16-GCD point. A
large-rank protocol tournament (issue
[#61](https://github.com/ahojukka5/OpenPFC/issues/61),
[`heffte_protocol_tournament.md`](heffte_protocol_tournament.md))
tests whether the ranking changes at hundreds or thousands of ranks.
Do not treat the 16-GCD table as that result.

On-node `use_reorder` / `gpu_aware` A/B (job **22166716**, 8 GCD
768³, same binary, banner confirms the flags):

| setting | median `wall_step` |
|---------|-------------------:|
| production (`use_reorder=1`, GPU-aware) | 75.9 ms |
| `HEAT3D_USE_REORDER=0` | 77.7 ms |
| `HEAT3D_GPU_AWARE=0` | 503 ms (**6.6×**) |

HEX `0x1.77859706219b1p+3` on all three (matches 22165163). Keep
rocFFT reorder and GPU-aware MPI. Host-staged HeFFTe is not a
production option.

Cubic weak scaling at constant *cell count* is not constant FFT work
for 1D z-slabs: 8 GCD `1536³` (job 22162696) OOM'd because the in-plane
transform grew 4×. The comparable protocol grows only the split axis
(`768×768×(768 P)`), keeping local `768³` and in-plane `768²`.

Job **22164976** OOM-killed on the default GPU-partition host quota
(`AllocTRES mem=49G`). MaxRSS ≈ 7.5 GiB/rank — the same host footprint
as 1 GCD `768³` (job 22162399, `mem=64G`) — so eight ranks needed
~60 GiB. The quota, not HBM, was the limit. `--mem=0` on `standard-g`
allocates 480 GiB and is the production submit path.

Constant-local-work weak scaling (inbox `452984832` = `768³` at every
point; I/O off; 20/1; `dt=0.01`; `use_pencils=0`). Efficiency vs 1 GCD
job 22162399 (416 ms). Binary SHA256
`c4e2f66771bf6f365d4fc201081bec73ee78e81f3f84783b94a50e9be142ac0e`.

| GCDs | grid | partition | bind | job | median `wall_step` | weak eff. |
|------|------|-----------|------|-----|-------------------:|----------:|
| 1 | `768³` | `dev-g` | none | 22162399 | 416 ms | 100% |
| 2 | `768×768×1536` | `dev-g` | none | 22165046 | 453 ms | 92% |
| 4 | `768×768×3072` | `dev-g` | none | 22165068 | 483 ms | 86% |
| 8 | `768×768×6144` | `dev-g` | none | 22165032 | 495 ms | 84% |
| 8 | `768×768×6144` | `standard-g` | CCD | **22165026** | **489 ms** | **85%** |
| 16 | `768×768×12288` | `standard-g` | CCD | **22165047** | **873 ms** | **48%** |
| 32 | `768×768×24576` | `standard-g` | CCD | **22165128** | **1146 ms** | **36%** |
| 64 | `768×768×49152` | `standard-g` | CCD | **22165443** | **1434 ms** | **29%** |

8 GCD `standard-g` HEX `0x1.7785970621e4fp+3` matches the `dev-g`
replicate. MaxRSS ≈ 7.6 GiB/rank, `AllocTRES mem=480G`. Extra vs 1 GCD:
2 GCD +37 ms, 4 GCD +67 ms, 8 GCD +73 ms — most of the on-node
transpose tax is paid by four ranks. First off-node (16 GCD / 2 nodes,
job **22165047**) is **873 ms** (48%): 1.78× the 8 GCD on-node step.
Inbox stays `768³`; MaxRSS ≈ 7.5 GiB/rank; `AllocTRES mem=960G`. The
drop is the Slingshot all-to-all, not local rocFFT. 32 GCD
`768×768×24576` (job **22165128**) is **1146 ms** (36%): 1.31× the
16 GCD step. The interconnect tax is **not** flattening after the
first off-node hop. 64 GCD `768×768×49152` (job **22165443**) is
**1434 ms** (29%): 1.25× the 32 GCD step. Doubling off-node ranks
still costs ~25–31%, but the *percentage-point* drip is shrinking
(−37 at 8→16, −12 at 16→32, −7 at 32→64). Stop here: further nodes
would repeat the same all-to-all volume growth. Local inbox stayed
`768³`; MaxRSS ≈ 7.6 GiB/rank.

## How to read a point

From each `timing_profile.json` (schema v4 summary; see
[Profiling export schema](profiling_export_schema.md)):

- `wall_step` median over accepted steps after a short warmup (the first
  frames include FFT planning);
- `fft` region vs the rest of the step;
- RSS / heap when `memory_samples` is true (sizing jobs).

For a baseline `p0 = 1` GCD:

```text
speedup(p)    = time(1 GCD) / time(p GCD)
efficiency(p) = speedup(p) / p
```

`p` is GCD count (equal to MPI ranks). Stop calling the curve “scaling” once
efficiency falls through a stated floor (for example 50%) or the job will not
start. That floor *is* the one-node limit for this problem.

Correctness: compare a cheap observable (field checksum, L2, or HEX) at 1 GCD
vs N GCD on the same grid and step count. A performance point without that
check is not a valid `#87` result.

## After this slice

1. 1-GCD vs N-GCD field checksum / L2 on the same grid and step count.
2. FD envelope (`kobayashi_fd_hip`) and, when HIP twins exist, the Heat3D
   same-PDE comparison.
3. A larger spectral grid if the goal is to push past one node with
   acceptable efficiency (768³ saturates at 8 GCDs).

## See also

- [LUMI Slurm guide](../lumi_slurm/README.md)
- [Performance profiling](performance_profiling.md)
- [GPU path decision](gpu_path_decision.md)
