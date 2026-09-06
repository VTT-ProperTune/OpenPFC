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
| Same PDE, spectral vs finite difference (Heat3D HIP twins) | FD HIP driver `heat3d_fd_hip` exists; Heat3D *spectral* HIP and a published science figure are still later. |
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

# 3. Same grid, 2/3/4 nodes (16/24/32 GCDs). HeFFTe slabs (use_pencils=false);
#    JSON sessions also drop pencils when nproc >= 9.
TUNGSTEN_LX=768 PARTITION=standard-g ./docs/lumi_slurm/submit_tungsten_hip_scaling.sh multinode

# 4. 3D FD HIP twin (device halo + stencil), same node counts.
export HEAT3D_HIP_BIN=/flash/project_462001519/juaho/build/<tree>/apps/heat3d/heat3d_fd_hip
./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh size
HEAT3D_N=256 PARTITION=standard-g ./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh strong
HEAT3D_N=256 PARTITION=standard-g ./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh multinode
```

`PARTITION` defaults to `small-g`. Use `dev-g` for bring-up. `standard-g` is
the full-node queue; it is not required for a 1–8 GCD single-node curve.

Each job writes `input.toml`, a copy of the sbatch script, `run_meta.txt`,
and `timing_profile.json` in its run directory. Keep the Slurm job id with
those files. Do not commit profiles.

Inputs: [`docs/lumi_slurm/tungsten_hip_scaling.toml`](../lumi_slurm/tungsten_hip_scaling.toml)
(`saveat = -1`, no `[[fields]]`). The wrapper substitutes `__LX__` and
`__T1__`. The 8-GCD CPU map is used only when `ntasks` is 8; 1–4 GCD
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

Pencil `p2p_plined` (the first multi-node pins) is slower at 16 GCDs than
at 8. HeFFTe **slabs** (`use_pencils = false`, still `p2p_plined`, GPU-aware)
fix that. `alltoall` / `alltoallv` were slower. JSON sessions call
`apply_heffte_comm_scale` so `nproc >= 9` drops pencils. Campaign TOML
requests slabs. HIP `fft` exclusive tracks `wall_step` on multi-GCD runs
(`measure_barriered` is ~5–9 ms, not the 16-GCD dip). 1-GCD `fft` timers
remain untrusted.

Slabs 768³, 10 steps, I/O off, median `wall_step` after warmup, HIP tree
`openpfc-lumi-rocm-scale` (2026-09-06):

| GCDs | Nodes | Partition | Job | Median `wall_step` | Speedup | Efficiency |
|------|-------|-----------|-----|--------------------|---------|------------|
| 1 | 1 | `standard-g` | 21761281 | 851 ms | 1.00 | 100% |
| 8 | 1 | `standard-g` | 21761220 | 216 ms | 3.95 | 49% |
| 16 | 2 | `standard-g` | 21761221 | 182 ms | 4.68 | 29% |
| 24 | 3 | `standard-g` | 21761282 | 150 ms | 5.69 | 24% |
| 32 | 4 | `standard-g` | 21761283 | 122 ms | 6.96 | 22% |

Pins: `tests/baselines/perf/lumi-standard-g-tungsten-hip-slabs-{1,8,16,24,32}gcd-release-768.json`.
`SPECTRAL_CHECKSUM` 1 vs 16 GCD agrees to ~1e-12 relative. 1024³ and 896³
OOM on one GCD; 832³ fits but 16-GCD efficiency stays ~29% (FFT transpose
volume scales with \(N^3\)).

`submit_tungsten_hip_scaling.sh multinode` launches 16/24/32 GCD jobs on
`standard-g` (8 ranks per node, same CPU map per node). 24 GCDs (3 nodes)
starts (grid 2×3×4).

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
