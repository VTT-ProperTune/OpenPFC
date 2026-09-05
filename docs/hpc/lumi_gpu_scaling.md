<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# LUMI GPU scaling campaign (`tungsten_hip`)

This page is the first slice of issue `#87`: make `tungsten_hip` busy on one
LUMI-G GCD, then strong-scale that grid across 1, 2, 4, and 8 GCDs on one
node. It is a campaign recipe, not a completed curve. Numbers belong in
`BASELINES.md` and under scratch once jobs have run; they are not invented
here.

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
| Same PDE, spectral vs finite difference (Heat3D HIP twins) | Not started. Needs HIP drivers that do not exist yet. |
| Production FD envelope (`kobayashi_fd_hip`, 2D) | Later `#87` slice. Different PDE; report cells/s, not “FD is faster”. |
| Multi-node (>8 GCD) and LUMI-C CPU control | After the one-node GPU curve. |
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
TUNGSTEN_LX=512 ./docs/lumi_slurm/submit_tungsten_hip_scaling.sh strong
```

`PARTITION` defaults to `small-g`. Use `dev-g` for bring-up. `standard-g` is
the full-node queue; it is not required for a 1–8 GCD single-node curve.

Each job writes `input.toml`, a copy of the sbatch script, `run_meta.txt`,
and `timing_profile.json` in its run directory. Keep the Slurm job id with
those files. Do not commit profiles.

Inputs: [`docs/lumi_slurm/tungsten_hip_scaling.toml`](../lumi_slurm/tungsten_hip_scaling.toml)
(`saveat = -1`, no `[[fields]]`). The wrapper substitutes `__LX__` and
`__T1__`. Binding follows the LUMI-G one-rank-per-GCD map; GPU-aware MPI is
`MPICH_GPU_SUPPORT_ENABLED=1`.

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

1. Record job ids, the chosen `Lx`, and the 1–8 GCD table in `BASELINES.md`.
2. Extend the same recipe off-node (16, 32, … GCDs) until efficiency or
   memory stops the run.
3. Add the FD envelope (`kobayashi_fd_hip`) and, when HIP twins exist, the
   Heat3D same-PDE comparison.

## See also

- [LUMI Slurm guide](../lumi_slurm/README.md)
- [Performance profiling](performance_profiling.md)
- [GPU path decision](gpu_path_decision.md)
