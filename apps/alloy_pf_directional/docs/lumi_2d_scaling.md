<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Al-Cu FTA 2D scaling on LUMI (steps 0–3)

Campaign for the regular-grid FTA solver (`alloy_pf_directional_openmp`, `alloy_pf_directional_mpi`,
`alloy_pf_directional_hip`). Ji 2022 isotropic stencil stays on. Default store is \(e^u\)/\(u\);
PNG/VTK off in timed windows. This is **2D only** (`Nz=1`, `n_dim=2`). Do not
start 3D production bricks or the moving window until the step-3 gate below
says kernel-bound.

Parent contract: [scalability analysis plan](../../../docs/hpc/scalability_analysis_plan.md).

## Physics family

Laptop DS brick at default \(W_0=5\,\mathrm{nm}\), \(\Delta x/W_0=1\):

| GRID | \(L_x\times L_y\) | \(N_x\times N_y\) |
|------|-------------------|-------------------|
| `1280x160` | \(6.40\times0.80\,\mu\mathrm{m}\) | 1280×160 |
| `2560x320` | \(12.80\times1.60\,\mu\mathrm{m}\) | 2560×320 |
| `5120x640` | \(25.60\times3.20\,\mu\mathrm{m}\) | 5120×640 |
| `3600x1280` | \(18.0\times6.40\,\mu\mathrm{m}\) | 3600×1280 |
| `7200x2560` | same \(L\), \(W_0=2.5\,\mathrm{nm}\) | 7200×2560 |
| `10240x1280` | \(51.20\times6.40\,\mu\mathrm{m}\) | 10240×1280 |
| `20480x2560` | \(102.40\times12.80\,\mu\mathrm{m}\) | 20480×2560 |
| `w0_10nm` | same \(L\) as laptop, \(W_0=10\,\mathrm{nm}\) | 640×80 |

Noise off, one grain, periodic \(y\), stop-on-right off. Env helper:
`scripts/alcu_2d_env.sh` / `alcu_2d_apply_grid`.

## Step 0 — correctness

**CPU (this laptop or LUMI-C):** OpenMP vs MPI `np=1`, ~800 steps, I/O off.

```bash
BUILD=builds/macos-cpu-release STEPS=800 \
  ./apps/alloy_pf_directional/scripts/check_nz1_vs_2d.sh
# LUMI-C:
sbatch apps/alloy_pf_directional/scripts/lumi_step0_cpu.sh
```

**GPU (LUMI-G only):** 1 GCD, `alloy_pf_directional_hip` vs the CPU `ALCU_VERIFY` log.
Always `MPICH_GPU_SUPPORT_ENABLED=1`, one rank per GCD. If GPU-aware MPI
storms, `OPENPFC_HIP_FORCE_PACKED_HALO=1` (the name in
`padded_device_halo_exchange_gpu.hpp`).

```bash
sbatch apps/alloy_pf_directional/scripts/check_hip_vs_cpu.sh
```

## Production OpenMP profile (LUMI-C, Aug 2026)

Queued/finished `alloy_pf_directional_openmp` DS jobs print `ALCU_PERF` over the whole
loop. On meshes \(\gtrsim 10^6\) cells with 16–32 threads, **serial
`fill_ghosts` z-plane copies** (`Nz=1` still copied two full \(N_x\times N_y\)
planes per field) were \(\sim 38\)–\(43\%\) of wall time. Next were iso
solute (\(\sim 23\)–\(26\%\), including two unused z-planes in the nodal
fill) and two `refresh_eu_u` passes (\(\sim 17\)–\(21\%\), each ending
in the same z-copies). Grain + Euler kernels were \(\le 16\%\) combined.
I/O was \(\lt 0.1\%\).

The OpenMP engine now skips z-ghost fill and extra nodal z-planes when
`Nz=1`, and skips grain-2 field fills when `n_grains=1`. MPI 2D uses
`Full2D` (not 26-neighbour `Full3D`) and one rank per core
(`OMP_NUM_THREADS=1`). Rebuild before the large CPU ladder; do not
overwrite the installed binary while the Lx18 physics jobs are still
queued unless those runs should pick up this change.

## Step 1 — 2D CPU (LUMI-C, `small`)

Warm-up 10 + 50–80 timed steps. OpenMP 8/16/32 threads on one node (not a
full exclusive 128-core node). MPI is **1 rank/core** (`--cpus-per-task=1`);
the MPI engine is not OpenMP-parallel.

```bash
# or the sweep:
./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --cpu-only
# single point:
MODE=strong GRID=1280x160 sbatch --nodes=1 --ntasks=32 --cpus-per-task=1 \
  apps/alloy_pf_directional/scripts/lumi_scale_cpu.sh
```

## Step 1b — large 2D CPU (`standard`, up to project capacity)

`standard` is by-node, max **512 nodes** (128 cores each). Project
`project_462001519` has a large unused CPU allocation; still start at
`--nodes-max=64` (8192 cores) and raise only if efficiency stays high.

```bash
# OpenMP 16–128 threads + MPI strong 20480×2560, 1–64 nodes:
./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --cpu-large
# later, if 64-node efficiency is still kernel-bound:
./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --cpu-large --nodes-max=256
```

Do **not** start 3D from this ladder. 3D is a separate go-ahead.

## Step 2 — 2D GPU (LUMI-G, `small-g`)

1 rank/GCD, 8 GCDs/node. Strong: 1 → 8 → 16 GCDs. Weak: grow \(N_y\) with GCD count.

```bash
./apps/alloy_pf_directional/scripts/lumi_submit_2d_scale.sh --gpu-only
MODE=strong GRID=1280x160 sbatch --nodes=1 --ntasks=8 --gpus-per-node=8 \
  apps/alloy_pf_directional/scripts/lumi_scale_gpu.sh
```

HIP build (this project tree, not the juaho `scripts/build.sh` flash path):

```bash
./apps/alloy_pf_directional/scripts/lumi_build_hip.sh configure   # login
sbatch apps/alloy_pf_directional/scripts/lumi_build_hip.sh        # compile on GPU
```

## Step 3 — decision gate (analysis only)

```bash
python3 apps/alloy_pf_directional/scripts/analyze_2d_scaling.py \
  results/alloy_pf_directional_nz1_check \
  /scratch/project_462001519/tpinomaa/alcu_fta/scale_2d
```

| Observation | Action |
|-------------|--------|
| `halo_pct` &lt; 25% at 8–16 GCDs | **kernel-bound** → proceed to 3D GPU |
| `halo_pct` ≥ 40% at 8 GCDs | **halo-bound** → fix device halo before 3D |
| GPU logs missing | wait on LUMI-G; use local OpenMP vs MPI as the CPU baseline |

Do **not** start 8000-long 3D jobs from this campaign.
