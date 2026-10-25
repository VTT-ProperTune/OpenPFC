<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Heat3D one-node GPU cost matrix

Recipe for the Heat3D benchmark behind
[issue #124](https://github.com/ahojukka5/OpenPFC/issues/124). It times one
GPU node at two cubic resolutions and four operators. It does not store a
measured campaign.

## What it compares

The matrix is fixed in `apps/heat3d/scripts/n3_cost_matrix.py`:

- resolutions $N = 512$ and $N = 1024$;
- spectral slabs, FD-2, FD-8, and FD-12;
- three independent repeats of each cell.

Each admitted repeat contributes the median barriered `wall_step` from
`timing_profile.json`, after the warmup frames. `check` prints the step
count, warmup, and timestep the driver will submit.

## What you need

- A HIP build that produces `heat3d_spectral_hip` and `heat3d_fd_hip`.
- One GPU node, eight MPI ranks, one rank per GCD, GPU-aware MPI.
- Spectral cells on the production slab layout (`use_pencils` off).
- A clean source tree. The harvester rejects a dirty revision.

CPU, CUDA, other orders, other sizes, and multi-node runs are outside
this matrix.

## Run it

Set the two binaries, the build tree, and a scratch root you can write.
`ACCOUNT` must be one the submit script allows. `DRY_RUN=1` prints the
build command and does not submit.

```bash
export BUILD_DIR=/path/to/hip-build
export HEAT3D_SPECTRAL_HIP_BIN="${BUILD_DIR}/apps/heat3d/heat3d_spectral_hip"
export HEAT3D_HIP_BIN="${BUILD_DIR}/apps/heat3d/heat3d_fd_hip"
export OPENPFC_SCALING_ROOT=/path/to/scratch/heat3d-n3-cost
export OPENPFC_LOG_DIR=/path/to/scratch/logs

./docs/lumi_slurm/submit_heat3d_n3_cost.sh check
./docs/lumi_slurm/submit_heat3d_n3_cost.sh build
./docs/lumi_slurm/submit_heat3d_n3_cost.sh submit
python3 apps/heat3d/scripts/n3_cost_matrix.py \
  --harvest "${OPENPFC_SCALING_ROOT}" \
  --out /path/to/out
```

`submit` reads `HEAT3D_STEPS`, `HEAT3D_WARMUP`, and `HEAT3D_DT` when they
are set. Leave them unset to keep the matrix protocol that `check`
prints. A repeat is admitted only when its `run_meta.txt` matches that
protocol, the profile has the expected number of accepted frames, the
tree was clean, MPI was GPU-aware, and a spectral run did not enable
pencils. Anything else is kept, with `admitted=no` and a
`reject_reason`, and stays out of the summary.

The shell `collect` mode is the same harvest with its output directory
set beside this page. Do not commit either CSV. They are run output.

## Output schema

`heat3d_n3_cost_repeats.csv`, one row per discovered run:

`tag`, `repeat`, `N`, `method`, `fd_order`, `job`, `wall_step_ms`,
`n_accepted`, `gpu_aware`, `use_pencils`, `revision`, `dirty`,
`bin_sha256`, `account`, `partition`, `nodes`, `ntasks`, `run_dir`,
`admitted`, `reject_reason`.

`heat3d_n3_cost_summary.csv`, one row per admitted `(N, method,
fd_order)`:

`N`, `method`, `fd_order`, `n_repeats`, `wall_step_ms_median`,
`wall_step_ms_min`, `wall_step_ms_max`, `cv_percent`, `jobs`.

`wall_step_ms` is the median frame time in milliseconds. `cv_percent`
is the coefficient of variation of the admitted repeats.

## What this recipe does not establish

Two sizes on one node do not decide an $N^3$ or $N^3 \log N$ law,
and they do not rank the operators for another machine, precision,
layout, or binary. Publish a result from a study that keeps the
profiles, the revision, and the job provenance. This page is only how
to produce the tables again.
