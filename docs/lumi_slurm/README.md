<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Legacy Slurm launchers on LUMI

These launchers and campaign recipes are retained for historical reference.
They do not define the current build or submission defaults and are not extended.
For a new run, start with the [LUMI installation guide](../hpc/INSTALL.LUMI.md)
and [first Slurm job tutorial](../tutorials/hpc_slurm_day_one.md). The current
project account is `project_462001519`; older paths and accounts below describe
the original campaigns, not an instruction to submit new work to them.

Scripts and TOML inputs for running tungsten on LUMI-C (CPU / FFTW) and LUMI-G
(HIP / rocFFT).

## Historical 0.2 GPU scaling campaign

Issue `#87` first slice: size `tungsten_hip` on one GCD, then strong-scale
1/2/4/8 GCDs with I/O off. Recipe, account, and scratch paths:
[LUMI GPU scaling campaign](../hpc/lumi_gpu_scaling.md).

```bash
export TUNGSTEN_HIP_BIN=/path/to/tungsten_hip   # 0.2 HIP build
./docs/lumi_slurm/submit_tungsten_hip_scaling.sh size
TUNGSTEN_LX=512 ./docs/lumi_slurm/submit_tungsten_hip_scaling.sh strong
PARTITION=standard-g TUNGSTEN_LX=768 ./docs/lumi_slurm/submit_tungsten_hip_scaling.sh multinode
```

3D FD HIP twin (`heat3d_fd_hip`, device halo + stencil):

```bash
export HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip
./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh size
HEAT3D_N=256 ./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh strong
PARTITION=standard-g HEAT3D_N=256 ./docs/lumi_slurm/submit_heat3d_fd_hip_scaling.sh multinode
```

Issue `#25` FD-2 **weak** scaling (constant 256³ interior / GCD):
[LUMI-G Heat3D FD weak](../hpc/lumi_gpu_heat3d_fd_weak.md).

```bash
export HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh check
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh clean
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh diag
export HEAT3D_HALO_BIN=/path/to/23_halo_microtiming
./docs/lumi_slurm/submit_heat3d_fd_hip_weak.sh halo
```

Issue `#108` FD-order × halo-overlap weak scaling (account
`project_462001245`, separate scratch tree):
[FD-order scaling](../hpc/fd_order_scaling.md).

```bash
export HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip
./docs/lumi_slurm/submit_fd_order_scaling.sh check
./docs/lumi_slurm/submit_fd_order_scaling.sh clean
./docs/lumi_slurm/submit_fd_order_scaling.sh diag
```

OpenPFC `#124` 1-node GPU cost matrix
($N=512,1024$, spectral + FD-2/8/12, three repeats):
[Heat3D N^3 cost](../hpc/heat3d_n3_cost.md).

```bash
./docs/lumi_slurm/submit_heat3d_n3_cost.sh check
./docs/lumi_slurm/submit_heat3d_n3_cost.sh build
./docs/lumi_slurm/submit_heat3d_n3_cost.sh submit
./docs/lumi_slurm/submit_heat3d_n3_cost.sh collect
```

3D spectral HIP twin (`heat3d_spectral_hip`, implicit Euler, 2 FFTs/step):

```bash
export HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip
./docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh size
HEAT3D_N=768 ./docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh strong
PARTITION=standard-g HEAT3D_N=768 ./docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh multinode
```

Issue `#61` HeFFTe reshape-algorithm **tournament** (account
`project_462001245`, separate scratch tree):
[HeFFTe protocol tournament](../hpc/heffte_protocol_tournament.md).

```bash
export HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh check
./docs/lumi_slurm/submit_heffte_protocol_tournament.sh 768
```

Issue `#106` topology-crossover follow-up (randomized protocol order,
dense node counts, account `project_462001245`, separate scratch tree):
[HeFFTe topology crossover](../hpc/heffte_topology_crossover.md).

```bash
export HEAT3D_SPECTRAL_HIP_BIN=/path/to/heat3d_spectral_hip
./docs/lumi_slurm/submit_heffte_topology_crossover.sh check
./docs/lumi_slurm/submit_heffte_topology_crossover.sh wave1
```

Issue `#119` HeFFTe-trace MPI bandwidth vs GPU OSU `alltoall` (account
`project_462001519`, diagnostic `heffte-rocm-trace`, `standard-g`):
[HeFFTe bandwidth calibrate](../hpc/heffte_bandwidth_calibrate.md).

```bash
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh check
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh build
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh osu-build
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh 768
./docs/lumi_slurm/submit_heffte_bandwidth_calibrate.sh pencils
```

Files: `tungsten_hip_scaling.sbatch`, `tungsten_hip_scaling.toml`,
`submit_tungsten_hip_scaling.sh`, `heat3d_fd_hip_scaling.sbatch`,
`submit_heat3d_fd_hip_scaling.sh`, `heat3d_spectral_hip_scaling.sbatch`,
`submit_heat3d_spectral_hip_scaling.sh`,
`heffte_bandwidth_calibrate.sbatch`,
`submit_heffte_bandwidth_calibrate.sh`. Account `project_462001519`. Do
not point these jobs at the 0.1.4 binaries or `project_462001245`
scratch used below.

Weibel showcase (one node, 8 GCDs, reduced frames on scratch):
`apps/vlasov_maxwell/slurm/showcase_weibel.sbatch`.

Dendrite 2-D showcase (one node, 8 GCDs, elastic off/on, scratch frames):
`apps/alloy_dendrite_elastic/slurm/showcase_dendrite2d.sbatch`.
3-D spherical-seed path: `apps/alloy_dendrite_elastic/slurm/showcase_dendrite3d.sbatch`.

## Layout (legacy 1024³ helpers)

- This directory (under the git repo): `*.sbatch`, `submit_tungsten_performance.sh`, `verify_gpu_aware_mpi.sh`, and `tungsten_performance_*.toml`.
- Domain size: `tungsten_performance_*.toml` use 1024³ grid points (heavy run; sbatch time limit 12 h).
- Scratch (runtime logs and per-job working dirs): `/scratch/project_462001245/$USER/tungsten_perf_jobs/{logs,runs}/`.
- Binaries (default): `/projappl/project_462001245/openpfc/0.1.4-hip/bin/tungsten` and `tungsten_hip`. Override with `TUNGSTEN_CPU_BIN` / `TUNGSTEN_HIP_BIN` inside the job environment if needed.

## Submit all six jobs (1, 2, 4 nodes × CPU + GPU)

```bash
./docs/lumi_slurm/submit_tungsten_performance.sh
```

Or single job (CLI `--nodes` overrides the `#SBATCH --nodes` default):

```bash
sbatch --nodes=2 --job-name=tperf-cpu-2n docs/lumi_slurm/tungsten_cpu.sbatch
sbatch --nodes=2 --job-name=tperf-gpu-2n docs/lumi_slurm/tungsten_gpu.sbatch
```

## LUMI-specific behaviour

CPU (`tungsten_cpu.sbatch`)

- `partition/C` + `small`: 128 MPI ranks per node (one rank per physical core), full node (`--exclusive`).
- `srun --cpu-bind=cores` as recommended in [Distribution and binding](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/).
- TOML uses `use_gpu_aware = false` for the FFTW path.

GPU (`tungsten_gpu.sbatch`)

- `partition/G` + `small-g`: 8 MPI ranks per node (one per GCD), `--gpus-per-node=8`, `--exclusive`.
- `MPICH_GPU_SUPPORT_ENABLED=1` for GPU-aware MPI (required for device pointers; see [INSTALL.LUMI.md](../hpc/INSTALL.LUMI.md)).
- Optional smoke check: `sbatch docs/lumi_slurm/verify_gpu_aware_mpi.sh` (or run `verify_gpu_aware_mpi` from the OpenPFC `bin/` after setting `VERIFY_GPU_MPI_BIN` if needed).
- `tungsten_hip` / `heat3d_fd_hip` / `heat3d_spectral_hip` call `bind_local_device()` (`local_rank % n_devices`). Scaling sbatch leaves every GCD visible so GPU-aware MPI can use intra-node HIP IPC; it does not wrap ranks with `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`.
- Full-node 8-GCD jobs bind 7 cores per GCD with the LUMI CCD `mask_cpu` map (not a single `map_cpu` thread). That cut 16-GCD 768³ `wall_step` from 138 ms to 136 ms.
- Optional `OPENPFC_FFT_SLAB_AXIS=x|y|z` forces the 1D HeFFTe slab split.

## Outputs

Each job creates a directory under `runs/` named with the Slurm job name and ID. Profiling files use the `[profiling] output = "timing_profile"` stem in that directory (see `docs/performance_profiling.md` in the repo).

## Account and partition

The scripts use `#SBATCH --account=project_462001519`. Change if your billing project differs. Use `standard` / `standard-g` instead of `small` / `small-g` if you need the default full-node binding policy without `--exclusive`.
