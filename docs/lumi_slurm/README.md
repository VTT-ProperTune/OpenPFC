<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Slurm: tungsten performance on LUMI

Scripts and TOML inputs for running the tungsten performance case (`tungsten_performance.toml`–style settings) on **LUMI-C** (CPU / FFTW) and **LUMI-G** (HIP / rocFFT).

## Layout

- **This directory** (under the git repo): `*.sbatch`, `submit_tungsten_performance.sh`, `verify_gpu_aware_mpi.sh`, and `tungsten_performance_*.toml`.
- **Domain size:** `tungsten_performance_*.toml` use **1024³** grid points (heavy run; sbatch time limit **12 h**).
- **Scratch** (runtime logs and per-job working dirs): `/scratch/project_462001245/$USER/tungsten_perf_jobs/{logs,runs}/`.
- **Binaries** (default): `/projappl/project_462001245/openpfc/0.1.4-hip/bin/tungsten` and `tungsten_hip`. Override with `TUNGSTEN_CPU_BIN` / `TUNGSTEN_HIP_BIN` inside the job environment if needed.

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

**CPU (`tungsten_cpu.sbatch`)**

- `partition/C` + `small`: 128 MPI ranks per node (one rank per physical core), full node (`--exclusive`).
- `srun --cpu-bind=cores` as recommended in [Distribution and binding](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/).
- TOML uses `use_gpu_aware = false` for the FFTW path.

**GPU (`tungsten_gpu.sbatch`)**

- `partition/G` + `small-g`: **8 MPI ranks per node** (one per **GCD**), `--gpus-per-node=8`, `--exclusive`.
- `MPICH_GPU_SUPPORT_ENABLED=1` for GPU-aware MPI (required for device pointers; see [INSTALL.LUMI.md](../INSTALL.LUMI.md)).
- Optional smoke check: `sbatch docs/lumi_slurm/verify_gpu_aware_mpi.sh` (or run `verify_gpu_aware_mpi` from the OpenPFC `bin/` after setting `VERIFY_GPU_MPI_BIN` if needed).
- Wrapper script sets `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID` so each rank sees a single GPU as device 0 ([LUMI-G MPI example](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/)).
- `srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41` matches the documented LUMI-G rank/NUMA/GPU mapping ([GPU binding](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding)).

## Outputs

Each job creates a directory under `runs/` named with the Slurm job name and ID. Profiling files use the `[profiling] output = "timing_profile"` stem in that directory (see `docs/performance_profiling.md` in the repo).

## Account and partition

The scripts use `#SBATCH --account=project_462001245`. Change if your billing project differs. Use `standard` / `standard-g` instead of `small` / `small-g` if you need the default full-node binding policy without `--exclusive`.
