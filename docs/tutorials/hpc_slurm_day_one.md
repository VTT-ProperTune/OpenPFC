<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: batch jobs (Slurm) — day one

This page is a **minimal** pattern for running OpenPFC under **Slurm** on a typical Linux cluster. Site-specific partitions, modules, and filesystems differ — adapt names to your center.

## 1. What you need on the node

- OpenPFC **built for the same CPU/GPU and MPI** you load in the job (see [`INSTALL.md`](../../INSTALL.md)).  
- A **wrapper binary** path to `tungsten` (or another `apps/` target) and a **JSON/TOML** input reachable from the job’s working directory.

## 2. Working directory and paths

Slurm starts the job in **`$SLURM_SUBMIT_DIR`** (where you ran `sbatch`) or a directory you set with `#SBATCH -D` / `cd` in the script.

**Rule:** Paths in JSON (`fields[].data`, logs) are usually **relative to the job’s current working directory**. Use **absolute paths** if you prefer not to depend on `cd`.

See also: [`../mpi_io_layout_checklist.md`](../hpc/mpi_io_layout_checklist.md).

## 3. Minimal CPU job (`mpirun` or `srun`)

Many sites provide Open MPI; you may use `mpirun` or `srun` as your scheduler documents.

```bash
#!/bin/bash
#SBATCH -J openpfc-tungsten
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 01:00:00
#SBATCH -p <your-partition>

module purge
module load gcc/11.2.0
module load openmpi/4.1.1

cd "$SLURM_SUBMIT_DIR"   # or cd /scratch/$USER/run123

mpirun -n "${SLURM_NTASKS}" ./tungsten ./inputs/tungsten_single_seed.json
```

Replace `module load` lines with your site’s modules. Confirm `which mpirun` matches the MPI used to **build** OpenPFC ([`../troubleshooting.md`](../troubleshooting.md)).

## 4. Using `srun` instead of `mpirun`

On Slurm clusters, **`srun`** is often the supported launcher:

```bash
srun --ntasks="${SLURM_NTASKS}" --cpus-per-task=1 ./tungsten ./inputs/tungsten_single_seed.json
```

Binding (`--cpu-bind`, GPUs) is **site-specific** — follow your center’s OpenPFC or MPI guide.

## 5. LUMI and GPU-aware MPI

For LUMI-G, ROCm, and GPU-aware settings, use the dedicated docs:

- [`../INSTALL.LUMI.md`](../hpc/INSTALL.LUMI.md)  
- [`../lumi_slurm/README.md`](../lumi_slurm/README.md)  
- GPU apps: [`gpu_app_quickstart.md`](gpu_app_quickstart.md)

## See also

- [`../learning_paths.md`](../learning_paths.md) — **Run simulations** track  
- [`../quickstart.md`](../quickstart.md) — local `mpirun` first  
- [`../INSTALL.md`](../../INSTALL.md) — build and modules  
