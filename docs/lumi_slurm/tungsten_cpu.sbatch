#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001245
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001245/juaho/tungsten_perf_jobs/logs/%x-%j.out
#SBATCH --chdir=/scratch/project_462001245/juaho/tungsten_perf_jobs

# Tungsten performance (CPU / FFTW) on LUMI-C.
# Profiling with format "both" needs OpenPFC built with -DOpenPFC_ENABLE_HDF5=ON (timing_profile.h5).
# Load cray-hdf5 (serial), not cray-hdf5-parallel: rank-0 HDF5 export matches serial lib; parallel HDF5 can error at finalize.
# Submit with explicit node count, e.g.:  sbatch --nodes=2 --job-name=tperf-cpu-2n tungsten_cpu.sbatch
#
# Binding: full-node allocation + srun --cpu-bind=cores (see
# https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/ )

set -euo pipefail

JOBROOT=/scratch/project_462001245/juaho/tungsten_perf_jobs
REPO_INPUT=/users/juaho/dev/openpfc/docs/lumi_slurm/tungsten_performance_cpu.toml
OPENPFC_BIN_DEFAULT=/projappl/project_462001245/juaho/openpfc/bin/tungsten
TUNGSTEN_BIN="${TUNGSTEN_CPU_BIN:-$OPENPFC_BIN_DEFAULT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw cray-hdf5 lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUNDIR="${JOBROOT}/runs/cpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}/data"
cp -f "${REPO_INPUT}" "${RUNDIR}/input.toml"
cd "${RUNDIR}"

echo "=== Tungsten CPU === nodes=${SLURM_NNODES} ntasks=${SLURM_NTASKS} bin=${TUNGSTEN_BIN}"
echo "=== cwd=${RUNDIR}"

srun --cpu-bind=cores "${TUNGSTEN_BIN}" input.toml

echo "=== done"
