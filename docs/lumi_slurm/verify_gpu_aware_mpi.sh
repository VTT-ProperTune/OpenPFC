#!/bin/bash
# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run the OpenPFC verify_gpu_aware_mpi helper on LUMI-G (2 MPI ranks, 2 GPUs).
# Usage:
#   sbatch verify_gpu_aware_mpi.sh
# Or from an interactive GPU allocation:
#   bash verify_gpu_aware_mpi.sh
#
# Override binary:
#   export VERIFY_GPU_MPI_BIN=/projappl/.../bin/verify_gpu_aware_mpi

# sbatch reads the following directives (lines are shell comments on the compute node):
#SBATCH --account=project_462001245
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --job-name=verify-gpu-mpi

set -euo pipefail

VERIFY_BIN="${VERIFY_GPU_MPI_BIN:-/projappl/project_462001245/openpfc/0.1.4-hip/bin/verify_gpu_aware_mpi}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "=== verify_gpu_aware_mpi (job ${SLURM_JOB_ID}) ==="
  srun --ntasks=2 --gpus-per-node=2 --cpu-bind=cores "${VERIFY_BIN}"
else
  echo "=== verify_gpu_aware_mpi (interactive; needs GPU allocation) ==="
  srun --partition=small-g --account="${SLURM_ACCOUNT:-project_462001245}" \
    --time=00:05:00 --nodes=1 --ntasks=2 --gpus-per-node=2 \
    --cpu-bind=cores "${VERIFY_BIN}"
fi
