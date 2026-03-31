#!/bin/bash
# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# GPU-aware MPI check across **four GPU nodes**: one MPI rank per node so rank 0
# and rank 1 exchange device buffers over the network (not just intra-node).
#
# Submit:
#   sbatch docs/lumi_slurm/verify_gpu_aware_mpi_4nodes.sbatch
#
# Override binary (e.g. fresh build):
#   export VERIFY_GPU_MPI_BIN=/path/to/build/lumi-release/apps/tungsten/verify_gpu_aware_mpi
#   sbatch docs/lumi_slurm/verify_gpu_aware_mpi_4nodes.sbatch

#SBATCH --account=project_462001245
#SBATCH --partition=small-g
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --job-name=verify-gpu-mpi-4n

set -euo pipefail

VERIFY_BIN="${VERIFY_GPU_MPI_BIN:-/projappl/project_462001245/juaho/openpfc/bin/verify_gpu_aware_mpi}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

echo "=== verify_gpu_aware_mpi 4 nodes ==="
echo "=== host=$(hostname) VERIFY_GPU_MPI_BIN=${VERIFY_BIN}"
echo "=== SLURM_NNODES=${SLURM_NNODES} SLURM_NTASKS=${SLURM_NTASKS}"

srun --cpu-bind=cores "${VERIFY_BIN}"

echo "=== done (exit $? from srun is above)"
