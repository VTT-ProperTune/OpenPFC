#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Device Green-operator cost vs the host solve on one MI250X GCD (#157).
#
# Required: ALLOY_COST_BIN, ALLOY_ELASTIC_PARITY_BIN

set -uo pipefail

: "${ALLOY_COST_BIN:?set ALLOY_COST_BIN to alloy_dendrite_coupled_cost}"
: "${ALLOY_ELASTIC_PARITY_BIN:?set ALLOY_ELASTIC_PARITY_BIN to openpfc_microelasticity_hip_parity}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/device_green_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"
CSV="${OUT}/device_green_cost.csv"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

echo "=== device Green cost, job ${SLURM_JOB_ID} ==="

srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
  "${ALLOY_ELASTIC_PARITY_BIN}" --n=24

# Two ranks: parity numbers are the acceptance check; HeFFTe may still
# complain on MPI_Finalize. Do not abort the cost table for that.
srun --ntasks=2 --gpus-per-node=2 --cpu-bind=none \
  "${ALLOY_ELASTIC_PARITY_BIN}" --n=24 || true

set -e

for n in 64 96 128; do
  srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
    "${ALLOY_COST_BIN}" --csv="${CSV}" --run-id="host_${n}" \
    --nx=$n --ny=$n --nz=$n --steps=20 --warmup=4 --elastic=1 --device=0
  srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
    "${ALLOY_COST_BIN}" --csv="${CSV}" --run-id="device_${n}" \
    --nx=$n --ny=$n --nz=$n --steps=20 --warmup=4 --elastic=1 --device=1
done

echo "=== CSV ==="
cat "${CSV}"
echo "=== done ==="
