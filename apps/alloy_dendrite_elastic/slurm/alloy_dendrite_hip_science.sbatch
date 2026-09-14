#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Coupled GPU science: elastic off / on / solutal / thermal / both (#157).
#
# Required: ALLOY_HIP_GROWTH_BIN

set -euo pipefail

: "${ALLOY_HIP_GROWTH_BIN:?set ALLOY_HIP_GROWTH_BIN to alloy_dendrite_hip_growth}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/hip_science_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

echo "=== HIP coupled science, job ${SLURM_JOB_ID} ==="

run() {
  local id="$1"; shift
  srun --ntasks=1 --gpus-per-node=1 --cpu-bind=none \
    "${ALLOY_HIP_GROWTH_BIN}" --nx=192 --ny=192 --nz=1 --dx=0.8 \
    --steps=4000 --sample-every=200 --csv="${OUT}/${id}.csv" \
    --run-id="${id}" "$@"
}

run off --elastic=0 --lambda-el=0
# Modulus contrast, no eigenstrain: dynamics should match "off".
run on --elastic=1 --lambda-el=1 --eps-c=0 --eps-T=0
run solutal --elastic=1 --lambda-el=1 --eps-c=0.01 --eps-T=0
run thermal --elastic=1 --lambda-el=1 --eps-c=0 --eps-T=0.01 --evolve-theta=1 --Dth=2 --Mc=0.5
run both --elastic=1 --lambda-el=1 --eps-c=0.01 --eps-T=0.01 --evolve-theta=1 --Dth=2 --Mc=0.5

echo "=== done ${OUT} ==="
