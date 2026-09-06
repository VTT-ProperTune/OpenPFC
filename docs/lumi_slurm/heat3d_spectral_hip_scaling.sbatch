#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# heat3d_spectral_hip on LUMI-G: one MPI rank per GCD, I/O off, schema-v4 profile.
# Submit with docs/lumi_slurm/submit_heat3d_spectral_hip_scaling.sh.
#
# Required: HEAT3D_SPECTRAL_HIP_BIN
# Optional: HEAT3D_N (default 256), HEAT3D_STEPS (20), HEAT3D_DT (0.01),
#           OPENPFC_SCALING_ROOT

set -euo pipefail

: "${HEAT3D_SPECTRAL_HIP_BIN:?set HEAT3D_SPECTRAL_HIP_BIN to the 0.2 heat3d_spectral_hip binary}"

N="${HEAT3D_N:-256}"
STEPS="${HEAT3D_STEPS:-20}"
DT="${HEAT3D_DT:-0.01}"
JOBROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true

NTASKS="${SLURM_NTASKS:?SLURM_NTASKS is not set}"
RUNDIR="${JOBROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}"
cd "${RUNDIR}"

export HEAT3D_PROFILE_JSON="${RUNDIR}/timing_profile.json"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-1}"

PER_NODE="${SLURM_NTASKS_PER_NODE:-${NTASKS}}"
SRUN_BIND=()
CPU_BIND="none"
if (( PER_NODE == 8 )); then
  CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
  SRUN_BIND=(--cpu-bind="${CPU_BIND}")
elif (( PER_NODE > 8 )); then
  echo "at most 8 ranks per node (one per GCD); ntasks-per-node=${PER_NODE}" >&2
  exit 1
fi

{
  echo "=== heat3d_spectral_hip scaling ==="
  echo "job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME}"
  echo "nodes=${SLURM_NNODES} ntasks=${NTASKS} partition=${SLURM_JOB_PARTITION}"
  echo "bin=${HEAT3D_SPECTRAL_HIP_BIN}"
  echo "N=${N} steps=${STEPS} dt=${DT} cwd=${RUNDIR}"
  echo "cpu-bind=${CPU_BIND}"
  echo "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
  echo "MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY:-unset}"
} | tee "${RUNDIR}/run_meta.txt"

srun "${SRUN_BIND[@]}" \
  "${HEAT3D_SPECTRAL_HIP_BIN}" "${N}" "${STEPS}" "${DT}"

echo "=== done ==="
