#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# GPU-only: kobayashi_fd_cuda **np=2**, **1 CPU per rank**, **2 GPUs**.

#SBATCH --job-name=kob-cuda-n2
#SBATCH --partition=nvidia_h100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=kobayashi_fd_cuda_h100_np2_1cpu_%j.out
#SBATCH --error=kobayashi_fd_cuda_h100_np2_1cpu_%j.err

set -euo pipefail

OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${KOBAYASHI_CUDA_OPENPFC_BUILD_DIR:-}}"
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  _d="${SLURM_SUBMIT_DIR:-.}/builds/kobayashi-cuda-h100"
  if [[ -d "${_d}" ]]; then
    OPENPFC_BUILD_DIR="${_d}"
  fi
fi
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR (CUDA build with kobayashi_fd_cuda)." >&2
  exit 1
fi

BIN_GPU="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_cuda"
if [[ ! -x "${BIN_GPU}" ]]; then
  echo "ERROR: missing ${BIN_GPU}" >&2
  exit 1
fi

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
module load "${CUDA_MODULE}"
hash -r

if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"

NX="${KOBAYASHI_CMP_NX:-${KOBAYASHI_SCALING_NX:-512}}"
NY="${KOBAYASHI_CMP_NY:-${KOBAYASHI_SCALING_NY:-512}}"
NSTEPS="${KOBAYASHI_CMP_STEPS:-${KOBAYASHI_SCALING_STEPS:-5000}}"
DT="${KOBAYASHI_CMP_DT:-${KOBAYASHI_SCALING_DT:-1e-4}}"
DX="${KOBAYASHI_CMP_DX:-${KOBAYASHI_SCALING_DX:-0.03}}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${SLURM_SUBMIT_DIR:-.}/kobayashi_fd_cuda_h100_1cpu_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1
# Timestep breakdown: KOBAYASHI_PERF_LOOP + OPENPFC_CUDA_PROFILE_HALO_SUMMARY (rank 0).
export OPENPFC_KOBAYASHI_PERF=1

echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX}"
echo "OUT_ROOT=${OUT_ROOT}"
nvidia-smi -L 2>/dev/null || true
echo ""

logf="${OUT_ROOT}/gpu_np2.log"
mpirun -np 2 --bind-to core \
  "${BIN_GPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}/gpu_np2" 2>&1 | tee "${logf}"
# Match `KOBAYASHI_CUDA device…` only (not `KOBAYASHI_CUDA_HALO_MODE`, which would duplicate .out).
# Print summary to stderr so Slurm .out is not duplicated (logf already has full mpirun stream).
grep -E '^(KOBAYASHI_CUDA device|KOBAYASHI_MPI|KOBAYASHI_CUDA_HALO|KOBAYASHI_PERF_LOOP|OPENPFC_CUDA_PROFILE_HALO_SUMMARY|KOBAYASHI_VERIFY )' "${logf}" >&2 || true
