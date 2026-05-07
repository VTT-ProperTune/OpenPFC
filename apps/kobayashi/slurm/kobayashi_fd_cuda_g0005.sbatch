#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Single-GPU Kobayashi FD CUDA run on node **g0005** (NVIDIA H100 on partition **nvidia_h100**).
# g0005 is a nodelist constraint, not a Slurm partition name.
#
# From the OpenPFC repository root:
#   export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-cuda-h100   # or your CUDA build tree
#   sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_g0005.sbatch
#
# Problem size: **`KOBAYASHI_G0005_*` overrides take precedence**, then **`KOBAYASHI_SCALING_*`**, then defaults
# (defaults: 256×256, 2000 steps — avoids accidental **`KOBAYASHI_SCALING_STEPS`** from a scaling shell).
#   KOBAYASHI_G0005_NX NY STEPS DT DX
#
# Optional: unset nodelist to use any nvidia_h100 node:
#   SBATCH_EXCLUDE_NODE_CONSTRAINT=1 sbatch ...  (edit script or export before patching — simpler: comment --nodelist)

#SBATCH --job-name=kob-cuda-g0005
#SBATCH --partition=nvidia_h100
#SBATCH --nodelist=g0005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=kobayashi_fd_cuda_g0005_%j.out
#SBATCH --error=kobayashi_fd_cuda_g0005_%j.err

set -euo pipefail

OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${KOBAYASHI_CUDA_OPENPFC_BUILD_DIR:-}}"
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  _d="${SLURM_SUBMIT_DIR:-.}/builds/kobayashi-cuda-h100"
  if [[ -x "${_d}/apps/kobayashi/kobayashi_fd_cuda" ]]; then
    OPENPFC_BUILD_DIR="${_d}"
  fi
fi
if [[ -z "${OPENPFC_BUILD_DIR}" ]]; then
  echo "ERROR: set OPENPFC_BUILD_DIR or KOBAYASHI_CUDA_OPENPFC_BUILD_DIR to your CUDA CMake build tree." >&2
  echo "Hint: sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch (builds on this node)." >&2
  exit 1
fi

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_cuda"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing executable: ${BIN}" >&2
  exit 1
fi

GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
if ! module load "${CUDA_MODULE}" 2>/dev/null; then
  echo "ERROR: CUDA module ${CUDA_MODULE} failed." >&2
  exit 1
fi
hash -r

if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"
SRUN_MPI_TYPE="${KOBAYASHI_SRUN_MPI:-pmix}"
LAUNCHER="${KOBAYASHI_MPI_LAUNCHER:-srun}"

NX="${KOBAYASHI_G0005_NX:-${KOBAYASHI_SCALING_NX:-256}}"
NY="${KOBAYASHI_G0005_NY:-${KOBAYASHI_SCALING_NY:-256}}"
NSTEPS="${KOBAYASHI_G0005_STEPS:-${KOBAYASHI_SCALING_STEPS:-2000}}"
DT="${KOBAYASHI_G0005_DT:-${KOBAYASHI_SCALING_DT:-1e-4}}"
DX="${KOBAYASHI_G0005_DX:-${KOBAYASHI_SCALING_DX:-0.03}}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${SLURM_SUBMIT_DIR:-.}/kobayashi_fd_cuda_g0005_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "BIN=${BIN}"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX}"
echo "OUT_ROOT=${OUT_ROOT}"

nvidia-smi -L 2>/dev/null || true
nvidia-smi 2>/dev/null | head -20 || true

LOG="${OUT_ROOT}/run.log"
if [[ "${LAUNCHER}" == "mpirun" ]]; then
  mpirun -np 1 --bind-to core \
    "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}" 2>&1 | tee "${LOG}"
else
  srun --mpi="${SRUN_MPI_TYPE}" -n 1 --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    "${BIN}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}" 2>&1 | tee "${LOG}"
fi

echo ""
echo "Log: ${LOG}"
grep -E '^KOBAYASHI_(CUDA|VERIFY)' "${LOG}" || true
