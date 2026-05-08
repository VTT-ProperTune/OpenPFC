#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# **nvidia_h100 queue only** (`#SBATCH --partition=nvidia_h100`): do **not** run this
# wrapper on the login node for real timings — always **`sbatch`** this script.
#
# Profile **kobayashi_fd_cuda** with **2 MPI ranks / 2 GPUs** under **NVIDIA Nsight
# Systems** (`nsys`). Submit from the OpenPFC repo root:
#   export OPENPFC_REPO=$PWD
#   sbatch apps/kobayashi/slurm/kobayashi_fd_cuda_h100_np2_nsys.sbatch
#
# Output: `kobayashi_fd_cuda_h100_nsys_<jobid>/` with `*.nsys-rep`, app log, and optional
# `nsys stats` text summaries. Default problem is **shorter** than production so reports
# stay small; override with **`KOBAYASHI_NSYS_STEPS`**, **`KOBAYASHI_CMP_NX`**, etc.
#
# Large `.nsys-rep` files: if **`/WRKDIR/$USER/openpfc`** exists (fast NVMe), artifacts go
# there; else under **`SLURM_SUBMIT_DIR`**. Override with **`KOBAYASHI_NSYS_ARTIFACT_BASE`**.
# See **`pick_nsys_artifact_base.bash`**.

#SBATCH --job-name=kob-cuda-n2-nsys
#SBATCH --partition=nvidia_h100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=kobayashi_fd_cuda_h100_np2_nsys_%j.out
#SBATCH --error=kobayashi_fd_cuda_h100_np2_nsys_%j.err

set -euo pipefail

SLURM_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=pick_nsys_artifact_base.bash
source "${SLURM_SCRIPT_DIR}/pick_nsys_artifact_base.bash"

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

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not on PATH after loading CUDA module ${CUDA_MODULE}" >&2
  exit 1
fi

if [[ "${KOBAYASHI_OMPI_KEEP_OFI:-0}" != "1" ]]; then
  export OMPI_MCA_mtl=^ofi
fi
export SLURM_MPI_TYPE="${SLURM_MPI_TYPE:-pmix}"

NX="${KOBAYASHI_NSYS_NX:-${KOBAYASHI_CMP_NX:-${KOBAYASHI_SCALING_NX:-512}}}"
NY="${KOBAYASHI_NSYS_NY:-${KOBAYASHI_CMP_NY:-${KOBAYASHI_SCALING_NY:-512}}}"
NSTEPS="${KOBAYASHI_NSYS_STEPS:-200}"
DT="${KOBAYASHI_CMP_DT:-${KOBAYASHI_SCALING_DT:-1e-4}}"
DX="${KOBAYASHI_CMP_DX:-${KOBAYASHI_SCALING_DX:-0.03}}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_ROOT="${KOBAYASHI_NSYS_ARTIFACT_BASE}/kobayashi_fd_cuda_h100_nsys_${JOB_TAG}"
mkdir -p "${OUT_ROOT}"

export OPENPFC_KOBAYASHI_SKIP_PNG=1
export OPENPFC_KOBAYASHI_QUIET=1
export OPENPFC_KOBAYASHI_PERF=1

echo "host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "partition=nvidia_h100 (fixed in #SBATCH)"
echo "nsys=$(command -v nsys) $(nsys --version 2>/dev/null | head -1 || true)"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "KOBAYASHI_NSYS_ARTIFACT_BASE=${KOBAYASHI_NSYS_ARTIFACT_BASE} (${KOBAYASHI_NSYS_ARTIFACT_BASE_REASON})"
echo "problem: Nx=${NX} Ny=${NY} steps=${NSTEPS} dt=${DT} dx=${DX} (override NSTEPS with KOBAYASHI_NSYS_STEPS)"
echo "OUT_ROOT=${OUT_ROOT}"
nvidia-smi -L 2>/dev/null || true
echo ""

NSYS_BASE="${OUT_ROOT}/kob_nsys"
logf="${OUT_ROOT}/gpu_np2_nsys.log"

# `--mpi-impl=openmpi`: required with `--trace=mpi` so Nsight wraps Open MPI correctly.
# Shorter default NSTEPS keeps .nsys-rep uploadable; increase KOBAYASHI_NSYS_STEPS as needed.
nsys profile \
  --trace=cuda,nvtx,osrt,mpi \
  --mpi-impl=openmpi \
  --force-overwrite=true \
  -o "${NSYS_BASE}" \
  -- \
  mpirun -np 2 --bind-to core \
    "${BIN_GPU}" "${NX}" "${NY}" "${NSTEPS}" "${DT}" "${DX}" "${OUT_ROOT}/gpu_np2" \
  2>&1 | tee "${logf}"

echo ""
echo "=== nsys report files ==="
ls -la "${OUT_ROOT}"/*.nsys-rep 2>/dev/null || ls -la "${OUT_ROOT}/" || true

echo ""
echo "=== nsys stats (CUDA API + kernels, first .nsys-rep) ==="
shopt -s nullglob
_rep=( "${OUT_ROOT}"/kob_nsys*.nsys-rep )
if [[ ${#_rep[@]} -gt 0 ]]; then
  f0="${_rep[0]}"
  echo "Using: $f0"
  nsys stats -q --report cuda_gpu_kern_sum --report cuda_api_sum --report osrt_sum \
    "$f0" 2>/dev/null | head -120 || echo "(nsys stats failed — open the .nsys-rep in Nsight Systems GUI)"
else
  echo "No .nsys-rep found under ${OUT_ROOT}"
fi

echo ""
grep -E '^(KOBAYASHI_CUDA device|KOBAYASHI_MPI|KOBAYASHI_CUDA_HALO|KOBAYASHI_PERF_LOOP|OPENPFC_CUDA_PROFILE_HALO_SUMMARY|KOBAYASHI_VERIFY )' "${logf}" >&2 || true

echo ""
echo "DONE: download ${OUT_ROOT}/*.nsys-rep and open in **nsys-ui** (Nsight Systems) for timeline."
