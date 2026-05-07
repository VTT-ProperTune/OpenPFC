#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Configure and build OpenPFC with HIP on an AMDGPU compute node (Slurm).
#
# Submit from the repository root:
#   export OPENPFC_REPO=$PWD
#   export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-hip-amdgpu   # optional
#   sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_amdgpu.sbatch
#
# Override modules with ROCM_MODULE / GCC_MODULE / OPENMPI_MODULE. Point CMAKE_PREFIX_PATH at
# ROCm and your HeFFTe install (CPU HeFFTe is enough for kobayashi_fd — spectral HIP paths need
# ROCm HeFFTe; see INSTALL.md §8).

#SBATCH --job-name=kob-rebuild-hip
#SBATCH --partition=amdgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=kobayashi_rebuild_openpfc_amdgpu_%j.out
#SBATCH --error=kobayashi_rebuild_openpfc_amdgpu_%j.err

set -euo pipefail

# Prefer explicit OPENPFC_REPO; else Slurm submit dir; else repo root inferred from this script
# (so `sh apps/kobayashi/slurm/…sbatch` from any cwd still finds CMakeLists.txt).
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_FROM_SCRIPT="$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -n "${OPENPFC_REPO:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/CMakeLists.txt" ]]; then
  OPENPFC_REPO="${SLURM_SUBMIT_DIR}"
elif [[ -f "${REPO_FROM_SCRIPT}/CMakeLists.txt" ]]; then
  OPENPFC_REPO="${REPO_FROM_SCRIPT}"
else
  OPENPFC_REPO="$(pwd)"
fi
if [[ ! -f "${OPENPFC_REPO}/CMakeLists.txt" ]]; then
  echo "ERROR: OPENPFC_REPO=${OPENPFC_REPO} is not the OpenPFC root (no CMakeLists.txt)." >&2
  exit 1
fi

OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${OPENPFC_REPO}/builds/kobayashi-hip-amdgpu}"
GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
ROCM_MODULE="${ROCM_MODULE:-rocm}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
if module load "${ROCM_MODULE}" 2>/dev/null; then
  echo "Loaded ROCm module: ${ROCM_MODULE}"
else
  echo "WARN: module ${ROCM_MODULE} failed — ensure ROCm is on PATH (ROCM_PATH / CMAKE_PREFIX_PATH)." >&2
fi
hash -r

MPI_CC="$(command -v mpicc)"
MPI_CXX="$(command -v mpicxx)"
HOST_CXX="$(command -v g++)"
if [[ -z "${MPI_CC}" || -z "${MPI_CXX}" ]]; then
  echo "ERROR: mpicc/mpicxx not on PATH after module load." >&2
  exit 1
fi
if [[ -z "${HOST_CXX}" ]]; then
  echo "ERROR: g++ not on PATH (needed as CMAKE_CXX_COMPILER for HIP + MPI)." >&2
  exit 1
fi

echo "=== kobayashi HIP rebuild (AMDGPU) ==="
echo "host=$(hostname) JOB_ID=${SLURM_JOB_ID:-}"
echo "OPENPFC_REPO=${OPENPFC_REPO}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"

if [[ "${KOBAYASHI_REBUILD_KEEP_BUILD_DIR:-0}" != "1" ]]; then
  echo "Removing build dir: ${OPENPFC_BUILD_DIR}"
  rm -rf "${OPENPFC_BUILD_DIR}"
fi
mkdir -p "${OPENPFC_BUILD_DIR}"

CMAKE_EXTRA=( )
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  CMAKE_EXTRA+=( -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" )
fi
if [[ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]]; then
  CMAKE_EXTRA+=( -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}" )
fi

# Prefer GPU-aware MPI off for host-staged halo drivers unless you verified device buffers.
cmake -S "${OPENPFC_REPO}" -B "${OPENPFC_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${MPI_CC}" \
  -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
  -DMPI_CXX_COMPILER="${MPI_CXX}" \
  -DMPI_C_COMPILER="${MPI_CC}" \
  -DOpenPFC_ENABLE_HIP=ON \
  -DOpenPFC_ENABLE_CUDA=OFF \
  -DOpenPFC_MPI_HIP_AWARE=OFF \
  -DOpenPFC_BUILD_TESTS=OFF \
  -DOpenPFC_BUILD_EXAMPLES=OFF \
  -DOpenPFC_BUILD_DOCUMENTATION=OFF \
  -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
  "${CMAKE_EXTRA[@]}"

cmake --build "${OPENPFC_BUILD_DIR}" -j"${SLURM_CPUS_PER_TASK:-16}" --target kobayashi_fd_hip

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_hip"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing ${BIN}" >&2
  exit 1
fi

echo ""
echo "OK: ${BIN}"
echo "GPU scaling (1 vs 2 MPI ranks / GPUs on one node):"
echo "  export OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "  sbatch apps/kobayashi/slurm/kobayashi_hip_scaling_amdgpu.sbatch"
