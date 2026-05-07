#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Configure and build OpenPFC with CUDA on an NVIDIA H100 node (Slurm).
#
# Submit from the repository root:
#   export OPENPFC_REPO=$PWD
#   sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_cuda_h100.sbatch
#
# Build directory defaults to ${OPENPFC_REPO}/builds/kobayashi-cuda-h100.
# Do not rely on OPENPFC_BUILD_DIR from another workflow (e.g. AMDGPU HIP) — override only with:
#   export KOBAYASHI_CUDA_OPENPFC_BUILD_DIR=/path/to/cuda/build
#
# Override modules: GCC_MODULE, OPENMPI_MODULE, CUDA_MODULE; architecture: CMAKE_CUDA_ARCHITECTURES (default 90 = H100).
#
# GPU-aware MPI: by default CMake uses **-DOpenPFC_MPI_CUDA_AWARE=ON** (Open MPI
# `MPIX_Query_cuda_support()` probe on this GPU node). Halos then use device
# pointers when the runtime query succeeds. Set **`KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0`**
# to force **OFF** (packed host faces only), e.g. if your Open MPI lacks CUDA UCX.

#SBATCH --job-name=kob-rebuild-cuda
#SBATCH --partition=nvidia_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=kobayashi_rebuild_openpfc_cuda_h100_%j.out
#SBATCH --error=kobayashi_rebuild_openpfc_cuda_h100_%j.err

set -euo pipefail

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

# Ignore generic OPENPFC_BUILD_DIR when set to e.g. a HIP tree — use explicit override or CUDA default.
if [[ -n "${KOBAYASHI_CUDA_OPENPFC_BUILD_DIR:-}" ]]; then
  OPENPFC_BUILD_DIR="${KOBAYASHI_CUDA_OPENPFC_BUILD_DIR}"
else
  OPENPFC_BUILD_DIR="${OPENPFC_REPO}/builds/kobayashi-cuda-h100"
fi
GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
# Login builds against cuda/12.x often fail at runtime on GPU nodes that only ship CUDA 13+ libraries.
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0}"

module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
if module load "${CUDA_MODULE}" 2>/dev/null; then
  echo "Loaded CUDA module: ${CUDA_MODULE}"
else
  echo "ERROR: CUDA module ${CUDA_MODULE} failed — need nvcc for kobayashi_fd_cuda." >&2
  exit 1
fi
hash -r

MPI_CC="$(command -v mpicc)"
MPI_CXX="$(command -v mpicxx)"
if [[ -z "${MPI_CC}" || -z "${MPI_CXX}" ]]; then
  echo "ERROR: mpicc/mpicxx not on PATH after module load." >&2
  exit 1
fi
# Use **mpicxx** as the host C++ compiler so CMake's `OpenPFCGpuAwareMpi.cmake` probe
# compiles with Open MPI's `mpi.h` (`OPEN_MPI` defined). Plain **g++** made the probe
# report "non-Open MPI" and skip `MPIX_Query_cuda_support()` at configure time.
CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-${MPI_CXX}}"
echo "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"

echo "=== kobayashi CUDA rebuild (H100) ==="
echo "host=$(hostname) JOB_ID=${SLURM_JOB_ID:-}"
echo "OPENPFC_REPO=${OPENPFC_REPO}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"

if [[ "${KOBAYASHI_REBUILD_KEEP_BUILD_DIR:-0}" != "1" ]]; then
  echo "Removing build dir: ${OPENPFC_BUILD_DIR}"
  rm -rf "${OPENPFC_BUILD_DIR}"
fi
mkdir -p "${OPENPFC_BUILD_DIR}"

CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}"

OPENPFC_MPI_CUDA_AWARE="${OPENPFC_MPI_CUDA_AWARE:-ON}"
if [[ "${KOBAYASHI_REBUILD_CUDA_MPI_AWARE:-1}" == "0" ]]; then
  OPENPFC_MPI_CUDA_AWARE=OFF
fi
echo "OpenPFC_MPI_CUDA_AWARE=${OPENPFC_MPI_CUDA_AWARE} (override: KOBAYASHI_REBUILD_CUDA_MPI_AWARE=0 forces OFF)"

CMAKE_EXTRA=( )
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  CMAKE_EXTRA+=( -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" )
fi

# Finite-difference CUDA: CPU HeFFTe is enough (no CUDA spectral required).
cmake -S "${OPENPFC_REPO}" -B "${OPENPFC_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${MPI_CC}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DMPI_CXX_COMPILER="${MPI_CXX}" \
  -DMPI_C_COMPILER="${MPI_CC}" \
  -DOpenPFC_ENABLE_CUDA=ON \
  -DOpenPFC_ENABLE_HIP=OFF \
  -DOpenPFC_MPI_CUDA_AWARE="${OPENPFC_MPI_CUDA_AWARE}" \
  -DOpenPFC_BUILD_TESTS=OFF \
  -DOpenPFC_BUILD_EXAMPLES=OFF \
  -DOpenPFC_BUILD_DOCUMENTATION=OFF \
  -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
  "${CMAKE_EXTRA[@]}"

cmake --build "${OPENPFC_BUILD_DIR}" -j"${SLURM_CPUS_PER_TASK:-16}" --target kobayashi_fd_cuda

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_cuda"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing ${BIN}" >&2
  exit 1
fi

echo ""
echo "OK: ${BIN}"
echo "GPU scaling (1 vs 2 MPI ranks / GPUs on one node):"
echo "  export OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "  sbatch apps/kobayashi/slurm/kobayashi_cuda_scaling_h100.sbatch"
