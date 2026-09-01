#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# HIP build for alloy_pf_directional_hip on LUMI-G (tpinomaa / project_462001519).
# Prefer CPU HeFFTe for the CMake find (this app is FD, not spectral).
#
# Login (network):  ./apps/alloy_pf_directional/scripts/lumi_build_hip.sh configure
# Compile on GPU:   sbatch apps/alloy_pf_directional/scripts/lumi_build_hip.sh
#
#SBATCH --job-name=alcu-hip-bld
#SBATCH --account=project_462001519
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/build-hip-%j.out
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  # shellcheck source=lumi_paths.sh
  source "${SCRIPT_DIR}/lumi_paths.sh"
  SRC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh
  SRC_ROOT="${LUMI_SRC}"
fi

load_gpu_modules() {
  module purge
  module load "${LUMI_STACK}" partition/G cpeGNU cray-fftw lumi-CrayPath
  export CC=cc CXX=CC
  lumi_gpu_runtime_env
  export MPICH_GPU_SUPPORT_ENABLED=1
  if [[ -d "${HOME}/privatemodules" ]]; then
    module use "${HOME}/privatemodules"
    module load heffte-rocm 2>/dev/null || true
  fi
}

heffte_config_dir() {
  local p
  for p in \
    "${LUMI_HEFFTE_PREFIX}/lib64/cmake/Heffte" \
    "${LUMI_HEFFTE_PREFIX}/lib/cmake/Heffte"
  do
    if [[ -f "${p}/HeffteConfig.cmake" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

cmd_configure() {
  load_gpu_modules
  mkdir -p "${LUMI_PREFIX_HIP}" "${LUMI_BUILD_HIP}" "${LUMI_LOGS}"
  local heffte_dir
  heffte_dir="$(heffte_config_dir)" || {
    echo "HeFFTe CMake package not found under ${LUMI_HEFFTE_PREFIX}" >&2
    echo "CPU HeFFTe is enough to link alloy_pf_directional_hip (no FFT). Build it via lumi_build_cpu.sh first." >&2
    exit 1
  }
  cmake -S "${SRC_ROOT}" -B "${LUMI_BUILD_HIP}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_INSTALL_PREFIX="${LUMI_PREFIX_HIP}" \
    -DHeffte_DIR="${heffte_dir}" \
    -DOpenPFC_ENABLE_CUDA=OFF \
    -DOpenPFC_ENABLE_HIP=ON \
    -DOpenPFC_MPI_HIP_AWARE=ON \
    -DOpenPFC_ENABLE_HDF5=OFF \
    -DOpenPFC_FETCH_HEFFTE=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DGPU_TARGETS=gfx90a
}

cmd_compile() {
  load_gpu_modules
  local jobs="${SLURM_CPUS_PER_TASK:-8}"
  if [[ ! -f "${LUMI_BUILD_HIP}/CMakeCache.txt" ]]; then
    echo "no HIP CMake cache; run: $0 configure" >&2
    exit 1
  fi
  cmake --build "${LUMI_BUILD_HIP}" -j"${jobs}" --target alloy_pf_directional_hip
  cmake --install "${LUMI_BUILD_HIP}/apps/alloy_pf_directional"
  echo "installed ${LUMI_BIN_HIP}"
  ls -l "${LUMI_BIN_HIP}"
}

MODE="${1:-}"
if [[ -n "${SLURM_JOB_ID:-}" && -z "${MODE}" ]]; then
  MODE=compile
fi
case "${MODE}" in
  configure) cmd_configure ;;
  compile) cmd_compile ;;
  all)
    cmd_configure
    cmd_compile
    ;;
  *)
    echo "usage: $0 configure|compile|all" >&2
    exit 2
    ;;
esac
