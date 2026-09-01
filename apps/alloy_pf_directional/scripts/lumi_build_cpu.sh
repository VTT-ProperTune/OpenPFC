#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CPU (no HIP) OpenPFC + HeFFTe build for alloy_pf_directional_openmp on LUMI-C.
#
# Login node (needs the network for HeFFTe tarball + CMake FetchContent):
#   ./apps/alloy_pf_directional/scripts/lumi_build_cpu.sh configure
#
# Compile (8 cores on partition=small, ~1 h cap — not a physics run):
#   sbatch apps/alloy_pf_directional/scripts/lumi_build_cpu.sh
#   ./apps/alloy_pf_directional/scripts/lumi_build_cpu.sh compile   # if already in a job
#
# Do not use scripts/build.sh here: that defaults to HIP / LUMI-G.
#SBATCH --job-name=alcu-build
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1750
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/alcu_fta/logs/build-%j.out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  # shellcheck source=lumi_paths.sh
  source "${SCRIPT_DIR}/lumi_paths.sh"
  SRC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  # shellcheck source=lumi_paths.sh
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_directional/scripts/lumi_paths.sh
  SRC_ROOT="${LUMI_SRC}"
fi

load_cpu_modules() {
  module purge
  module load "${LUMI_STACK}" partition/C cpeGNU cray-fftw lumi-CrayPath
  export CC=cc CXX=CC
  lumi_cpu_runtime_env
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

cmd_configure_heffte() {
  load_cpu_modules
  mkdir -p "${LUMI_PROJAPPL}/src" "${LUMI_PROJAPPL}/build" "${LUMI_LOGS}" \
    "${LUMI_HEFFTE_PREFIX}"
  if heffte_config_dir >/dev/null; then
    echo "HeFFTe already installed: $(heffte_config_dir)"
    return 0
  fi
  local tar="${LUMI_PROJAPPL}/src/heffte-v${HEFFTE_VERSION}.tar.gz"
  if [[ ! -d "${LUMI_HEFFTE_SRC}" ]]; then
    if [[ ! -f "${tar}" ]]; then
      wget -q -O "${tar}" \
        "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${HEFFTE_VERSION}.tar.gz"
    fi
    tar xf "${tar}" -C "${LUMI_PROJAPPL}/src"
  fi
  cmake -S "${LUMI_HEFFTE_SRC}" -B "${LUMI_HEFFTE_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_INSTALL_PREFIX="${LUMI_HEFFTE_PREFIX}" \
    -DHeffte_ENABLE_FFTW=ON \
    -DHeffte_ENABLE_CUDA=OFF \
    -DHeffte_ENABLE_ROCM=OFF \
    -DHeffte_ENABLE_TESTING=OFF
}

cmd_configure_openpfc() {
  load_cpu_modules
  mkdir -p "${LUMI_PREFIX}" "${LUMI_LOGS}"
  local heffte_dir
  heffte_dir="$(heffte_config_dir)"
  cmake -S "${SRC_ROOT}" -B "${LUMI_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_INSTALL_PREFIX="${LUMI_PREFIX}" \
    -DHeffte_DIR="${heffte_dir}" \
    -DOpenPFC_ENABLE_CUDA=OFF \
    -DOpenPFC_ENABLE_HIP=OFF \
    -DOpenPFC_ENABLE_HDF5=OFF \
    -DOpenPFC_FETCH_HEFFTE=OFF
}

cmd_configure() {
  cmd_configure_heffte
  if heffte_config_dir >/dev/null; then
    cmd_configure_openpfc
  else
    echo "HeFFTe is configured but not installed yet. On the login node after a compile job:"
    echo "  $0 configure-openpfc"
  fi
}

cmd_compile() {
  load_cpu_modules
  local jobs="${SLURM_CPUS_PER_TASK:-${LUMI_BUILD_CPUS}}"
  local did=0
  if [[ -d "${LUMI_HEFFTE_BUILD}" ]] && ! heffte_config_dir >/dev/null; then
    cmake --build "${LUMI_HEFFTE_BUILD}" -j"${jobs}"
    cmake --install "${LUMI_HEFFTE_BUILD}"
    did=1
  fi
  if [[ -f "${LUMI_BUILD}/CMakeCache.txt" ]]; then
    cmake --build "${LUMI_BUILD}" -j"${jobs}" --target alloy_pf_directional_openmp alloy_pf_directional_mpi
    cmake --install "${LUMI_BUILD}/apps/alloy_pf_directional"
    echo "installed ${LUMI_BIN} ${LUMI_BIN_MPI}"
    ls -l "${LUMI_BIN}" "${LUMI_BIN_MPI}"
    did=1
  elif (( did == 1 )); then
    echo "HeFFTe installed. On the login node run: $0 configure-openpfc"
    echo "then sbatch this script again to compile alloy_pf_directional_openmp."
  else
    echo "nothing to compile; run: $0 configure-heffte" >&2
    exit 1
  fi
}

usage() {
  echo "usage: $0 configure|configure-heffte|configure-openpfc|compile|all" >&2
  exit 2
}

MODE="${1:-}"
if [[ -n "${SLURM_JOB_ID:-}" && -z "${MODE}" ]]; then
  MODE=compile
fi
case "${MODE}" in
  configure) cmd_configure ;;
  configure-heffte) cmd_configure_heffte ;;
  configure-openpfc) cmd_configure_openpfc ;;
  compile) cmd_compile ;;
  all)
    cmd_configure
    cmd_compile
    ;;
  *) usage ;;
esac
