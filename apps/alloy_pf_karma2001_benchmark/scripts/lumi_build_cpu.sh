#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Configure on the login node (needs the network), compile on LUMI-C:
#   ./apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh configure
#   sbatch apps/alloy_pf_karma2001_benchmark/scripts/lumi_build_cpu.sh
#
# Do not use scripts/build.sh here (that defaults to HIP / LUMI-G).
#SBATCH --job-name=karma-build
#SBATCH --account=project_462001519
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1750
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001519/tpinomaa/karma2001/logs/build-%j.out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/lumi_paths.sh" ]]; then
  # shellcheck source=lumi_paths.sh
  source "${SCRIPT_DIR}/lumi_paths.sh"
  SRC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  # shellcheck source=lumi_paths.sh
  source /projappl/project_462001519/tpinomaa/src/OpenPFC/apps/alloy_pf_karma2001_benchmark/scripts/lumi_paths.sh
  SRC_ROOT="${LUMI_KARMA_SRC}"
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

cmd_configure() {
  load_cpu_modules
  mkdir -p "${LUMI_KARMA_BUILD}" "${LUMI_KARMA_LOGS}"
  local heffte_dir
  heffte_dir="$(heffte_config_dir)"
  cmake -S "${SRC_ROOT}" -B "${LUMI_KARMA_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DHeffte_DIR="${heffte_dir}" \
    -DOpenPFC_ENABLE_CUDA=OFF \
    -DOpenPFC_ENABLE_HIP=OFF \
    -DOpenPFC_ENABLE_HDF5=OFF \
    -DOpenPFC_FETCH_HEFFTE=OFF \
    -DOpenPFC_BUILD_TESTS=OFF \
    -DOpenPFC_BUILD_EXAMPLES=OFF \
    -DOpenPFC_BUILD_BENCHMARKS=OFF
}

cmd_compile() {
  load_cpu_modules
  local jobs="${SLURM_CPUS_PER_TASK:-8}"
  cmake --build "${LUMI_KARMA_BUILD}" -j"${jobs}" --target alloy_pf_karma2001_benchmark_openmp
  ls -l "${LUMI_KARMA_BIN}"
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
