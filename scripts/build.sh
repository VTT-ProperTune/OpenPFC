#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Standalone configure, build, and test driver for OpenPFC on supported machines.
# Agents and humans should use this script instead of invoking cmake/ctest by hand.

if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Empty MACHINE → auto-detect (LUMI login/compute vs Tohtori).
MACHINE="${MACHINE:-}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-}"
ADD_TIMESTAMP="${ADD_TIMESTAMP:-0}"
WITH_CUDA="${WITH_CUDA:-0}"
WITH_ROCM="${WITH_ROCM:-0}"
BACKEND_EXPLICIT=0
RUN_TESTS="${RUN_TESTS:-1}"
RUN_MPI_TESTS="${RUN_MPI_TESTS:-1}"
JOBS="${JOBS:-32}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"
SUBMIT="${SUBMIT:-}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-}"

OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.1}"
ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.1}"
# Tohtori's site openmpi/5.0.10 module links UCX 1.17 built WITHOUT --with-cuda
# (confirmed via `ucx_info -v` on the shared libs OpenMPI actually loads:
# /share/apps/ucx/1.17). Open MPI's own MPIX_Query_cuda_support() probe still
# reports CUDA support (it only checks its own layer), but UCX's shared-memory
# transport then memcpy()s a raw device pointer as host memory and segfaults
# the moment OpenPFC hands one to MPI_Send/Isend with GPU-aware MPI enabled.
#
# scripts/build_tohtori.sh --cuda builds a genuinely CUDA-aware replacement
# (UCX with cuda_copy/cuda_ipc transports + Open MPI's accelerator/cuda
# component) at OPENMPI_ROOT_CUDA. When that install exists, CUDA builds use
# it automatically in place of the site module and MPI_CUDA_AWARE defaults ON;
# otherwise this falls back to the old safe default (site module, OFF).
OPENMPI_ROOT_CUDA="${OPENMPI_ROOT_CUDA:-${HOME}/opt/openmpi/5.0.10-cuda}"
OPENPFC_GCC_MODULE_CUDA="${OPENPFC_GCC_MODULE_CUDA:-gcc/15.2.0}"
MPI_CUDA_AWARE="${MPI_CUDA_AWARE:-}"
MPI_HIP_AWARE="${MPI_HIP_AWARE:-}"
HEFFTE_VERSION="${HEFFTE_VERSION:-2.4.1}"
HEFFTE_PREFIX="${HEFFTE_PREFIX:-}"
HEFFTE_MODULE="${HEFFTE_MODULE:-}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
ROCM_ARCHITECTURES="${ROCM_ARCHITECTURES:-}"

LUMI_STACK="${LUMI_STACK:-LUMI/25.09}"
LUMI_ACCOUNT="${LUMI_ACCOUNT:-project_462001519}"
LUMI_PARTITION="${LUMI_PARTITION:-standard-g}"
LUMI_GPUS="${LUMI_GPUS:-8}"
LUMI_TIME="${LUMI_TIME:-}"
LUMI_PRIVATE_MODULES="${LUMI_PRIVATE_MODULES:-${HOME}/privatemodules}"
LUMI_FLASH_ROOT="${LUMI_FLASH_ROOT:-/flash/project_462001519/juaho/build}"
LUMI_SCRATCH_LOGS="${LUMI_SCRATCH_LOGS:-/scratch/project_462001519/juaho/logs}"
ENABLE_HDF5="${ENABLE_HDF5:-}"

declare -a EXTRA_CMAKE_ARGS=()
declare -a ORIG_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: ./scripts/build.sh [options]

Canonical configure / build / test entry point for OpenPFC. Do not invoke
cmake / cmake --build / ctest by hand for routine work — this script loads
the correct modules (compiler, MPI, HeFFTe) and machine toolchain.

On Tohtori the default is a 32-way Release CPU build in builds/release.
On LUMI the default is a HIP/ROCm Release build: configure on the login
node (FetchContent needs the network), then submit compile + ctest to a
GPU partition (standard-g by default, or dev-g).

Options:
  --machine=NAME          tohtori or lumi (default: auto-detect from host)
  --build-type=TYPE       Debug or Release
  --build-dir=PATH        Build directory
                          Tohtori default: builds/debug or builds/release
                          LUMI default: /flash/project_462001519/juaho/build/openpfc-lumi-<backend>-<flavor>
  --with-timestamp        Append YYYYmmdd-HHMMSS to the build directory
  --without-timestamp     Do not append a timestamp (default)
  --with-cuda             Enable CUDA (Tohtori only; not available on LUMI)
  --with-rocm             Enable HIP/ROCm (LUMI default)
  --cpu                   Disable CUDA and ROCm
  --test                  Run Python tests and CTest after building (default)
  --no-test               Configure and build without running tests
  --mpi-tests             Register the 2-, 3-, and 4-rank MPI suites (default)
  --no-mpi-tests          Do not register the explicit multi-rank MPI suites
  --jobs=N, -j N          Parallel build/test jobs (default: 32)
  --clean                 Remove the selected build directory before configuring
  --cmake-arg=ARG         Append one argument to the CMake configure command
  --submit                LUMI: submit compile+test to Slurm (default on a login node)
  --no-submit             LUMI: run configure/build/test in this process
  --partition=NAME        LUMI GPU partition: standard-g (default) or dev-g
  --account=NAME          LUMI Slurm account (default: project_462001519)
  --time=LIMIT            LUMI Slurm time limit (dev-g default 02:30:00)
  --gpus=N                LUMI GPUs per node (default: 8)
  --wait                  LUMI: sbatch --wait (block until the job finishes)
  -h, --help              Show this help

Environment variables mirror the CLI:
  MACHINE, BUILD_TYPE, BUILD_DIR, ADD_TIMESTAMP, WITH_CUDA, WITH_ROCM,
  RUN_TESTS, RUN_MPI_TESTS, JOBS, CLEAN_BUILD, SUBMIT,
  LUMI_ACCOUNT, LUMI_PARTITION, LUMI_GPUS, LUMI_TIME, LUMI_STACK,
  HEFFTE_MODULE, HEFFTE_PREFIX

Tohtori environment overrides:
  OPENMPI_MODULE          Default: openmpi/5.0.10
  CUDA_MODULE             Default: cuda/13.1
  ROCM_MODULE             Default: rocm/7.2.1
  HEFFTE_PREFIX           Backend-specific install prefix; selected automatically
  HEFFTE_MODULE           Optional module to load; CUDA defaults to
                          heffte/2.4.1-cuda-openmpi5
  HEFFTE_VERSION          Default: 2.4.1
  CUDA_ARCHITECTURES      Passed to CMAKE_CUDA_ARCHITECTURES (CUDA default: 90)
  ROCM_ARCHITECTURES      Passed to CMAKE_HIP_ARCHITECTURES when set
  OPENMPI_ROOT_CUDA       Default: $HOME/opt/openmpi/5.0.10-cuda. When this
                          contains a working mpicc, CUDA builds use it in
                          place of the site openmpi/5.0.10 module (whose
                          linked UCX 1.17 lacks --with-cuda and segfaults on
                          GPU-Direct MPI sends despite Open MPI's own probe
                          claiming support) and MPI_CUDA_AWARE defaults ON.
  OPENPFC_GCC_MODULE_CUDA Compiler module for OPENMPI_ROOT_CUDA (default: gcc/15.2.0).
  MPI_CUDA_AWARE          Default: auto (1 when OPENMPI_ROOT_CUDA is usable, else 0).
  MPI_HIP_AWARE           Default: 0 on Tohtori, 1 on LUMI (Cray MPICH +
                          MPICH_GPU_SUPPORT_ENABLED=1).

LUMI environment:
  HEFFTE_MODULE           Default: heffte-rocm (from $HOME/privatemodules)
  LUMI_STACK              Default: LUMI/25.09
  LUMI_ACCOUNT            Default: project_462001519
  LUMI_PARTITION          Default: standard-g (use dev-g for the short queue)
  LUMI_GPUS               Default: 8
  LUMI_TIME               Default: 02:30:00 on dev-g, 06:00:00 on standard-g
  LUMI_FLASH_ROOT         Default: /flash/project_462001519/juaho/build
  LUMI_SCRATCH_LOGS       Default: /scratch/project_462001519/juaho/logs
  ENABLE_HDF5             Default: ON on Tohtori, OFF on LUMI

Examples:
  ./scripts/build.sh
  ./scripts/build.sh --build-type=Debug --with-timestamp
  ./scripts/build.sh --machine=tohtori --with-cuda --with-timestamp --test
  ./scripts/build.sh --machine=lumi --with-rocm
  ./scripts/build.sh --machine=lumi --partition=standard-g --wait
  WITH_ROCM=1 ADD_TIMESTAMP=1 JOBS=32 ./scripts/build.sh
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2-}"
  [[ -n "${value}" ]] || die "${option} requires a value"
}

as_bool() {
  case "${1,,}" in
    1|on|true|yes) echo 1 ;;
    0|off|false|no) echo 0 ;;
    *) die "invalid boolean value '${1}' (use 1/0, on/off, true/false, or yes/no)" ;;
  esac
}

detect_machine() {
  local host
  host="$(hostname -s 2>/dev/null || hostname)"
  if [[ -n "${LUMI_LMOD_FAMILY_LUMI:-}" || -d /appl/lumi || "${host}" == uan* || "${host}" == nid* ]]; then
    echo lumi
  else
    echo tohtori
  fi
}

on_slurm_job() {
  [[ -n "${SLURM_JOB_ID:-}" ]]
}

WAIT_FOR_JOB=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine=*) MACHINE="${1#*=}" ;;
    --machine)
      require_value "$1" "${2-}"
      MACHINE="$2"
      shift
      ;;
    --build-type=*) BUILD_TYPE="${1#*=}" ;;
    --build-type)
      require_value "$1" "${2-}"
      BUILD_TYPE="$2"
      shift
      ;;
    --build-dir=*) BUILD_DIR="${1#*=}" ;;
    --build-dir)
      require_value "$1" "${2-}"
      BUILD_DIR="$2"
      shift
      ;;
    --with-timestamp) ADD_TIMESTAMP=1 ;;
    --without-timestamp) ADD_TIMESTAMP=0 ;;
    --with-cuda) WITH_CUDA=1; WITH_ROCM=0; BACKEND_EXPLICIT=1 ;;
    --with-rocm) WITH_ROCM=1; WITH_CUDA=0; BACKEND_EXPLICIT=1 ;;
    --cpu) WITH_CUDA=0; WITH_ROCM=0; BACKEND_EXPLICIT=1 ;;
    --test) RUN_TESTS=1 ;;
    --no-test) RUN_TESTS=0 ;;
    --mpi-tests) RUN_MPI_TESTS=1 ;;
    --no-mpi-tests) RUN_MPI_TESTS=0 ;;
    --jobs=*) JOBS="${1#*=}" ;;
    --jobs|-j)
      require_value "$1" "${2-}"
      JOBS="$2"
      shift
      ;;
    --clean) CLEAN_BUILD=1 ;;
    --cmake-arg=*) EXTRA_CMAKE_ARGS+=("${1#*=}") ;;
    --submit) SUBMIT=1 ;;
    --no-submit) SUBMIT=0 ;;
    --partition=*) LUMI_PARTITION="${1#*=}" ;;
    --partition)
      require_value "$1" "${2-}"
      LUMI_PARTITION="$2"
      shift
      ;;
    --account=*) LUMI_ACCOUNT="${1#*=}" ;;
    --account)
      require_value "$1" "${2-}"
      LUMI_ACCOUNT="$2"
      shift
      ;;
    --time=*) LUMI_TIME="${1#*=}" ;;
    --time)
      require_value "$1" "${2-}"
      LUMI_TIME="$2"
      shift
      ;;
    --gpus=*) LUMI_GPUS="${1#*=}" ;;
    --gpus)
      require_value "$1" "${2-}"
      LUMI_GPUS="$2"
      shift
      ;;
    --wait) WAIT_FOR_JOB=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
  shift
done

ADD_TIMESTAMP="$(as_bool "${ADD_TIMESTAMP}")"
WITH_CUDA="$(as_bool "${WITH_CUDA}")"
WITH_ROCM="$(as_bool "${WITH_ROCM}")"
RUN_TESTS="$(as_bool "${RUN_TESTS}")"
RUN_MPI_TESTS="$(as_bool "${RUN_MPI_TESTS}")"
CLEAN_BUILD="$(as_bool "${CLEAN_BUILD}")"

if [[ -z "${MACHINE}" ]]; then
  MACHINE="$(detect_machine)"
fi
case "${MACHINE,,}" in
  tohtori) MACHINE="tohtori" ;;
  lumi) MACHINE="lumi" ;;
  *) die "unsupported machine '${MACHINE}'; expected tohtori or lumi" ;;
esac

if [[ "${MACHINE}" == "lumi" && "${BACKEND_EXPLICIT}" -eq 0 ]]; then
  WITH_ROCM=1
  WITH_CUDA=0
fi

if [[ "${MACHINE}" == "lumi" && "${WITH_CUDA}" -eq 1 ]]; then
  die "CUDA is not available on LUMI (AMD/HIP only). Use --with-rocm or --cpu. Run CUDA builds on Tohtori."
fi

if [[ -z "${SUBMIT}" ]]; then
  if [[ "${MACHINE}" == "lumi" ]] && ! on_slurm_job; then
    SUBMIT=1
  else
    SUBMIT=0
  fi
fi
SUBMIT="$(as_bool "${SUBMIT}")"

# Auto-detect the custom CUDA-aware Open MPI built by
# `scripts/build_tohtori.sh --build-ucx --build-openmpi --cuda` (see
# OPENMPI_ROOT_CUDA above). Only engages for CUDA builds, and only when the
# caller hasn't already pointed OPENMPI_ROOT somewhere else themselves.
USE_CUSTOM_CUDA_MPI=0
if (( WITH_CUDA )) && [[ -z "${OPENMPI_ROOT:-}" ]] &&
   [[ -x "${OPENMPI_ROOT_CUDA}/bin/mpicc" ]]; then
  USE_CUSTOM_CUDA_MPI=1
fi

if [[ -z "${MPI_CUDA_AWARE}" ]]; then
  MPI_CUDA_AWARE=$(( USE_CUSTOM_CUDA_MPI ? 1 : 0 ))
fi
if [[ -z "${MPI_HIP_AWARE}" ]]; then
  if [[ "${MACHINE}" == "lumi" ]]; then
    MPI_HIP_AWARE=1
  else
    MPI_HIP_AWARE=0
  fi
fi
MPI_CUDA_AWARE="$(as_bool "${MPI_CUDA_AWARE}")"
MPI_HIP_AWARE="$(as_bool "${MPI_HIP_AWARE}")"

if [[ -z "${ENABLE_HDF5}" ]]; then
  if [[ "${MACHINE}" == "lumi" ]]; then
    ENABLE_HDF5=0
  else
    ENABLE_HDF5=1
  fi
fi
ENABLE_HDF5="$(as_bool "${ENABLE_HDF5}")"

case "${BUILD_TYPE,,}" in
  debug) BUILD_TYPE="Debug"; BUILD_FLAVOR="debug" ;;
  release) BUILD_TYPE="Release"; BUILD_FLAVOR="release" ;;
  *) die "unsupported build type '${BUILD_TYPE}'; expected Debug or Release" ;;
esac

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"
(( WITH_CUDA == 0 || WITH_ROCM == 0 )) ||
  die "WITH_CUDA and WITH_ROCM cannot both be enabled"
[[ "${LUMI_GPUS}" =~ ^[1-9][0-9]*$ ]] || die "LUMI_GPUS must be a positive integer"

BACKEND="cpu"
if (( WITH_CUDA )); then
  BACKEND="cuda"
  if [[ -z "${CUDA_ARCHITECTURES}" ]]; then
    CUDA_ARCHITECTURES="90"
  fi
  # heffte/2.4.1-cuda-openmpi5 depends_on the site openmpi/5.0.10 module,
  # which would re-prepend the site MPI onto PATH and undo the custom stack
  # below. HEFFTE_PREFIX + CMAKE_PREFIX_PATH (already set up further down)
  # are sufficient without the module in that case.
  if [[ -z "${HEFFTE_MODULE}" ]] && (( ! USE_CUSTOM_CUDA_MPI )); then
    HEFFTE_MODULE="heffte/${HEFFTE_VERSION}-cuda-openmpi5"
  fi
elif (( WITH_ROCM )); then
  BACKEND="rocm"
  if [[ "${MACHINE}" == "lumi" && -z "${ROCM_ARCHITECTURES}" ]]; then
    ROCM_ARCHITECTURES="gfx90a"
  fi
  if [[ "${MACHINE}" == "lumi" && -z "${HEFFTE_MODULE}" ]]; then
    HEFFTE_MODULE="heffte-rocm"
  fi
fi

if [[ -z "${BUILD_DIR}" ]]; then
  if [[ "${MACHINE}" == "lumi" ]]; then
    BUILD_DIR="${LUMI_FLASH_ROOT}/openpfc-lumi-${BACKEND}-${BUILD_FLAVOR}"
  else
    BUILD_DIR="builds/${BUILD_FLAVOR}"
  fi
fi
if (( ADD_TIMESTAMP )); then
  BUILD_DIR="${BUILD_DIR}-$(date +%Y%m%d-%H%M%S)"
fi
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

case "${BUILD_DIR}" in
  ""|"/"|"${REPO_ROOT}") die "refusing unsafe build directory '${BUILD_DIR}'" ;;
esac

if [[ "${MACHINE}" == "lumi" && "${BUILD_DIR}" == "${REPO_ROOT}"/* ]]; then
  echo "WARNING: LUMI build directory is inside the git clone (${BUILD_DIR})." >&2
  echo "         Prefer --build-dir under ${LUMI_FLASH_ROOT} (inode quota)." >&2
fi

if [[ "${MACHINE}" == "tohtori" ]]; then
  TOOLCHAIN="${REPO_ROOT}/cmake/toolchains/tohtori-gcc11-openmpi.cmake"
else
  TOOLCHAIN="${REPO_ROOT}/cmake/toolchains/lumi-gcc12-mpich.cmake"
fi

CONFIGURE_SECONDS=0
BUILD_SECONDS=0
TEST_SECONDS=0
TEST_BATCHES=0
PYTHON_TESTS="not run"
FAILED_PHASE=""
OVERALL_START="$(date +%s)"
SKIP_SUMMARY=0

format_duration() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

summary() {
  local status="$?"
  if (( SKIP_SUMMARY )); then
    return
  fi
  local total_seconds=$(( $(date +%s) - OVERALL_START ))
  echo
  echo "================================================================"
  if (( status == 0 )); then
    echo "OpenPFC build and test: PASS"
  else
    echo "OpenPFC build and test: FAIL${FAILED_PHASE:+ (${FAILED_PHASE})}"
  fi
  echo "Machine:       ${MACHINE}"
  echo "Backend:       ${BACKEND}"
  echo "Build type:    ${BUILD_TYPE}"
  echo "Build dir:     ${BUILD_DIR}"
  echo "Jobs:          ${JOBS}"
  echo "Configure:     $(format_duration "${CONFIGURE_SECONDS}")"
  echo "Build:         $(format_duration "${BUILD_SECONDS}")"
  if (( RUN_TESTS )); then
    echo "Tests:         $(format_duration "${TEST_SECONDS}")"
    echo "Python tests:  ${PYTHON_TESTS}"
    echo "Test batches:  ${TEST_BATCHES} (aggregate CTest commands)"
  else
    echo "Tests:         skipped"
  fi
  echo "Total:         $(format_duration "${total_seconds}")"
  echo "Logs:          ${BUILD_DIR}/*.log"
  if (( status != 0 )) && [[ -s "${BUILD_DIR}/Testing/Temporary/LastTestsFailed.log" ]]; then
    echo "FAILED TEST BATCHES:"
    sed 's/^/  /' "${BUILD_DIR}/Testing/Temporary/LastTestsFailed.log"
  fi
  echo "Exit code:     ${status}"
  echo "================================================================"
}
trap summary EXIT

FAILED_PHASE="environment"

init_lmod() {
  if command -v module >/dev/null 2>&1; then
    return
  fi
  local init_file
  for init_file in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash \
                   /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [[ -f "${init_file}" ]]; then
      # shellcheck source=/dev/null
      source "${init_file}"
      break
    fi
  done
  command -v module >/dev/null 2>&1 || die "Lmod 'module' command not found"
}

resolve_heffte_dir() {
  HEFFTE_DIR=""
  if [[ -n "${HEFFTE_DIR_ENV:-}" && -f "${HEFFTE_DIR_ENV}/HeffteConfig.cmake" ]]; then
    HEFFTE_DIR="${HEFFTE_DIR_ENV}"
    return
  fi
  local candidate
  for candidate in "${HEFFTE_PREFIX}/lib64/cmake/Heffte" \
                   "${HEFFTE_PREFIX}/lib/cmake/Heffte"; do
    if [[ -f "${candidate}/HeffteConfig.cmake" ]]; then
      HEFFTE_DIR="${candidate}"
      return
    fi
  done
}

# Locate the ROCm prefix that contains hip-config.cmake. Do not use
# `dirname hipcc`/.. alone: on Tohtori hipcc can be /usr/bin/hipcc.
resolve_rocm_path() {
  local cand p dir hipcc
  _rocm_has_hip_config() {
    [[ -f "${1}/lib/cmake/hip/hip-config.cmake" ]] ||
      [[ -f "${1}/lib64/cmake/hip/hip-config.cmake" ]] ||
      [[ -f "${1}/lib/cmake/hip/HIPConfig.cmake" ]]
  }
  for cand in "${ROCM_PATH:-}" "${HIP_PATH:-}" "${EBROOTROCM:-}" \
              /opt/rocm-7.2.1 /opt/rocm; do
    [[ -n "${cand}" ]] || continue
    if _rocm_has_hip_config "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done
  hipcc="$(command -v hipcc)" || return 1
  dir="$(cd "$(dirname "${hipcc}")" && pwd -P)"
  p="${dir}"
  local i
  for i in 1 2 3 4 5; do
    p="$(cd "${p}/.." && pwd -P)" || break
    if _rocm_has_hip_config "${p}"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

setup_tohtori_env() {
  if (( USE_CUSTOM_CUDA_MPI )); then
    echo "Using custom CUDA-aware Open MPI: ${OPENMPI_ROOT_CUDA}"
    module load "${OPENPFC_GCC_MODULE_CUDA}"
    export PATH="${OPENMPI_ROOT_CUDA}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OPENMPI_ROOT_CUDA}/lib64:${OPENMPI_ROOT_CUDA}/lib:${LD_LIBRARY_PATH:-}"
    export OPENMPI_ROOT="${OPENMPI_ROOT_CUDA}"
  else
    module load "${OPENMPI_MODULE}"
  fi
  if (( WITH_CUDA )); then
    module load "${CUDA_MODULE}"
  elif (( WITH_ROCM )); then
    module load "${ROCM_MODULE}"
    command -v hipcc >/dev/null 2>&1 ||
      die "hipcc not found after loading ${ROCM_MODULE}"
    # Tohtori rocm/7.2.1 sets PATH + LD_LIBRARY_PATH but not CMAKE_PREFIX_PATH
    # or ROCM_PATH. `dirname hipcc`/.. is not enough: hipcc may live in
    # /usr/bin, which would set ROCM_PATH=/usr and break enable_language(HIP).
    ROCM_PATH="$(resolve_rocm_path)" ||
      die "could not locate ROCm prefix (hip-config.cmake) after loading ${ROCM_MODULE}"
    export ROCM_PATH
    export CMAKE_PREFIX_PATH="${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    echo "  ROCM_PATH:   ${ROCM_PATH} (hipcc=$(command -v hipcc))"
  fi
  if [[ -n "${HEFFTE_MODULE}" ]]; then
    module load "${HEFFTE_MODULE}"
  fi

  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export OPENPFC_GCC_ROOT="$(cd "$(dirname "${CXX}")/.." && pwd)"
  if (( ! USE_CUSTOM_CUDA_MPI )); then
    unset OPENMPI_ROOT
  fi

  [[ -x "${CC}" && -x "${CXX}" ]] || die "compiler not found after module load"
  command -v mpicc >/dev/null 2>&1 || die "mpicc not found after loading ${OPENMPI_MODULE}"
  command -v mpicxx >/dev/null 2>&1 || die "mpicxx not found after loading ${OPENMPI_MODULE}"

  if [[ -z "${HEFFTE_PREFIX}" ]]; then
    HEFFTE_PREFIX="${HOME}/opt/heffte/${HEFFTE_VERSION}-${BACKEND}"
  fi
}

setup_lumi_env() {
  export PATH="${HOME}/.local/bin:${PATH}"
  module load "${LUMI_STACK}" partition/G cpeGNU cray-fftw lumi-CrayPath
  if [[ -d "${LUMI_PRIVATE_MODULES}" ]]; then
    module use "${LUMI_PRIVATE_MODULES}"
  fi
  if [[ -n "${HEFFTE_MODULE}" ]]; then
    module load "${HEFFTE_MODULE}"
  fi
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
  # FindMPI on Cray PE fails to populate MPI_C_LIB_NAMES unless the PE
  # wrappers use dynamic linking (same as the working LUMI CMake caches).
  export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
  export CC=cc
  export CXX=CC
  unset OPENPFC_GCC_ROOT || true

  command -v cc >/dev/null 2>&1 || die "Cray cc wrapper not found after loading ${LUMI_STACK}"
  command -v CC >/dev/null 2>&1 || die "Cray CC wrapper not found after loading ${LUMI_STACK}"

  if [[ -z "${HEFFTE_PREFIX}" ]]; then
    if [[ -n "${HEFFTE_ROOT:-}" ]]; then
      HEFFTE_PREFIX="${HEFFTE_ROOT}"
    elif [[ -d "${HOME}/opt/heffte/${HEFFTE_VERSION}-rocm" ]]; then
      HEFFTE_PREFIX="${HOME}/opt/heffte/${HEFFTE_VERSION}-rocm"
    else
      HEFFTE_PREFIX="/projappl/project_462001245/heffte/${HEFFTE_VERSION}-rocm"
    fi
  fi
}

init_lmod
module purge
if [[ "${MACHINE}" == "lumi" ]]; then
  setup_lumi_env
else
  setup_tohtori_env
fi

[[ -f "${TOOLCHAIN}" ]] || die "missing toolchain ${TOOLCHAIN}"

HEFFTE_DIR_ENV="${HEFFTE_DIR:-}"
resolve_heffte_dir
[[ -n "${HEFFTE_DIR}" ]] ||
  die "HeFFTe ${BACKEND} package not found (prefix ${HEFFTE_PREFIX}; module '${HEFFTE_MODULE:-none}')"

if [[ -z "${HEFFTE_MODULE}" ]]; then
  # No module loaded to set HeFFTe's runtime LD_LIBRARY_PATH / build-time CPATH
  # (custom-MPI CUDA path skips the module deliberately, see above) — replicate
  # both by hand. CPATH matters because some CUDA test targets #include
  # <heffte.h> directly while only linking `openpfc` (which links Heffte
  # PRIVATE, so its include dir doesn't propagate via CMake target visibility;
  # the site heffte module's own CPATH prepend is what actually made that
  # compile before — see heffte/2.4.1-cuda-openmpi5.lua).
  export LD_LIBRARY_PATH="${HEFFTE_PREFIX}/lib64:${HEFFTE_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export CPATH="${HEFFTE_PREFIX}/include:${CPATH:-}"
fi

if (( CLEAN_BUILD )) && [[ -e "${BUILD_DIR}" ]]; then
  echo "Removing build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

export CMAKE_PREFIX_PATH="${HEFFTE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

if [[ -z "${CMAKE_GENERATOR}" ]]; then
  if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
  else
    CMAKE_GENERATOR="Unix Makefiles"
  fi
fi

declare -a CMAKE_ARGS=(
  -S "${REPO_ROOT}"
  -B "${BUILD_DIR}"
  -G "${CMAKE_GENERATOR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
  -DHeffte_DIR="${HEFFTE_DIR}"
  -DOpenPFC_BUILD_TESTS=$([[ ${RUN_TESTS} -eq 1 ]] && echo ON || echo OFF)
  -DOpenPFC_RUN_MPI_SUITES=$([[ ${RUN_MPI_TESTS} -eq 1 ]] && echo ON || echo OFF)
  -DOpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS=$([[ ${RUN_MPI_TESTS} -eq 1 ]] && echo ON || echo OFF)
  -DOpenPFC_BUILD_APPS=ON
  -DOpenPFC_BUILD_EXAMPLES=ON
  -DOpenPFC_BUILD_DOCUMENTATION=OFF
  -DOpenPFC_ENABLE_CODE_COVERAGE=OFF
  -DOpenPFC_ENABLE_HDF5=$([[ ${ENABLE_HDF5} -eq 1 ]] && echo ON || echo OFF)
  -DOpenPFC_ENABLE_CUDA=$([[ ${WITH_CUDA} -eq 1 ]] && echo ON || echo OFF)
  -DOpenPFC_ENABLE_HIP=$([[ ${WITH_ROCM} -eq 1 ]] && echo ON || echo OFF)
)

if [[ -n "${CUDA_ARCHITECTURES}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
fi
if [[ -n "${ROCM_ARCHITECTURES}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCHITECTURES}")
fi
if (( WITH_CUDA )); then
  CMAKE_ARGS+=("-DOpenPFC_MPI_CUDA_AWARE=$([[ ${MPI_CUDA_AWARE} -eq 1 ]] && echo ON || echo OFF)")
fi
if (( WITH_ROCM )); then
  CMAKE_ARGS+=("-DOpenPFC_MPI_HIP_AWARE=$([[ ${MPI_HIP_AWARE} -eq 1 ]] && echo ON || echo OFF)")
  if [[ "${MACHINE}" == "tohtori" && -n "${OPENPFC_GCC_ROOT:-}" ]]; then
    # ROCm clang compiles .hip TUs; without GCC's libstdc++ it cannot find
    # C++20 headers (<span>, <compare>, ...).
    CMAKE_ARGS+=("-DCMAKE_HIP_FLAGS=--gcc-toolchain=${OPENPFC_GCC_ROOT} -stdlib=libstdc++")
  fi
  if [[ "${MACHINE}" == "lumi" ]]; then
    CMAKE_ARGS+=("-DGPU_TARGETS=${ROCM_ARCHITECTURES:-gfx90a}")
  fi
fi
if [[ "${MACHINE}" == "lumi" ]]; then
  CMAKE_ARGS+=(
    "-DMPI_C_COMPILER=$(command -v cc)"
    "-DMPI_CXX_COMPILER=$(command -v CC)"
  )
  if [[ -n "${MPICH_DIR:-}" && -f "${MPICH_DIR}/include/mpi.h" ]]; then
    CMAKE_ARGS+=("-DMPI_C_HEADER_DIR=${MPICH_DIR}/include")
  elif [[ -f /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include/mpi.h ]]; then
    CMAKE_ARGS+=("-DMPI_C_HEADER_DIR=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include")
  fi
fi
CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")

echo "OpenPFC automated build"
echo "  machine:    ${MACHINE}"
echo "  backend:    ${BACKEND}"
echo "  build type: ${BUILD_TYPE}"
echo "  build dir:  ${BUILD_DIR}"
echo "  generator:  ${CMAKE_GENERATOR}"
echo "  jobs:       ${JOBS}"
echo "  tests:      $([[ ${RUN_TESTS} -eq 1 ]] && echo enabled || echo disabled)"
echo "  MPI suites: $([[ ${RUN_MPI_TESTS} -eq 1 ]] && echo enabled || echo disabled)"
echo "  compiler:   ${CXX} ($(command -v "${CXX}"))"
if [[ "${MACHINE}" == "tohtori" ]]; then
  echo "  MPI:        $(command -v mpicxx)$([[ ${USE_CUSTOM_CUDA_MPI} -eq 1 ]] && echo " (custom CUDA-aware build)")"
else
  echo "  MPI:        Cray MPICH (cc/CC wrappers)"
  echo "  partition:  ${LUMI_PARTITION}  account=${LUMI_ACCOUNT}"
  echo "  submit:     $([[ ${SUBMIT} -eq 1 ]] && echo yes || echo no)"
fi
if (( WITH_CUDA )); then
  echo "  MPI CUDA-aware: $([[ ${MPI_CUDA_AWARE} -eq 1 ]] && echo ON || echo OFF)"
fi
if (( WITH_ROCM )); then
  echo "  MPI HIP-aware:  $([[ ${MPI_HIP_AWARE} -eq 1 ]] && echo ON || echo OFF)"
fi
echo "  HeFFTe:     ${HEFFTE_DIR}"
if [[ -n "${HEFFTE_MODULE}" ]]; then
  echo "  HeFFTe mod: ${HEFFTE_MODULE}"
fi

run_configure() {
  phase_start="$(date +%s)"
  FAILED_PHASE="configure"
  if ! cmake "${CMAKE_ARGS[@]}" 2>&1 | tee "${BUILD_DIR}/configure.log"; then
    CONFIGURE_SECONDS=$(( $(date +%s) - phase_start ))
    FAILED_PHASE="configure"
    exit 1
  fi
  CONFIGURE_SECONDS=$(( $(date +%s) - phase_start ))
  if (( WITH_ROCM )); then
    if ! grep -q 'OpenPFC_ENABLE_HIP.*= ON (✅ HIP available)' "${BUILD_DIR}/configure.log"; then
      die "HIP was requested (--with-rocm) but CMake did not enable it (see ${BUILD_DIR}/configure.log)"
    fi
  fi
}

submit_lumi_job() {
  case "${LUMI_PARTITION}" in
    dev-g|standard-g) ;;
    *) die "LUMI partition must be dev-g or standard-g (got '${LUMI_PARTITION}')" ;;
  esac
  if [[ -z "${LUMI_TIME}" ]]; then
    if [[ "${LUMI_PARTITION}" == "dev-g" ]]; then
      LUMI_TIME="02:30:00"
    else
      LUMI_TIME="06:00:00"
    fi
  fi
  mkdir -p "${LUMI_SCRATCH_LOGS}"
  local sbatch_file="${BUILD_DIR}/lumi-build.sbatch"
  local ntasks="${LUMI_GPUS}"
  if (( ntasks < 8 )); then
    ntasks=8
  fi
  local cpus_per_task=7
  cat > "${sbatch_file}" <<EOF
#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Generated by scripts/build.sh — compile and test OpenPFC on a LUMI-G node.
#SBATCH --account=${LUMI_ACCOUNT}
#SBATCH --partition=${LUMI_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${ntasks}
#SBATCH --gpus-per-node=${LUMI_GPUS}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --time=${LUMI_TIME}
#SBATCH --job-name=openpfc-build
#SBATCH --output=${LUMI_SCRATCH_LOGS}/%x-%j.out
#SBATCH --error=${LUMI_SCRATCH_LOGS}/%x-%j.err

set -euo pipefail
export PATH="\${HOME}/.local/bin:\${PATH}"
cd "${REPO_ROOT}"
exec "${REPO_ROOT}/scripts/build.sh" \\
  --machine=lumi \\
  --no-submit \\
  --build-type=${BUILD_TYPE} \\
  --build-dir=${BUILD_DIR} \\
  --jobs=${JOBS} \\
  $([[ ${WITH_ROCM} -eq 1 ]] && echo --with-rocm || echo --cpu) \\
  $([[ ${RUN_TESTS} -eq 1 ]] && echo --test || echo --no-test) \\
  $([[ ${RUN_MPI_TESTS} -eq 1 ]] && echo --mpi-tests || echo --no-mpi-tests)
EOF
  chmod +x "${sbatch_file}"

  local -a sbatch_cmd=(sbatch --account="${LUMI_ACCOUNT}"
                         --partition="${LUMI_PARTITION}")
  if (( WAIT_FOR_JOB )); then
    sbatch_cmd+=(--wait)
  fi
  sbatch_cmd+=("${sbatch_file}")
  # Login shells here export SBATCH_ACCOUNT=project_462001245, which
  # overrides #SBATCH --account. The CLI flag above wins; drop the env
  # so a later nested sbatch cannot silently revert.
  unset SBATCH_ACCOUNT SLURM_ACCOUNT SBATCH_PARTITION SLURM_PARTITION

  echo
  echo "Submitting LUMI GPU job:"
  echo "  script:    ${sbatch_file}"
  echo "  partition: ${LUMI_PARTITION}"
  echo "  account:   ${LUMI_ACCOUNT}"
  echo "  gpus:      ${LUMI_GPUS}"
  echo "  time:      ${LUMI_TIME}"
  echo "  logs:      ${LUMI_SCRATCH_LOGS}/openpfc-build-<jobid>.out"
  echo "  (configure already ran on this login node so FetchContent can use the network)"
  echo

  local sbatch_out
  sbatch_out="$("${sbatch_cmd[@]}")"
  echo "${sbatch_out}"
  SKIP_SUMMARY=1
  echo "OpenPFC LUMI job submitted. Compile and ctest run on ${LUMI_PARTITION}."
  echo "Watch:  squeue -u ${USER}"
  echo "Logs:   ${LUMI_SCRATCH_LOGS}/openpfc-build-*.out"
  if (( ! WAIT_FOR_JOB )); then
    echo "Re-run with --wait to block until the job finishes."
  fi
}

# LUMI login nodes have outbound HTTP (FetchContent). Compute nodes often do
# not. Configure here whenever we are about to submit, then let the GPU job
# rebuild and test.
run_configure

if (( RUN_TESTS )) && [[ "${SUBMIT}" -eq 0 ]]; then
  TEST_BATCHES="$(ctest --test-dir "${BUILD_DIR}" -N 2>/dev/null |
    awk '/Total Tests:/ {print $3}')"
  [[ "${TEST_BATCHES}" =~ ^[0-9]+$ ]] || die "could not enumerate CTest batches"
  echo "Registered CTest batches: ${TEST_BATCHES}"
fi

if [[ "${MACHINE}" == "lumi" && "${SUBMIT}" -eq 1 ]]; then
  if on_slurm_job; then
    echo "Already inside Slurm job ${SLURM_JOB_ID}; not re-submitting."
  else
    FAILED_PHASE="submit"
    submit_lumi_job
    exit 0
  fi
fi

phase_start="$(date +%s)"
FAILED_PHASE="build"
if ! cmake --build "${BUILD_DIR}" --parallel "${JOBS}" 2>&1 |
     tee "${BUILD_DIR}/build.log"; then
  BUILD_SECONDS=$(( $(date +%s) - phase_start ))
  FAILED_PHASE="build"
  exit 1
fi
BUILD_SECONDS=$(( $(date +%s) - phase_start ))

if (( RUN_TESTS )); then
  phase_start="$(date +%s)"
  FAILED_PHASE="tests"
  if [[ -d "${REPO_ROOT}/scripts/tests" ]]; then
    # check_doc_links.py needs Python 3.8+ (from __future__ import annotations).
    # LUMI compute nodes often have /usr/bin/python3 = 3.6 plus a newer
    # python3.11 without pytest — skip rather than fail a GPU job for that.
    PYTEST_PYTHON=""
    for cand in ${PYTHON:-} python3.12 python3.11 python3; do
      [[ -n "${cand}" ]] || continue
      command -v "${cand}" >/dev/null 2>&1 || continue
      if "${cand}" -c 'import sys, pytest; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' \
           >/dev/null 2>&1; then
        PYTEST_PYTHON="${cand}"
        break
      fi
    done
    if [[ -n "${PYTEST_PYTHON}" ]]; then
      if ! "${PYTEST_PYTHON}" -m pytest "${REPO_ROOT}/scripts/tests" 2>&1 |
           tee "${BUILD_DIR}/python-test.log"; then
        TEST_SECONDS=$(( $(date +%s) - phase_start ))
        FAILED_PHASE="python tests"
        exit 1
      fi
      PYTHON_TESTS="passed (${PYTEST_PYTHON})"
    else
      echo "pytest+Python>=3.8 not available; skipping scripts/tests"
      PYTHON_TESTS="skipped (need Python>=3.8 with pytest)"
    fi
  else
    PYTHON_TESTS="not found"
  fi
  if ! ctest --test-dir "${BUILD_DIR}" --output-on-failure --parallel "${JOBS}" 2>&1 |
       tee "${BUILD_DIR}/test.log"; then
    TEST_SECONDS=$(( $(date +%s) - phase_start ))
    FAILED_PHASE="tests"
    exit 1
  fi
  TEST_SECONDS=$(( $(date +%s) - phase_start ))
fi

FAILED_PHASE=""
