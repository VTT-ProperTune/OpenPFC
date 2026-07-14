#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Standalone configure, build, and test driver for OpenPFC on supported machines.

if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MACHINE="${MACHINE:-tohtori}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-}"
ADD_TIMESTAMP="${ADD_TIMESTAMP:-0}"
WITH_CUDA="${WITH_CUDA:-0}"
WITH_ROCM="${WITH_ROCM:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_MPI_TESTS="${RUN_MPI_TESTS:-1}"
JOBS="${JOBS:-32}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"

OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.1}"
ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.1}"
HEFFTE_VERSION="${HEFFTE_VERSION:-2.4.1}"
HEFFTE_PREFIX="${HEFFTE_PREFIX:-}"
HEFFTE_MODULE="${HEFFTE_MODULE:-}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
ROCM_ARCHITECTURES="${ROCM_ARCHITECTURES:-}"

declare -a EXTRA_CMAKE_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/build.sh [options]

Configure, build, and test OpenPFC. The default is a 32-way Release CPU build
on Tohtori with tests enabled.

Options:
  --machine=NAME          Machine configuration (currently: tohtori)
  --build-type=TYPE       Debug or Release
  --build-dir=PATH        Build directory (default: builds/debug or builds/release)
  --with-timestamp        Append YYYYmmdd-HHMMSS to the build directory
  --without-timestamp     Do not append a timestamp (default)
  --with-cuda             Enable CUDA and use the CUDA HeFFTe prefix
  --with-rocm             Enable HIP/ROCm and use the ROCm HeFFTe prefix
  --cpu                   Disable CUDA and ROCm
  --test                  Run Python tests and CTest after building (default)
  --no-test               Configure and build without running tests
  --mpi-tests             Register the 2-, 3-, and 4-rank MPI suites (default)
  --no-mpi-tests          Do not register the explicit multi-rank MPI suites
  --jobs=N, -j N          Parallel build/test jobs (default: 32)
  --clean                 Remove the selected build directory before configuring
  --cmake-arg=ARG         Append one argument to the CMake configure command
  -h, --help              Show this help

Environment variables mirror the CLI:
  MACHINE, BUILD_TYPE, BUILD_DIR, ADD_TIMESTAMP, WITH_CUDA, WITH_ROCM,
  RUN_TESTS, RUN_MPI_TESTS, JOBS, CLEAN_BUILD

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

Examples:
  ./scripts/build.sh
  ./scripts/build.sh --build-type=Debug --with-timestamp
  ./scripts/build.sh --machine=tohtori --with-cuda --with-timestamp --test
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
    --with-cuda) WITH_CUDA=1; WITH_ROCM=0 ;;
    --with-rocm) WITH_ROCM=1; WITH_CUDA=0 ;;
    --cpu) WITH_CUDA=0; WITH_ROCM=0 ;;
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

case "${BUILD_TYPE,,}" in
  debug) BUILD_TYPE="Debug"; BUILD_FLAVOR="debug" ;;
  release) BUILD_TYPE="Release"; BUILD_FLAVOR="release" ;;
  *) die "unsupported build type '${BUILD_TYPE}'; expected Debug or Release" ;;
esac

[[ "${MACHINE,,}" == "tohtori" ]] ||
  die "unsupported machine '${MACHINE}'; currently only tohtori is available"
MACHINE="tohtori"

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"
(( WITH_CUDA == 0 || WITH_ROCM == 0 )) ||
  die "WITH_CUDA and WITH_ROCM cannot both be enabled"

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="builds/${BUILD_FLAVOR}"
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

BACKEND="cpu"
if (( WITH_CUDA )); then
  BACKEND="cuda"
  if [[ -z "${CUDA_ARCHITECTURES}" ]]; then
    CUDA_ARCHITECTURES="90"
  fi
  if [[ -z "${HEFFTE_MODULE}" ]]; then
    HEFFTE_MODULE="heffte/${HEFFTE_VERSION}-cuda-openmpi5"
  fi
elif (( WITH_ROCM )); then
  BACKEND="rocm"
fi

if [[ -z "${HEFFTE_PREFIX}" ]]; then
  HEFFTE_PREFIX="${HOME}/opt/heffte/${HEFFTE_VERSION}-${BACKEND}"
fi

TOOLCHAIN="${REPO_ROOT}/cmake/toolchains/tohtori-gcc11-openmpi.cmake"
CONFIGURE_SECONDS=0
BUILD_SECONDS=0
TEST_SECONDS=0
TEST_BATCHES=0
PYTHON_TESTS="not run"
FAILED_PHASE=""
OVERALL_START="$(date +%s)"

format_duration() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

summary() {
  local status="$?"
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

init_lmod
module purge
module load "${OPENMPI_MODULE}"
if (( WITH_CUDA )); then
  module load "${CUDA_MODULE}"
elif (( WITH_ROCM )); then
  module load "${ROCM_MODULE}"
fi
if [[ -n "${HEFFTE_MODULE}" ]]; then
  module load "${HEFFTE_MODULE}"
fi

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export OPENPFC_GCC_ROOT="$(cd "$(dirname "${CXX}")/.." && pwd)"
unset OPENMPI_ROOT

[[ -x "${CC}" && -x "${CXX}" ]] || die "compiler not found after module load"
command -v mpicc >/dev/null 2>&1 || die "mpicc not found after loading ${OPENMPI_MODULE}"
command -v mpicxx >/dev/null 2>&1 || die "mpicxx not found after loading ${OPENMPI_MODULE}"
[[ -f "${TOOLCHAIN}" ]] || die "missing toolchain ${TOOLCHAIN}"

HEFFTE_DIR=""
for candidate in "${HEFFTE_PREFIX}/lib64/cmake/Heffte" \
                 "${HEFFTE_PREFIX}/lib/cmake/Heffte"; do
  if [[ -f "${candidate}/HeffteConfig.cmake" ]]; then
    HEFFTE_DIR="${candidate}"
    break
  fi
done
[[ -n "${HEFFTE_DIR}" ]] ||
  die "HeFFTe ${BACKEND} package not found under ${HEFFTE_PREFIX}"

if (( CLEAN_BUILD )) && [[ -e "${BUILD_DIR}" ]]; then
  echo "Removing build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

export CMAKE_PREFIX_PATH="${HEFFTE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

declare -a CMAKE_ARGS=(
  -S "${REPO_ROOT}"
  -B "${BUILD_DIR}"
  -G Ninja
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
  -DOpenPFC_ENABLE_HDF5=ON
  -DOpenPFC_ENABLE_CUDA=$([[ ${WITH_CUDA} -eq 1 ]] && echo ON || echo OFF)
  -DOpenPFC_ENABLE_HIP=$([[ ${WITH_ROCM} -eq 1 ]] && echo ON || echo OFF)
)

if [[ -n "${CUDA_ARCHITECTURES}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
fi
if [[ -n "${ROCM_ARCHITECTURES}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCHITECTURES}")
fi
CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")

echo "OpenPFC automated build"
echo "  machine:    ${MACHINE}"
echo "  backend:    ${BACKEND}"
echo "  build type: ${BUILD_TYPE}"
echo "  build dir:  ${BUILD_DIR}"
echo "  jobs:       ${JOBS}"
echo "  tests:      $([[ ${RUN_TESTS} -eq 1 ]] && echo enabled || echo disabled)"
echo "  MPI suites: $([[ ${RUN_MPI_TESTS} -eq 1 ]] && echo enabled || echo disabled)"
echo "  compiler:   ${CXX}"
echo "  MPI:        $(command -v mpicxx)"
echo "  HeFFTe:     ${HEFFTE_DIR}"
if [[ -n "${HEFFTE_MODULE}" ]]; then
  echo "  HeFFTe mod: ${HEFFTE_MODULE}"
fi

phase_start="$(date +%s)"
FAILED_PHASE="configure"
if ! cmake "${CMAKE_ARGS[@]}" 2>&1 | tee "${BUILD_DIR}/configure.log"; then
  CONFIGURE_SECONDS=$(( $(date +%s) - phase_start ))
  FAILED_PHASE="configure"
  exit 1
fi
CONFIGURE_SECONDS=$(( $(date +%s) - phase_start ))

if (( RUN_TESTS )); then
  TEST_BATCHES="$(ctest --test-dir "${BUILD_DIR}" -N 2>/dev/null |
    awk '/Total Tests:/ {print $3}')"
  [[ "${TEST_BATCHES}" =~ ^[0-9]+$ ]] || die "could not enumerate CTest batches"
  echo "Registered CTest batches: ${TEST_BATCHES}"
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
    command -v python3 >/dev/null 2>&1 || die "python3 not found"
    if ! python3 -m pytest "${REPO_ROOT}/scripts/tests" 2>&1 |
         tee "${BUILD_DIR}/python-test.log"; then
      TEST_SECONDS=$(( $(date +%s) - phase_start ))
      FAILED_PHASE="python tests"
      exit 1
    fi
    PYTHON_TESTS="passed"
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
