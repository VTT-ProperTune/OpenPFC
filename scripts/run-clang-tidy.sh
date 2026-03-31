#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run clang-tidy on OpenPFC sources (.clang-tidy, header filter). By default only
# diagnostics promoted by .clang-tidy (WarningsAsErrors) fail; with --fail-fast,
# also passes --warnings-as-errors=* and exits on the first failing file.
# Requires a Ninja build directory with compile_commands.json (see --configure).
#
# Typical tohtori workflow (load modules first: gcc/11.2.0, openmpi/4.1.1):
#   ./scripts/run-clang-tidy.sh --configure
#   ./scripts/run-clang-tidy.sh
#
# Environment:
#   OPENPFC_TIDY_BUILD_DIR  Build dir relative to repo root (default: build-tidy)
#   CMAKE_TOOLCHAIN_FILE    Used by --configure (default: cmake/toolchains/tohtori-gcc11-openmpi.cmake)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${OPENPFC_TIDY_BUILD_DIR:-build-tidy}"
TOOLCHAIN="${CMAKE_TOOLCHAIN_FILE:-${ROOT}/cmake/toolchains/tohtori-gcc11-openmpi.cmake}"
CONFIGURE=0
FAIL_FAST=0
FILE_FILTER=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run clang-tidy on all *.cpp under src/, apps/, and tests/ (same file set as CI).
Failures follow .clang-tidy (WarningsAsErrors) unless --fail-fast is set.

Options:
  -c, --configure   Run CMake configure in BUILD_DIR (Ninja + tohtori toolchain by default)
      --build-dir=D Set build directory (default: build-tidy or \$OPENPFC_TIDY_BUILD_DIR)
      --fail-fast     Treat every tidy diagnostic as an error (--warnings-as-errors=*)
                      and stop after the first .cpp on which clang-tidy exits non-zero
                      (for iterative fix → commit → re-run workflows)
      --file=PATH     Run only one .cpp relative to repo root (e.g. src/openpfc/foo.cpp);
                      must exist. Combines with --fail-fast for quick checks after a fix.
  -h, --help        Show this help

Examples:
  $(basename "$0") --configure
  $(basename "$0") --build-dir=build-tidy
  $(basename "$0") --fail-fast
  $(basename "$0") --file=src/openpfc/kernel/profiling/session.cpp
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c | --configure) CONFIGURE=1 ;;
    --build-dir=*)
      BUILD_DIR="${1#*=}"
      ;;
    --fail-fast) FAIL_FAST=1 ;;
    --file=*)
      FILE_FILTER="${1#*=}"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

BUILD_ABS="${ROOT}/${BUILD_DIR}"

if [[ "$CONFIGURE" -eq 1 ]]; then
  echo "Configuring ${BUILD_ABS} (toolchain: ${TOOLCHAIN})"
  cmake -S "$ROOT" -B "$BUILD_ABS" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DOpenPFC_BUILD_TESTS=ON \
    -DOpenPFC_BUILD_EXAMPLES=ON \
    -DOpenPFC_BUILD_APPS=ON \
    -DOpenPFC_BUILD_DOCUMENTATION=OFF \
    -DOpenPFC_RUN_MPI_SUITES=OFF
  echo "Configure finished. Run: $(basename "$0")   (without --configure) to run clang-tidy."
  exit 0
fi

if [[ ! -f "${BUILD_ABS}/compile_commands.json" ]]; then
  echo "Missing ${BUILD_ABS}/compile_commands.json" >&2
  echo "Run: $0 --configure   (or configure CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)" >&2
  exit 1
fi

if ! command -v clang-tidy >/dev/null 2>&1; then
  echo "clang-tidy not found in PATH" >&2
  exit 1
fi

if [[ -n "$FILE_FILTER" ]]; then
  if [[ "$FILE_FILTER" != *.cpp ]]; then
    echo "--file must end with .cpp: $FILE_FILTER" >&2
    exit 2
  fi
  if [[ ! -f "${ROOT}/${FILE_FILTER}" ]]; then
    echo "No such file: ${ROOT}/${FILE_FILTER}" >&2
    exit 2
  fi
fi

echo "Using compile_commands: ${BUILD_ABS}/compile_commands.json"
if [[ "$FAIL_FAST" -eq 1 ]]; then
  echo "Fail-fast: --warnings-as-errors=* enabled; will exit on the first failing .cpp."
fi

# Source set matched to GitHub Actions CI "Run clang-tidy" (CPU-only: skip TUs that
# use #error or HIP/CUDA headers when GPU toolchains are not configured).
list_cpp_for_tidy() {
  cd "$ROOT" && find src apps tests -name '*.cpp' \
    ! -name 'test_tungsten_cuda_vtk.cpp' \
    ! -name 'test_tungsten_hip_vtk.cpp' \
    ! -name 'test_tungsten_cpu_vs_cuda.cpp' \
    ! -name 'test_tungsten_cpu_vs_hip.cpp' \
    ! -name 'test_diffop.cpp' \
    ! -name 'test_sparsevector_cuda.cpp' \
    ! -path 'apps/tungsten/src/cuda/tungsten.cpp' \
    ! -path 'apps/tungsten/src/hip/tungsten.cpp' \
    ! -path 'apps/tungsten/src/verify_gpu_aware_mpi.cpp' \
    -print0
}

list_cpp_for_tidy_stream() {
  if [[ -n "$FILE_FILTER" ]]; then
    printf '%s\0' "$FILE_FILTER"
  else
    list_cpp_for_tidy
  fi
}

TIDY_EXTRA=()
if [[ "$FAIL_FAST" -eq 1 ]]; then
  TIDY_EXTRA+=(--warnings-as-errors='*')
fi

failed=0
while IFS= read -r -d '' file; do
  if ! clang-tidy -p "$BUILD_ABS" --quiet \
    -header-filter='include/openpfc/.*' \
    "${TIDY_EXTRA[@]}" \
    "$file"; then
    failed=1
    echo "$(basename "$0"): clang-tidy failed for: $file" >&2
    if [[ "$FAIL_FAST" -eq 1 ]]; then
      exit 1
    fi
  fi
done < <(cd "$ROOT" && list_cpp_for_tidy_stream)

exit "$failed"
