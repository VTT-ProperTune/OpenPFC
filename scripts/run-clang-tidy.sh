#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run clang-tidy on OpenPFC sources the same way as CI (.clang-tidy, header filter).
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

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run clang-tidy on all *.cpp under src/, apps/, and tests/ (same as CI).

Options:
  -c, --configure   Run CMake configure in BUILD_DIR (Ninja + tohtori toolchain by default)
      --build-dir=D Set build directory (default: build-tidy or \$OPENPFC_TIDY_BUILD_DIR)
  -h, --help        Show this help

Examples:
  $(basename "$0") --configure
  $(basename "$0") --build-dir=build-tidy
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c | --configure) CONFIGURE=1 ;;
    --build-dir=*)
      BUILD_DIR="${1#*=}"
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

echo "Using compile_commands: ${BUILD_ABS}/compile_commands.json"
failed=0
while IFS= read -r -d '' file; do
  if ! clang-tidy -p "$BUILD_ABS" --quiet \
    -header-filter='include/openpfc/.*' \
    "$file"; then
    failed=1
  fi
done < <(cd "$ROOT" && find src apps tests -name '*.cpp' -print0)

exit "$failed"
