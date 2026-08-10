#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build and install HeFFTe with the ROCm (HIP) backend on LUMI-G, and generate
# a matching Lmod module so it can be loaded with:
#
#     module load heffte-rocm
#
# The install lands in ~/opt/heffte/<version>-rocm and the module file in
# ~/privatemodules/heffte-rocm/<version>.lua (~/privatemodules is already on
# this user's MODULEPATH). Re-run with a different --version (or HEFFTE_VERSION)
# to build additional versions side by side; each gets its own install prefix
# and modulefile, and the `heffte-rocm` default is updated to the newest build.
#
# This mirrors the verified recipe in docs/hpc/INSTALL.LUMI.md (§3). It is a
# LUMI-G-specific counterpart to scripts/build_tohtori.sh (CUDA) and the CPU
# scripts/install-heffte-ci.sh.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags or the matching environment variables)
# ---------------------------------------------------------------------------
HEFFTE_VERSION="${HEFFTE_VERSION:-2.4.1}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ROCM_ARCH="${ROCM_ARCH:-gfx90a}"
JOBS="${JOBS:-16}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/opt/heffte}"
MODULE_ROOT="${MODULE_ROOT:-$HOME/privatemodules/heffte-rocm}"
# Build tree + sources: prefer the fast Lustre flash area, fall back to /tmp.
WORK_ROOT="${WORK_ROOT:-/flash/project_462001519/$USER/heffte-build}"
LUMI_STACK="${LUMI_STACK:-LUMI/25.09}"
# Cray MPICH include dir for HIP translation units (mpi.h under hipcc). The
# default below matches cpeGNU on LUMI/25.09; rediscover after a PE upgrade with
#   CC -E -Wp,-v -xc++ /dev/null 2>&1 | grep mpich
MPI_INC="${MPI_INC:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include}"
GPU_AWARE_MPI="${GPU_AWARE_MPI:-ON}"
KEEP_BUILD="${KEEP_BUILD:-0}"

usage() {
  cat <<EOF
Usage: $0 [options]

Build HeFFTe with the ROCm backend on LUMI-G and install a 'heffte-rocm' module.

Options:
  --version=X.Y.Z     HeFFTe version to build (default: ${HEFFTE_VERSION})
  --build-type=TYPE   CMake build type (default: ${BUILD_TYPE})
  --arch=GFX          CMAKE_HIP_ARCHITECTURES (default: ${ROCM_ARCH})
  --jobs=N, -j N      Parallel build jobs (default: ${JOBS})
  --install-root=DIR  Install prefixes go to DIR/<version>-rocm
                      (default: ${INSTALL_ROOT})
  --module-root=DIR   Modulefiles go to DIR/<version>.lua
                      (default: ${MODULE_ROOT})
  --work-root=DIR     Sources + build trees under DIR (default: ${WORK_ROOT})
  --no-gpu-aware-mpi  Disable Heffte_ENABLE_GPU_AWARE_MPI (default: ON)
  --keep-build        Keep the build tree after install (default: remove)
  -h, --help          Show this help

Environment variables mirror the CLI: HEFFTE_VERSION, BUILD_TYPE, ROCM_ARCH,
JOBS, INSTALL_ROOT, MODULE_ROOT, WORK_ROOT, LUMI_STACK, MPI_INC, GPU_AWARE_MPI,
KEEP_BUILD.

Examples:
  $0                          # build the default version
  $0 --version=2.4.1 -j 32
  HEFFTE_VERSION=2.5.0 $0     # build a new version side by side
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version=*) HEFFTE_VERSION="${1#*=}" ;;
    --build-type=*) BUILD_TYPE="${1#*=}" ;;
    --arch=*) ROCM_ARCH="${1#*=}" ;;
    --jobs=*|-j) shift; JOBS="${1:-}" ;;
    --jobs=*) JOBS="${1#*=}" ;;
    --install-root=*) INSTALL_ROOT="${1#*=}" ;;
    --module-root=*) MODULE_ROOT="${1#*=}" ;;
    --work-root=*) WORK_ROOT="${1#*=}" ;;
    --no-gpu-aware-mpi) GPU_AWARE_MPI="OFF" ;;
    --keep-build) KEEP_BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
  shift
done

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"

INSTALL_PREFIX="${INSTALL_ROOT}/${HEFFTE_VERSION}-rocm"
MODULE_FILE="${MODULE_ROOT}/${HEFFTE_VERSION}.lua"
SRC_DIR="${WORK_ROOT}/src/heffte-${HEFFTE_VERSION}"
BUILD_DIR="${WORK_ROOT}/build/heffte-${HEFFTE_VERSION}-rocm"
ARCHIVE="${WORK_ROOT}/src/v${HEFFTE_VERSION}.tar.gz"

echo "HeFFTe ROCm build"
echo "  version:      ${HEFFTE_VERSION}"
echo "  build type:   ${BUILD_TYPE}"
echo "  HIP arch:     ${ROCM_ARCH}"
echo "  jobs:         ${JOBS}"
echo "  install:      ${INSTALL_PREFIX}"
echo "  modulefile:   ${MODULE_FILE}"
echo "  work root:    ${WORK_ROOT}"
echo "  GPU-aware MPI:${GPU_AWARE_MPI}"

# ---------------------------------------------------------------------------
# 1. Module environment (LUMI-G toolchain + ROCm)
# ---------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
  for init_file in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash \
                   /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [[ -f "${init_file}" ]]; then
      # shellcheck source=/dev/null
      source "${init_file}"
      break
    fi
  done
fi
command -v module >/dev/null 2>&1 || die "Lmod 'module' command not found"

module --force purge
module load "${LUMI_STACK}" partition/G cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

command -v hipcc >/dev/null 2>&1 || die "hipcc not found after loading ROCm (partition/G)"
[[ -f "${MPI_INC}/mpi.h" ]] || die "Cray MPICH mpi.h not found at ${MPI_INC} (set MPI_INC)"

# ROCm CMake packages (hip, rocfft, ...) come from the module's CMAKE_PREFIX_PATH.
export CMAKE_PREFIX_PATH="${EBROOTROCM:-}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# ---------------------------------------------------------------------------
# 2. Fetch and unpack the HeFFTe source
# ---------------------------------------------------------------------------
mkdir -p "${WORK_ROOT}/src" "${WORK_ROOT}/build"
if [[ ! -d "${SRC_DIR}" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Downloading HeFFTe v${HEFFTE_VERSION} ..."
    url="https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${HEFFTE_VERSION}.tar.gz"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "${ARCHIVE}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
      wget -q -O "${ARCHIVE}" "${url}"
    else
      die "neither curl nor wget available to download HeFFTe"
    fi
  fi
  tar xf "${ARCHIVE}" -C "${WORK_ROOT}/src"
fi
[[ -f "${SRC_DIR}/CMakeLists.txt" ]] || die "HeFFTe source not found at ${SRC_DIR}"

# ---------------------------------------------------------------------------
# 3. Configure, build, install
# ---------------------------------------------------------------------------
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DHeffte_ENABLE_FFTW=ON \
  -DHeffte_ENABLE_ROCM=ON \
  -DHeffte_ENABLE_CUDA=OFF \
  -DHeffte_ENABLE_GPU_AWARE_MPI="${GPU_AWARE_MPI}" \
  -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCH}" \
  -DCMAKE_HIP_FLAGS="-I${MPI_INC}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"
cmake --install "${BUILD_DIR}"

# Locate the installed HeffteConfig.cmake to sanity-check the layout.
HEFFTE_CMAKE_DIR=""
for cand in "${INSTALL_PREFIX}/lib64/cmake/Heffte" "${INSTALL_PREFIX}/lib/cmake/Heffte"; do
  if [[ -f "${cand}/HeffteConfig.cmake" ]]; then
    HEFFTE_CMAKE_DIR="${cand}"
    break
  fi
done
[[ -n "${HEFFTE_CMAKE_DIR}" ]] || die "HeffteConfig.cmake not found under ${INSTALL_PREFIX}"

# ---------------------------------------------------------------------------
# 4. Generate the Lmod modulefile
# ---------------------------------------------------------------------------
mkdir -p "${MODULE_ROOT}"
cat > "${MODULE_FILE}" <<LUA
-- Name: heffte-rocm
-- Version: ${HEFFTE_VERSION}
-- Generated by scripts/build_heffte_rocm.sh on $(date +%Y-%m-%d)

help([[Load HeFFTe ${HEFFTE_VERSION} built with the ROCm (HIP) backend for LUMI-G.
Sets CMAKE_PREFIX_PATH, library and include paths, and HeFFTe_DIR/ROOT so that
find_package(Heffte) picks up the ROCm-enabled install.]])

local module_base = "${INSTALL_PREFIX}"

-- HeFFTe runtime + build paths
prepend_path("CMAKE_PREFIX_PATH", module_base)
prepend_path("PATH", pathJoin(module_base, "bin"))
prepend_path("LD_LIBRARY_PATH", pathJoin(module_base, "lib64"))
prepend_path("LD_LIBRARY_PATH", pathJoin(module_base, "lib"))
prepend_path("CPATH", pathJoin(module_base, "include"))
prepend_path("INCLUDE", pathJoin(module_base, "include"))
prepend_path("PKG_CONFIG_PATH", pathJoin(module_base, "lib64", "pkgconfig"))
prepend_path("PKG_CONFIG_PATH", pathJoin(module_base, "lib", "pkgconfig"))
setenv("HEFFTE_DIR", pathJoin(module_base, "lib64", "cmake", "Heffte"))
setenv("HEFFTE_ROOT", module_base)

whatis("Name: heffte-rocm")
whatis("Version: ${HEFFTE_VERSION}")
whatis("Description: HeFFTe ${HEFFTE_VERSION} ROCm/HIP backend (LUMI-G, ${ROCM_ARCH})")
LUA

# Make the bare `heffte-rocm` name resolve to the newest version we built.
cat > "${MODULE_ROOT}/default" <<EOF
${HEFFTE_VERSION}
EOF

if [[ "${KEEP_BUILD}" != "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

echo
echo "================================================================"
echo "HeFFTe ROCm build: PASS"
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "  HeffteConfig:   ${HEFFTE_CMAKE_DIR}/HeffteConfig.cmake"
echo "  Modulefile:     ${MODULE_FILE}"
echo
echo "Load it with:"
echo "  module use \$HOME/privatemodules   # if not already on MODULEPATH"
echo "  module load heffte-rocm"
echo "================================================================"
