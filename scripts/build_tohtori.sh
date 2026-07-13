#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Invoked as `sh ./scripts/build_tohtori.sh`? Re-exec with bash (Lmod + bashisms).
if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" ${1+"$@"}
fi
#
# Build HeFFTe (CPU) and OpenPFC on VTT **tohtori** using the same recipe as
# INSTALL.md §1, §3, and §5, plus cmake/toolchains/tohtori-gcc11-openmpi.cmake.
#
# Usage (from anywhere):
#   sh ./scripts/build_tohtori.sh
#   ./scripts/build_tohtori.sh
#   OPENPFC_BUILD_DIR=build/my-tohtori ./scripts/build_tohtori.sh
#
# Options:
#   --skip-heffte     Do not download/build HeFFTe (use existing HEFFTE_PREFIX).
#   --heffte-only     Build and install HeFFTe only.
#   --clean-heffte    Remove HeFFTe build dir before configuring HeFFTe.
#   --clean-openpfc   Remove OpenPFC build dir before configuring OpenPFC.
#   --build-openmpi   Build/install OpenMPI under OPENMPI_PREFIX (Slurm PMI via --with-slurm),
#                     then set OPENMPI_ROOT for OpenPFC CMake (see toolchain). Skips module openmpi/5.0.10.
#   --openmpi-only    Only configure/build/install OpenMPI (--build-openmpi implied).
#   --clean-openmpi   Remove OpenMPI build dir before configuring OpenMPI.
#   --build-ucx       Build/install UCX to UCX_PREFIX, set UCX_HOME for OpenMPI (--with-ucx).
#   --ucx-only        Only build/install UCX.
#   --clean-ucx       Remove UCX VPATH build dir before configuring UCX.
#   --help            Show this help.
#
# Environment (optional):
#   HEFFTE_PREFIX     Default: $HOME/opt/heffte/2.4.1-cpu
#   HEFFTE_VER        Default: 2.4.1
#   OPENPFC_BUILD_DIR Default: <repo>/builds/tohtori-release
#   NINJA_JOBS        Default: $(nproc) for cmake --build
#   OPENMPI_VER       Default: 5.0.10 (used with --build-openmpi)
#   OPENMPI_PREFIX    Default: $HOME/opt/openmpi/$OPENMPI_VER
#   OPENMPI_SRC_ROOT  Default: $HOME/src
#   OPENMPI_BUILD_DIR Default: $HOME/opt/openmpi/build-$OPENMPI_VER
#   UCX_VER           Default: 1.20.0 (used with --build-ucx)
#   UCX_PREFIX        Default: $HOME/opt/ucx/$UCX_VER
#   UCX_SRC_ROOT      Default: $HOME/src
#   UCX_BUILD_DIR     Default: $HOME/opt/ucx/build-$UCX_VER
#   UCX_HOME          If set (and not using --build-ucx), OpenMPI gets --with-ucx=UCX_HOME
#
# This script loads the site **openmpi/5.0.10** module and its compiler dependency. For a
# custom Open MPI build, OPENPFC_GCC_MODULE selects the compiler module (default gcc/11.2.0).
# For long builds (especially OpenMPI), use **scripts/submit_build_tohtori_sbatch.sh**.
# If CMake fails finding FFTW, load your site's FFTW development module (e.g.
# after editing this script), or install fftw-devel (EL8), then re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HEFFTE_VER="${HEFFTE_VER:-2.4.1}"
OPENPFC_GCC_MODULE="${OPENPFC_GCC_MODULE:-gcc/11.2.0}"
HEFFTE_PREFIX="${HEFFTE_PREFIX:-${HOME}/opt/heffte/${HEFFTE_VER}-cpu}"
HEFFTE_SRC_ROOT="${HEFFTE_SRC_ROOT:-${HOME}/src}"
HEFFTE_BUILD_DIR="${HEFFTE_BUILD_DIR:-${HOME}/opt/heffte/build-cpu-${HEFFTE_VER}}"
OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${REPO_ROOT}/builds/tohtori-release}"
NINJA_JOBS="${NINJA_JOBS:-$(nproc)}"

OPENMPI_VER="${OPENMPI_VER:-5.0.10}"
OPENMPI_PREFIX="${OPENMPI_PREFIX:-${HOME}/opt/openmpi/${OPENMPI_VER}}"
OPENMPI_SRC_ROOT="${OPENMPI_SRC_ROOT:-${HOME}/src}"
OPENMPI_BUILD_DIR="${OPENMPI_BUILD_DIR:-${HOME}/opt/openmpi/build-${OPENMPI_VER}}"

UCX_VER="${UCX_VER:-1.20.0}"
UCX_PREFIX="${UCX_PREFIX:-${HOME}/opt/ucx/${UCX_VER}}"
UCX_SRC_ROOT="${UCX_SRC_ROOT:-${HOME}/src}"
UCX_BUILD_DIR="${UCX_BUILD_DIR:-${HOME}/opt/ucx/build-${UCX_VER}}"

# If the user exported OPENMPI_ROOT before this script, keep it when not passing --build-openmpi
# (otherwise we would unset it and CMake would fall back to site Open MPI in the toolchain).
_PRESERVE_OPENMPI_ROOT="${OPENMPI_ROOT-}"

SKIP_HEFFTE=0
HEFFTE_ONLY=0
CLEAN_HEFFTE=0
CLEAN_OPENPFC=0
BUILD_OPENMPI=0
OPENMPI_ONLY=0
CLEAN_OPENMPI=0
BUILD_UCX=0
UCX_ONLY=0
CLEAN_UCX=0

usage() {
  cat <<'EOF'
Usage: build_tohtori.sh [options]

Build HeFFTe 2.4.1 (CPU) to $HOME/opt/heffte/<ver>-cpu and OpenPFC with
cmake/toolchains/tohtori-gcc11-openmpi.cmake (see INSTALL.md).

Options:
  --skip-heffte      Skip HeFFTe; require existing install at HEFFTE_PREFIX.
  --heffte-only      Only build/install HeFFTe.
  --clean-heffte     Remove HeFFTe build directory before configuring HeFFTe.
  --clean-openpfc    Remove OpenPFC build directory before configuring.
  --build-openmpi    Build OpenMPI with Slurm support; use OPENMPI_PREFIX / OPENMPI_VER.
  --openmpi-only     Only build/install OpenMPI (implies --build-openmpi).
  --clean-openmpi    Remove OpenMPI VPATH build dir before configuring OpenMPI.
  --build-ucx        Build UCX from source; sets UCX_HOME for OpenMPI when combined.
  --ucx-only         Only build/install UCX.
  --clean-ucx        Remove UCX VPATH build dir before configuring UCX.
  -h, --help         Show this help.

Environment:
  HEFFTE_PREFIX      (default: $HOME/opt/heffte/2.4.1-cpu)
  HEFFTE_VER         (default: 2.4.1)
  OPENPFC_GCC_MODULE compiler module for custom MPI builds (default: gcc/11.2.0)
  OPENPFC_BUILD_DIR  (default: <repo>/builds/tohtori-release)
  NINJA_JOBS         parallel build jobs (default: nproc)
  OPENMPI_VER        (default: 5.0.10)
  OPENMPI_PREFIX     (default: $HOME/opt/openmpi/$OPENMPI_VER)
  OPENMPI_SRC_ROOT   (default: $HOME/src)
  OPENMPI_BUILD_DIR  (default: $HOME/opt/openmpi/build-$OPENMPI_VER)
  UCX_VER            (default: 1.20.0)
  UCX_PREFIX         (default: $HOME/opt/ucx/$UCX_VER)
  UCX_SRC_ROOT       (default: $HOME/src)
  UCX_BUILD_DIR      (default: $HOME/opt/ucx/build-$UCX_VER)
  UCX_HOME           optional; adds --with-ucx to OpenMPI (if not using --build-ucx)

UCX configure uses --without-go (skips Go bindings; they often break in VPATH builds).

Requires: Lmod (tohtori), FFTW dev libs, CMake 3.15+, Ninja optional.
The site **openmpi/5.0.10** module and its compiler dependency are loaded automatically.
Custom Open MPI builds use **OPENPFC_GCC_MODULE** (default **gcc/11.2.0**).

OpenMPI from source needs Slurm dev headers/libs on the build host for **--with-slurm**
(often true on login nodes; if configure fails, build inside **sbatch** on a cluster image that has them).

UCX needs RDMA / verbs headers on the build host (e.g. **rdma-core** / **libibverbs-devel** on EL8).
If **configure** fails, install those packages or build inside **sbatch** on a compute image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-heffte) SKIP_HEFFTE=1 ;;
    --heffte-only) HEFFTE_ONLY=1 ;;
    --clean-heffte) CLEAN_HEFFTE=1 ;;
    --clean-openpfc) CLEAN_OPENPFC=1 ;;
    --build-openmpi) BUILD_OPENMPI=1 ;;
    --openmpi-only) OPENMPI_ONLY=1; BUILD_OPENMPI=1 ;;
    --clean-openmpi) CLEAN_OPENMPI=1 ;;
    --build-ucx) BUILD_UCX=1 ;;
    --ucx-only) UCX_ONLY=1; BUILD_UCX=1 ;;
    --clean-ucx) CLEAN_UCX=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ "${OPENMPI_ONLY}" -eq 1 ]]; then
  HEFFTE_ONLY=0
  SKIP_HEFFTE=1
fi

if [[ "${UCX_ONLY}" -eq 1 ]]; then
  HEFFTE_ONLY=0
  SKIP_HEFFTE=1
  OPENMPI_ONLY=0
  BUILD_OPENMPI=0
fi

log() { echo "[build_tohtori] $*"; }

ucx_installed() {
  [[ -f "${UCX_PREFIX}/lib/libucp.so" || -f "${UCX_PREFIX}/lib64/libucp.so" ]]
}

build_ucx() {
  local ver src tarball url cfg_release
  ver="${UCX_VER}"
  src="${UCX_SRC_ROOT}/ucx-${ver}"
  tarball="${UCX_SRC_ROOT}/ucx-${ver}.tar.gz"
  url="https://github.com/openucx/ucx/releases/download/v${ver}/ucx-${ver}.tar.gz"

  mkdir -p "${UCX_SRC_ROOT}"

  if [[ ! -f "${src}/configure" ]]; then
    log "Fetching UCX ${ver} (${url})"
    if command -v wget >/dev/null 2>&1; then
      wget -q -O "${tarball}" "${url}" || {
        rm -f "${tarball}"
        echo "ERROR: wget failed for UCX tarball." >&2
        exit 1
      }
    elif command -v curl >/dev/null 2>&1; then
      curl -fsSL -L -o "${tarball}" "${url}" || {
        rm -f "${tarball}"
        echo "ERROR: curl failed for UCX tarball." >&2
        exit 1
      }
    else
      echo "ERROR: need wget or curl to download UCX." >&2
      exit 1
    fi
    tar xf "${tarball}" -C "${UCX_SRC_ROOT}"
  fi

  if [[ "${CLEAN_UCX}" -eq 1 ]]; then
    log "Removing UCX build dir ${UCX_BUILD_DIR}"
    rm -rf "${UCX_BUILD_DIR}"
  fi

  mkdir -p "${UCX_BUILD_DIR}"
  cfg_release="${src}/contrib/configure-release"

  log "Configuring UCX (VPATH) ${UCX_BUILD_DIR} -> ${UCX_PREFIX}"
  (
    cd "${UCX_BUILD_DIR}"
    if [[ -x "${cfg_release}" ]]; then
      "${cfg_release}" \
        --prefix="${UCX_PREFIX}" \
        --without-cuda \
        --without-rocm \
        --without-go
    else
      "${src}/configure" \
        --prefix="${UCX_PREFIX}" \
        --enable-optimizations \
        --without-cuda \
        --without-rocm \
        --without-go
    fi
  )

  log "Building and installing UCX (${NINJA_JOBS} parallel jobs)..."
  make -C "${UCX_BUILD_DIR}" -j"${NINJA_JOBS}"
  make -C "${UCX_BUILD_DIR}" install
  log "UCX installed at ${UCX_PREFIX}"
}

openmpi_installed() {
  [[ -x "${OPENMPI_PREFIX}/bin/mpicc" ]] || return 1
  [[ -f "${OPENMPI_PREFIX}/lib/libmpi.so" || -f "${OPENMPI_PREFIX}/lib64/libmpi.so" ]]
}

# Open MPI 5.x may link libbfd (pretty-print stacktrace / toolchain). Compute nodes often lack
# binutils-devel; add -L where libbfd.so already exists (EL: /usr/lib64, Debian: .../x86_64-linux-gnu).
prepend_libbfd_ldflags_if_present() {
  local d
  for d in /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu; do
    if compgen -G "${d}/libbfd.so*" >/dev/null 2>&1; then
      export LDFLAGS="-L${d} ${LDFLAGS:-}"
      log "Prepending LDFLAGS with -L${d} (libbfd found for Open MPI link)"
      return 0
    fi
  done
  log "No system libbfd.so found under /usr/lib*; if Open MPI fails with \"cannot find -lbfd\", install binutils-devel (EL) or libbfd-dev (Debian/Ubuntu), then re-run."
}

build_openmpi() {
  local ver majmin src tarball url
  ver="${OPENMPI_VER}"
  majmin="${ver%.*}"
  src="${OPENMPI_SRC_ROOT}/openmpi-${ver}"
  tarball="${OPENMPI_SRC_ROOT}/openmpi-${ver}.tar.bz2"
  # Note: path is .../v5.0/openmpi-X.Y.Z.tar.bz2 (no "downloads/" segment).
  url="https://download.open-mpi.org/release/open-mpi/v${majmin}/openmpi-${ver}.tar.bz2"

  mkdir -p "${OPENMPI_SRC_ROOT}"

  if [[ ! -f "${src}/configure" ]]; then
    log "Fetching OpenMPI ${ver} source (${url})"
    if command -v wget >/dev/null 2>&1; then
      wget -q --user-agent="Mozilla/5.0" -O "${tarball}" "${url}" || {
        rm -f "${tarball}"
        echo "ERROR: wget failed for OpenMPI tarball." >&2
        exit 1
      }
    elif command -v curl >/dev/null 2>&1; then
      curl -fsSL -A "Mozilla/5.0" -o "${tarball}" "${url}" || {
        rm -f "${tarball}"
        echo "ERROR: curl failed for OpenMPI tarball." >&2
        exit 1
      }
    else
      echo "ERROR: need wget or curl to download OpenMPI." >&2
      exit 1
    fi
    tar xf "${tarball}" -C "${OPENMPI_SRC_ROOT}"
  fi

  if [[ "${CLEAN_OPENMPI}" -eq 1 ]]; then
    log "Removing OpenMPI build dir ${OPENMPI_BUILD_DIR}"
    rm -rf "${OPENMPI_BUILD_DIR}"
  fi

  mkdir -p "${OPENMPI_BUILD_DIR}"

  log "Configuring OpenMPI (VPATH) ${OPENMPI_BUILD_DIR} -> prefix ${OPENMPI_PREFIX}"
  prepend_libbfd_ldflags_if_present
  # --with-slurm: PMI/PMIx + srun compatibility on Slurm clusters (scalability sbatch jobs).
  # --disable-pretty-print-stacktrace: reduces libbfd coupling on images without binutils-devel.
  local -a ucx_args=()
  if [[ -n "${UCX_HOME:-}" ]]; then
    ucx_args+=(--with-ucx="${UCX_HOME}")
    if [[ -e "${UCX_HOME}/lib64/libucp.so" ]]; then
      ucx_args+=(--with-ucx-libdir="${UCX_HOME}/lib64")
    else
      ucx_args+=(--with-ucx-libdir="${UCX_HOME}/lib")
    fi
  fi
  (
    cd "${OPENMPI_BUILD_DIR}"
    "${src}/configure" \
      --prefix="${OPENMPI_PREFIX}" \
      --with-slurm \
      --enable-mpirun-prefix-by-default \
      --enable-mpi-fortran=all \
      --disable-oshmem \
      --disable-pretty-print-stacktrace \
      "${ucx_args[@]}"
  )

  log "Building and installing OpenMPI (${NINJA_JOBS} parallel jobs)..."
  make -C "${OPENMPI_BUILD_DIR}" -j"${NINJA_JOBS}"
  make -C "${OPENMPI_BUILD_DIR}" install
  log "OpenMPI installed at ${OPENMPI_PREFIX}"
}

# Lmod is often a shell function: not available in plain non-login sh until init is sourced.
init_lmod() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  local f
  for f in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash \
           /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do
    if [[ -f "${f}" ]]; then
      # shellcheck source=/dev/null
      source "${f}"
      break
    fi
  done
}

init_lmod
if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: Lmod command 'module' not found after sourcing common init scripts." >&2
  echo "       Use a login shell or: source /etc/profile.d/lmod.sh  (path may differ)." >&2
  exit 1
fi

# The site MPI module owns its compiler dependency. Capture CC/CXX only after that dependency
# is loaded so Lmod cannot silently swap the compiler underneath the configured build.
if [[ "${BUILD_OPENMPI}" -eq 1 || -n "${_PRESERVE_OPENMPI_ROOT}" ]]; then
  log "Loading compiler module for custom Open MPI: ${OPENPFC_GCC_MODULE}"
  module load "${OPENPFC_GCC_MODULE}"
else
  log "Loading site MPI stack: openmpi/5.0.10 (including its compiler dependency)"
  module load openmpi/5.0.10
fi

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export OPENPFC_GCC_ROOT="$(cd "$(dirname "${CXX}")/.." && pwd)"
if [[ -z "${CC}" || -z "${CXX}" ]]; then
  echo "ERROR: gcc/g++ not on PATH after module load." >&2
  exit 1
fi

if [[ "${BUILD_UCX}" -eq 1 ]]; then
  _build_tohtori_ucx_skipped=0
  if [[ "${CLEAN_UCX}" -eq 1 ]] || ! ucx_installed; then
    build_ucx
  else
    log "UCX already present at ${UCX_PREFIX} (skipping build; use --clean-ucx to force rebuild)"
    _build_tohtori_ucx_skipped=1
  fi
  export UCX_HOME="${UCX_PREFIX}"
  export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${UCX_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
  export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  if [[ "${_build_tohtori_ucx_skipped}" -eq 1 ]]; then
    _uxi="${UCX_PREFIX}/bin/ucx_info"
    if [[ -x "${_uxi}" ]]; then
      log "UCX quick check (ucx_info -v):"
      "${_uxi}" -v 2>&1 | head -8 || log "WARNING: ucx_info failed; reinstall with --clean-ucx --build-ucx"
    else
      log "WARNING: missing ${_uxi} but libraries exist — install may be incomplete. Use --clean-ucx --build-ucx."
    fi
  fi
  unset _build_tohtori_ucx_skipped _uxi
fi

if [[ "${UCX_ONLY}" -eq 1 ]]; then
  log "Done (--ucx-only)."
  exit 0
fi

if [[ "${BUILD_OPENMPI}" -eq 1 ]]; then
  if [[ "${CLEAN_OPENMPI}" -eq 1 ]] || ! openmpi_installed; then
    build_openmpi
  else
    log "OpenMPI already present at ${OPENMPI_PREFIX} (use --clean-openmpi to rebuild)"
  fi
  export OPENMPI_ROOT="${OPENMPI_PREFIX}"
  export PATH="${OPENMPI_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${OPENMPI_PREFIX}/lib64:${OPENMPI_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  log "Using OPENMPI_ROOT=${OPENMPI_ROOT} (cmake/toolchains/tohtori-gcc11-openmpi.cmake)"
else
  if [[ -n "${_PRESERVE_OPENMPI_ROOT}" ]]; then
    export OPENMPI_ROOT="${_PRESERVE_OPENMPI_ROOT}"
    export PATH="${OPENMPI_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OPENMPI_ROOT}/lib64:${OPENMPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    log "Using pre-set OPENMPI_ROOT=${OPENMPI_ROOT} (skip site openmpi module)"
  else
    unset OPENMPI_ROOT
    log "Using site module stack already loaded above: openmpi/5.0.10"
  fi
fi
unset _PRESERVE_OPENMPI_ROOT

log "Using CC=${CC} CXX=${CXX}"
"${CC}" --version | head -1
command -v mpicc >/dev/null && mpicc --version | head -1 || log "WARNING: mpicc not on PATH"

if [[ "${OPENMPI_ONLY}" -eq 1 ]]; then
  log "Done (--openmpi-only)."
  exit 0
fi

heffte_configured() {
  [[ -f "${HEFFTE_PREFIX}/lib64/cmake/Heffte/HeffteConfig.cmake" ]] ||
    [[ -f "${HEFFTE_PREFIX}/lib/cmake/Heffte/HeffteConfig.cmake" ]]
}

build_heffte() {
  local tarball ver src
  ver="${HEFFTE_VER}"
  src="${HEFFTE_SRC_ROOT}/heffte-${ver}"
  tarball="${HEFFTE_SRC_ROOT}/v${ver}.tar.gz"

  mkdir -p "${HEFFTE_SRC_ROOT}" "${HOME}/opt/heffte"

  if [[ ! -d "${src}/CMakeLists.txt" ]]; then
    log "Fetching HeFFTe ${ver} source to ${HEFFTE_SRC_ROOT}"
    mkdir -p "${HEFFTE_SRC_ROOT}"
    if command -v wget >/dev/null 2>&1; then
      wget -q -O "${tarball}" \
        "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${ver}.tar.gz"
    elif command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "${tarball}" \
        "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${ver}.tar.gz"
    else
      echo "ERROR: need wget or curl to download HeFFTe." >&2
      exit 1
    fi
    tar xf "${tarball}" -C "${HEFFTE_SRC_ROOT}"
  fi

  if [[ "${CLEAN_HEFFTE}" -eq 1 ]]; then
    log "Removing HeFFTe build dir ${HEFFTE_BUILD_DIR}"
    rm -rf "${HEFFTE_BUILD_DIR}"
  fi

  local gen="Unix Makefiles"
  if command -v ninja >/dev/null 2>&1; then
    gen="Ninja"
  fi

  log "Configuring HeFFTe (CPU, FFTW) -> ${HEFFTE_PREFIX} (generator: ${gen})"
  cmake -S "${src}" -B "${HEFFTE_BUILD_DIR}" -G "${gen}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${HEFFTE_PREFIX}" \
    -DHeffte_ENABLE_FFTW=ON \
    -DHeffte_ENABLE_CUDA=OFF

  log "Building and installing HeFFTe..."
  cmake --build "${HEFFTE_BUILD_DIR}" -j"${NINJA_JOBS}"
  cmake --install "${HEFFTE_BUILD_DIR}"
  log "HeFFTe installed at ${HEFFTE_PREFIX}"
}

if [[ "${SKIP_HEFFTE}" -eq 0 ]]; then
  if heffte_configured; then
    log "HeFFTe already present at ${HEFFTE_PREFIX} (use --clean-heffte + rebuild to replace)"
  else
    build_heffte
  fi
else
  if ! heffte_configured; then
    echo "ERROR: --skip-heffte but HeFFTe not found under ${HEFFTE_PREFIX} (expected HeffteConfig.cmake)." >&2
    exit 1
  fi
  log "Skipping HeFFTe build (--skip-heffte), using ${HEFFTE_PREFIX}"
fi

if [[ "${HEFFTE_ONLY}" -eq 1 ]]; then
  log "Done (--heffte-only)."
  exit 0
fi

TOOLCHAIN="${REPO_ROOT}/cmake/toolchains/tohtori-gcc11-openmpi.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
  echo "ERROR: missing ${TOOLCHAIN}" >&2
  exit 1
fi

if [[ "${CLEAN_OPENPFC}" -eq 1 ]]; then
  log "Removing OpenPFC build dir ${OPENPFC_BUILD_DIR}"
  rm -rf "${OPENPFC_BUILD_DIR}"
fi

gen="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  gen="Ninja"
fi

export CMAKE_PREFIX_PATH="${HEFFTE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

log "Configuring OpenPFC -> ${OPENPFC_BUILD_DIR} (generator: ${gen})"
# CMAKE_PREFIX_PATH includes HEFFTE_PREFIX; toolchain also prepends $HOME/opt/heffte/2.4.1-cpu if present.
openpfc_cmake_mpi=()
if [[ -n "${OPENMPI_ROOT:-}" ]]; then
  # FindMPI caches include/lib paths separately from MPI_*_COMPILER. Unsetting only the compilers
  # leaves MPI_mpi_LIBRARY and MPI_*_LINK_FLAGS pointing at site Open MPI (wrong link line).
  _mpi_cache_vars=(
    MPI_DIR
    MPI_mpi_LIBRARY
    MPI_C_COMPILER
    MPI_CXX_COMPILER
    MPI_C_HEADER_DIR
    MPI_CXX_HEADER_DIR
    MPI_C_LIB_NAMES
    MPI_CXX_LIB_NAMES
    MPI_C_LINK_FLAGS
    MPI_CXX_LINK_FLAGS
    MPI_C_COMPILE_OPTIONS
    MPI_CXX_COMPILE_OPTIONS
    MPI_C_COMPILE_DEFINITIONS
    MPI_CXX_COMPILE_DEFINITIONS
    MPI_C_COMPILER_INCLUDE_DIRS
    MPI_CXX_COMPILER_INCLUDE_DIRS
    MPI_C_ADDITIONAL_INCLUDE_DIRS
    MPI_CXX_ADDITIONAL_INCLUDE_DIRS
    MPI_C_SKIP_MPICXX
    MPI_CXX_SKIP_MPICXX
    MPI_RESULT_CXX_test_mpi_MPICXX
    MPI_RESULT_CXX_test_mpi_normal
    MPI_RESULT_C_test_mpi_normal
  )
  for _v in "${_mpi_cache_vars[@]}"; do
    openpfc_cmake_mpi+=("-U${_v}")
  done
  unset _v _mpi_cache_vars
  openpfc_cmake_mpi+=(
    -DMPI_CXX_COMPILER="${OPENMPI_ROOT}/bin/mpicxx"
    -DMPI_C_COMPILER="${OPENMPI_ROOT}/bin/mpicc"
  )
  log "OpenPFC CMake: clearing FindMPI cache and forcing wrappers under ${OPENMPI_ROOT}/bin"
fi
cmake -S "${REPO_ROOT}" -B "${OPENPFC_BUILD_DIR}" -G "${gen}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  -DOpenPFC_ENABLE_CODE_COVERAGE=OFF \
  -DOpenPFC_BUILD_DOCUMENTATION=OFF \
  -DOpenPFC_ENABLE_HDF5=ON \
  "${openpfc_cmake_mpi[@]}"

if [[ ${#openpfc_cmake_mpi[@]} -gt 0 ]] && [[ -f "${OPENPFC_BUILD_DIR}/build.ninja" ]]; then
  log "Cleaning tungsten targets so they relink against the selected MPI"
  ninja -C "${OPENPFC_BUILD_DIR}" -t clean tungsten 2>/dev/null || true
  ninja -C "${OPENPFC_BUILD_DIR}" -t clean tungsten_scalability 2>/dev/null || true
fi

log "Building OpenPFC (${NINJA_JOBS} jobs)..."
cmake --build "${OPENPFC_BUILD_DIR}" -j"${NINJA_JOBS}"

log "Done. Binaries under ${OPENPFC_BUILD_DIR}/ (e.g. apps/tungsten/tungsten)."
log "For scalability jobs: export TUNGSTEN_BIN=${OPENPFC_BUILD_DIR}/apps/tungsten/tungsten"
