#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build a released OpenPFC tag on Tohtori, install it into a shared prefix,
# and write an Lmod modulefile for it. The build itself goes through the
# tag's own scripts/build.sh.

if [ -z "${BASH_VERSION-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="${REPO_ROOT}"

VERSION=""
BACKEND="cpu"
ROOT="${OPENPFC_MODULE_ROOT:-/task/${USER}}"
JOBS="${JOBS:-32}"
HEFFTE_PREFIX=""
OPENMPI_ROOT=""
GCC_MODULE="gcc/15.2.0"
OPENMPI_MODULE="openmpi/5.0.10"
CUDA_MODULE="cuda/13.1"
# A100, A40 and H100: the cuda and nvidia_h100 partitions.
CUDA_ARCHITECTURES="80;86;90"
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/install_release_module.sh --version=X.Y.Z [options]

Build tag vX.Y.Z on Tohtori, install it under ROOT/apps/openpfc/X.Y.Z-BACKEND
and write ROOT/modulefiles/openpfc/X.Y.Z-BACKEND.lua. Users then run:

  module use ROOT/modulefiles
  module load openpfc/X.Y.Z-BACKEND

Nothing installed or loaded may resolve through $HOME: on Tohtori it is
private to its owner. Run the build on a compute node, e.g.

  srun -p all6 -c 32 --mem=64G -t 2:00:00 \
    scripts/install_release_module.sh --version=0.4.0 --backend=cpu

Options:
  --version=X.Y.Z       Release to install; tag vX.Y.Z must exist locally
  --backend=NAME        cpu (default) or cuda
  --root=PATH           Shared root (default: /task/$USER)
  --repository=PATH     Git repository holding the tag (default: this checkout)
  --heffte-prefix=PATH  HeFFTe 2.4.1 install for this backend
                        cpu default:  ROOT/apps/heffte/2.4.1-cpu-ompi5
                        cuda default: ROOT/apps/heffte/2.4.1-cuda-ompi5
  --openmpi-root=PATH   cuda only: CUDA-aware Open MPI install
                        (default: ROOT/apps/openmpi/5.0.10-cuda)
  --cuda-architectures=LIST
                        cuda only: CMAKE_CUDA_ARCHITECTURES (default: 80;86;90)
  --jobs=N, -j N        Parallel build jobs (default: 32)
  --prepare-only        Export the sources and write the modulefile text
                        under the staging directory; do not build
  -h, --help            Show this help

The install prefix must not exist; an installed release is never replaced.
The staging tree ROOT/build/openpfc-X.Y.Z-BACKEND holds sources and the
build directory and can be removed after a successful install.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version=*) VERSION="${1#*=}" ;;
    --backend=*) BACKEND="${1#*=}" ;;
    --root=*) ROOT="${1#*=}" ;;
    --repository=*) SOURCE_REPO="${1#*=}" ;;
    --heffte-prefix=*) HEFFTE_PREFIX="${1#*=}" ;;
    --openmpi-root=*) OPENMPI_ROOT="${1#*=}" ;;
    --cuda-architectures=*) CUDA_ARCHITECTURES="${1#*=}" ;;
    --jobs=*) JOBS="${1#*=}" ;;
    -j)
      require_value "$1" "${2-}"
      JOBS="$2"
      shift
      ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
  shift
done

VERSION="${VERSION#v}"
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  die "--version=X.Y.Z is required (got '${VERSION}')"
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "invalid --jobs '${JOBS}'"
[[ "${ROOT}" == /* ]] || die "--root must be absolute (got '${ROOT}')"
[[ "${CUDA_ARCHITECTURES}" =~ ^[0-9]+(;[0-9]+)*$ ]] ||
  die "invalid --cuda-architectures '${CUDA_ARCHITECTURES}'"

case "${BACKEND}" in
  cpu)
    HEFFTE_PREFIX="${HEFFTE_PREFIX:-${ROOT}/apps/heffte/2.4.1-cpu-ompi5}"
    ;;
  cuda)
    HEFFTE_PREFIX="${HEFFTE_PREFIX:-${ROOT}/apps/heffte/2.4.1-cuda-ompi5}"
    OPENMPI_ROOT="${OPENMPI_ROOT:-${ROOT}/apps/openmpi/5.0.10-cuda}"
    ;;
  *) die "--backend must be cpu or cuda (got '${BACKEND}')" ;;
esac
[[ "${BACKEND}" == cuda || -z "${OPENMPI_ROOT}" ]] ||
  die "--openmpi-root applies to --backend=cuda only"

TAG="v${VERSION}"
NAME="${VERSION}-${BACKEND}"
PREFIX="${ROOT}/apps/openpfc/${NAME}"
MODULE_DIR="${ROOT}/modulefiles/openpfc"
MODULE_FILE="${MODULE_DIR}/${NAME}.lua"
STAGE="${ROOT}/build/openpfc-${NAME}"

# A path under $HOME is unreadable for everyone else who loads the module.
for path in "${ROOT}" "${HEFFTE_PREFIX}" ${OPENMPI_ROOT:+"${OPENMPI_ROOT}"}; do
  case "$(realpath -m "${path}")/" in
    "$(realpath -m "${HOME}")"/*|/home/*) die "${path} resolves under /home" ;;
  esac
done

REVISION="$(git -C "${SOURCE_REPO}" rev-parse --verify --quiet "${TAG}^{commit}")" ||
  die "tag ${TAG} not found in ${SOURCE_REPO} (git fetch --tags?)"
[[ -d "${HEFFTE_PREFIX}/lib64/cmake/Heffte" || -d "${HEFFTE_PREFIX}/lib/cmake/Heffte" ]] ||
  die "HeFFTe CMake package not found under ${HEFFTE_PREFIX}"
if [[ "${BACKEND}" == cuda ]]; then
  [[ -x "${OPENMPI_ROOT}/bin/mpicc" ]] || die "no mpicc in ${OPENMPI_ROOT}/bin"
fi
[[ ! -e "${PREFIX}" ]] || die "${PREFIX} already exists; an installed release is not replaced"
[[ ! -e "${MODULE_FILE}" ]] || die "${MODULE_FILE} already exists"
[[ ! -e "${STAGE}" ]] || die "staging directory ${STAGE} exists; remove it first"

write_modulefile() {
  local target="$1"
  local mpi_lines heffte_lib cuda_note=""
  heffte_lib="${HEFFTE_PREFIX}/lib64"
  [[ -d "${heffte_lib}" ]] || heffte_lib="${HEFFTE_PREFIX}/lib"
  if [[ "${BACKEND}" == cuda ]]; then
    cuda_note="
CUDA architectures: ${CUDA_ARCHITECTURES} (A100, A40, H100 by default)."
    mpi_lines="depends_on(\"${GCC_MODULE}\")
depends_on(\"${CUDA_MODULE}\")
prepend_path(\"PATH\", \"${OPENMPI_ROOT}/bin\")
prepend_path(\"LD_LIBRARY_PATH\", \"${OPENMPI_ROOT}/lib\")
prepend_path(\"CMAKE_PREFIX_PATH\", \"${OPENMPI_ROOT}\")
setenv(\"MPI_HOME\", \"${OPENMPI_ROOT}\")"
  else
    mpi_lines="depends_on(\"${OPENMPI_MODULE}\")"
  fi
  cat > "${target}" <<EOF
-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Written by scripts/install_release_module.sh from ${TAG} (${REVISION}).

whatis("Name: OpenPFC")
whatis("Version: ${VERSION} (${BACKEND})")
whatis("Description: phase field crystal and structured-grid PDE framework")

help([[
OpenPFC ${VERSION}, ${BACKEND} build of tag ${TAG} (${REVISION}).${cuda_note}

Puts the shipped applications on PATH and the CMake package on
CMAKE_PREFIX_PATH, so find_package(OpenPFC) works for code built against
this install. HeFFTe: ${HEFFTE_PREFIX}

Install provenance: ${PREFIX}/share/openpfc/install-provenance.txt
]])

family("openpfc")
${mpi_lines}

local root = "${PREFIX}"
setenv("OPENPFC_ROOT", root)
prepend_path("PATH", pathJoin(root, "bin"))
prepend_path("LD_LIBRARY_PATH", "${heffte_lib}")
prepend_path("CMAKE_PREFIX_PATH", "${HEFFTE_PREFIX}")
prepend_path("CMAKE_PREFIX_PATH", root)
EOF
}

mkdir -p "${STAGE}/src"
git -C "${SOURCE_REPO}" archive "${REVISION}" | tar -x -C "${STAGE}/src"
write_modulefile "${STAGE}/${NAME}.lua"
echo "Exported ${TAG} (${REVISION}) to ${STAGE}/src"

if (( PREPARE_ONLY )); then
  echo "Prepared modulefile: ${STAGE}/${NAME}.lua"
  exit 0
fi

declare -a BUILD_ARGS=(
  --machine=tohtori
  --build-type=Release
  "--build-dir=${STAGE}/build"
  --no-test
  "--jobs=${JOBS}"
  "--cmake-arg=-DCMAKE_INSTALL_PREFIX=${PREFIX}"
)
if [[ "${BACKEND}" == cuda ]]; then
  BUILD_ARGS+=(--with-cuda)
else
  BUILD_ARGS+=(--cpu)
fi

# The tag's build.sh would otherwise default to $HOME paths and private
# modules for HeFFTe and the CUDA-aware MPI.
HEFFTE_PREFIX="${HEFFTE_PREFIX}" HEFFTE_MODULE="" \
  OPENMPI_ROOT_CUDA="${OPENMPI_ROOT:-/nonexistent}" \
  CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  OPENMPI_MODULE="${OPENMPI_MODULE}" \
  "${STAGE}/src/scripts/build.sh" "${BUILD_ARGS[@]}"

cmake --install "${STAGE}/build"

mkdir -p "${PREFIX}/share/openpfc"
{
  echo "tag: ${TAG}"
  echo "revision: ${REVISION}"
  echo "backend: ${BACKEND}"
  echo "installed: $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${USER} on $(hostname -s)"
  echo "heffte: ${HEFFTE_PREFIX}"
  if [[ "${BACKEND}" == cuda ]]; then
    echo "mpi: ${OPENMPI_ROOT} (${GCC_MODULE}, ${CUDA_MODULE})"
    echo "cuda architectures: ${CUDA_ARCHITECTURES}"
  else
    echo "mpi: ${OPENMPI_MODULE}"
  fi
  echo "configure: $(grep -m1 -E '^CMAKE_CXX_COMPILER:' "${STAGE}/build/CMakeCache.txt" || true)"
} > "${PREFIX}/share/openpfc/install-provenance.txt"

# Refuse a module whose binaries would look for libraries under /home.
leaks="$(find "${PREFIX}" -type f \( -perm -u+x -o -name '*.so*' \) -print0 |
  xargs -0 -r readelf -d 2>/dev/null | grep -E 'R(UN)?PATH' | grep '/home/' || true)"
if [[ -n "${leaks}" ]]; then
  echo "${leaks}" >&2
  die "installed binaries have RUNPATH entries under /home; module not written"
fi

mkdir -p "${MODULE_DIR}"
cp "${STAGE}/${NAME}.lua" "${MODULE_FILE}"
chmod -R a+rX "${PREFIX}" "${ROOT}/modulefiles"
echo "Installed ${TAG} (${BACKEND}) to ${PREFIX}"
echo "Module: module use ${ROOT}/modulefiles && module load openpfc/${NAME}"
