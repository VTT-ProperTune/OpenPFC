#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reconfigure and build OpenPFC (and kobayashi_fd_manual) on a compute node using
# Open MPI with Slurm PMI/PMIx support — typically **module openmpi/5.0.10** on gen05_epyc.
#
# HeFFTe must have been built and linked against the **same** Open MPI you load here.
# If CMake finds the wrong libmpi or configure fails, rebuild HeFFTe after:
#   module load gcc/11.2.0 openmpi/5.0.10
#   (then follow INSTALL.md §3 into $HOME/opt/heffte/2.4.1-cpu or a separate suffix tree).
#
# Submit from the OpenPFC repository root (or set OPENPFC_REPO):
#   export OPENPFC_REPO=$PWD
#   export OPENPFC_BUILD_DIR=$PWD/builds/kobayashi-ompi510-slurm   # optional
#   sbatch apps/kobayashi/slurm/kobayashi_rebuild_openpfc_gen05_epyc.sbatch
#
# After this job, point scaling jobs at OPENPFC_BUILD_DIR and load **openmpi/5.0.10**
# in the runtime sbatch (see kobayashi_scaling_gen05_epyc.sbatch).

#SBATCH --job-name=kob-rebuild-ompi510
#SBATCH --partition=gen05_epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=kobayashi_rebuild_openpfc_%j.out
#SBATCH --error=kobayashi_rebuild_openpfc_%j.err

set -euo pipefail

OPENPFC_REPO="${OPENPFC_REPO:-${SLURM_SUBMIT_DIR:-.}}"
if [[ ! -f "${OPENPFC_REPO}/CMakeLists.txt" ]]; then
  echo "ERROR: OPENPFC_REPO=${OPENPFC_REPO} does not look like the OpenPFC root." >&2
  exit 1
fi

OPENPFC_BUILD_DIR="${OPENPFC_BUILD_DIR:-${OPENPFC_REPO}/builds/kobayashi-ompi510-slurm}"
GCC_MODULE="${GCC_MODULE:-gcc/11.2.0}"
OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"

# Clean module state so mpicc/mpirun resolve to this job's Open MPI, not a stale PATH entry.
module purge 2>/dev/null || true
module load "${GCC_MODULE}"
module load "${OPENMPI_MODULE}"
if module load "${CUDA_MODULE}" 2>/dev/null; then
  echo "Loaded CUDA module: ${CUDA_MODULE}"
else
  echo "WARN: CUDA module '${CUDA_MODULE}' unavailable; continuing CPU-only (set CUDA_MODULE or ignore)."
fi
hash -r

MPI_CC="$(command -v mpicc)"
MPI_CXX="$(command -v mpicxx)"
if [[ -z "${MPI_CC}" || -z "${MPI_CXX}" ]]; then
  echo "ERROR: mpicc/mpicxx not on PATH after module load." >&2
  exit 1
fi
# Do not `readlink -f` mpicc: some installs symlink to opal_wrapper under a different prefix
# (e.g. /export/... vs /share/...) and the wrapper then fails to find opal_wrapper-wrapper-data.txt.
OPENMPI_ROOT=""
if command -v mpicc >/dev/null 2>&1; then
  OPENMPI_ROOT="$(mpicc --showme:prefix 2>/dev/null || true)"
fi
if [[ -z "${OPENMPI_ROOT}" || ! -x "${OPENMPI_ROOT}/bin/mpicc" ]]; then
  OPENMPI_ROOT="$(dirname "$(dirname "${MPI_CC}")")"
fi
export OPENMPI_ROOT
if [[ ! -x "${OPENMPI_ROOT}/bin/mpicc" ]]; then
  echo "ERROR: OPENMPI_ROOT=${OPENMPI_ROOT} does not contain bin/mpicc." >&2
  exit 1
fi

echo "=== kobayashi rebuild (Slurm PMI-capable Open MPI) ==="
echo "host=$(hostname) JOB_ID=${SLURM_JOB_ID:-}"
echo "OPENPFC_REPO=${OPENPFC_REPO}"
echo "OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "OPENMPI_ROOT=${OPENMPI_ROOT}"
echo "MPI_CC=${MPI_CC}"
echo "MPI_CXX=${MPI_CXX}"
type -a mpicc mpicxx || true
gcc --version | head -n 1
"${MPI_CC}" --version | head -n 1
mpirun --version | head -n 3

# Many sites: openmpi/5.0.10 pulls a newer GCC (e.g. 15.x). That is fine if HeFFTe matches.
# Re-using an existing CMake build dir WITHOUT wiping leaves MPI_CXX_COMPILER pinned to an old Open MPI
# (CMake ignores CMAKE_TOOLCHAIN_FILE on incremental configure — see job logs).
if [[ "${KOBAYASHI_REBUILD_KEEP_BUILD_DIR:-0}" != "1" ]]; then
  echo "Removing build dir for clean MPI discovery: ${OPENPFC_BUILD_DIR}"
  rm -rf "${OPENPFC_BUILD_DIR}"
fi
mkdir -p "${OPENPFC_BUILD_DIR}"

TOOLCHAIN="${OPENPFC_REPO}/cmake/toolchains/tohtori-gcc11-openmpi.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
  echo "ERROR: missing toolchain: ${TOOLCHAIN}" >&2
  exit 1
fi

# Toolchain defaults to Open MPI 4.1.1 unless OPENMPI_ROOT is visible; pin wrappers explicitly.
cmake -S "${OPENPFC_REPO}" -B "${OPENPFC_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  -DMPI_C_COMPILER="${MPI_CC}" \
  -DMPI_CXX_COMPILER="${MPI_CXX}" \
  -DOpenPFC_BUILD_TESTS=OFF \
  -DOpenPFC_BUILD_EXAMPLES=OFF \
  -DOpenPFC_BUILD_DOCUMENTATION=OFF \
  -DOpenPFC_ENABLE_CODE_COVERAGE=OFF

if ! grep '^MPI_CXX_COMPILER:' "${OPENPFC_BUILD_DIR}/CMakeCache.txt" | grep -qF "${OPENMPI_ROOT}"; then
  echo "ERROR: MPI_CXX_COMPILER in CMakeCache is not under OPENMPI_ROOT=${OPENMPI_ROOT}" >&2
  grep '^MPI_' "${OPENPFC_BUILD_DIR}/CMakeCache.txt" || true
  exit 1
fi

cmake --build "${OPENPFC_BUILD_DIR}" -j"${SLURM_CPUS_PER_TASK:-16}" --target kobayashi_fd_manual

BIN="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_manual"
if ! ldd "${BIN}" 2>/dev/null | grep -q 'libmpi'; then
  echo "ERROR: ldd ${BIN} failed" >&2
  exit 1
fi
if ! ldd "${BIN}" 2>/dev/null | grep 'libmpi\.so' | grep -q "${OPENMPI_ROOT}"; then
  echo "ERROR: ${BIN} is not linked against libmpi under OPENMPI_ROOT=${OPENMPI_ROOT}" >&2
  ldd "${BIN}" | grep -E 'mpi|openmpi' || true
  exit 1
fi

BIN_OPENMP="${OPENPFC_BUILD_DIR}/apps/kobayashi/kobayashi_fd_openmp"
# OpenMP_CXX_FOUND is set by FindOpenMP but usually not written to CMakeCache.txt. OpenMP_CXX_FLAGS
# is cached (e.g. -fopenmp); treat non-empty, non-NOTFOUND flags as OpenMP CXX available.
_openmp_flags_line="$(grep -E '^OpenMP_CXX_FLAGS:' "${OPENPFC_BUILD_DIR}/CMakeCache.txt" 2>/dev/null | head -n1 || true)"
_openmp_flags_val="${_openmp_flags_line#*=}"
_openmp_flags_val="${_openmp_flags_val//$'\r'/}"
_openmp_ok=0
if [[ -n "${_openmp_flags_val}" && "${_openmp_flags_val}" != "NOTFOUND" ]]; then
  _openmp_ok=1
fi
_openmp_found_line="$(grep -E '^OpenMP_CXX_FOUND:' "${OPENPFC_BUILD_DIR}/CMakeCache.txt" 2>/dev/null | head -n1 || true)"
_openmp_found_val="${_openmp_found_line#*=}"
_openmp_found_val="${_openmp_found_val//$'\r'/}"
if [[ "${_openmp_found_val}" == "ON" || "${_openmp_found_val}" == "TRUE" || "${_openmp_found_val}" == "1" ]]; then
  _openmp_ok=1
fi
if [[ "${_openmp_ok}" == "1" ]]; then
  echo "Building kobayashi_fd_openmp (OpenMP thread scaling driver)..."
  cmake --build "${OPENPFC_BUILD_DIR}" -j"${SLURM_CPUS_PER_TASK:-16}" --target kobayashi_fd_openmp
  if [[ ! -x "${BIN_OPENMP}" ]]; then
    echo "ERROR: ${BIN_OPENMP} missing after build (OpenMP target failed)." >&2
    exit 1
  fi
  echo "OK: ${BIN_OPENMP}"
else
  echo "WARN: OpenMP_CXX not found at configure time — kobayashi_fd_openmp was not generated." >&2
  echo "      Install OpenMP-enabled GCC toolchain; OpenMP scaling sbatch needs this binary." >&2
fi

echo ""
echo "OK: ${BIN}"
echo "Submit MPI scaling with:"
echo "  export OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
echo "  sbatch apps/kobayashi/slurm/kobayashi_scaling_gen05_epyc.sbatch"
if [[ -x "${BIN_OPENMP}" ]]; then
  echo "Submit OpenMP thread scaling with:"
  echo "  export OPENPFC_BUILD_DIR=${OPENPFC_BUILD_DIR}"
  echo "  sbatch apps/kobayashi/slurm/kobayashi_openmp_scaling_gen05_epyc.sbatch"
fi
