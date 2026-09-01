#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# HIP configure/build/ctest on Tohtori amdgpu (MI210, gfx90a).
#
# Submit from the repository root:
#   sbatch scripts/sbatch_openpfc_hip_amdgpu.sbatch
#
# Rebuilds HeFFTe-ROCm into $HOME/opt/heffte/2.4.1-rocm-ompi5 when that prefix
# is missing (matches Open MPI 5.0.10 + ROCm 7.2.1). The older 2.4.1-rocm tree
# was Open MPI 4.1.1 / ROCm 6.4.0 and does not load on current amdgpu nodes.
#
#SBATCH --job-name=openpfc-hip-amdgpu
#SBATCH --partition=amdgpu
#SBATCH --account=ba5503
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:2
#SBATCH --time=02:00:00
#SBATCH --output=/home/juajukka/dev/openpfc/results/openpfc-hip-amdgpu-%j.out
#SBATCH --error=/home/juajukka/dev/openpfc/results/openpfc-hip-amdgpu-%j.err

set -euo pipefail

REPO_ROOT="${OPENPFC_REPO:-/home/juajukka/dev/openpfc}"
cd "${REPO_ROOT}"

export ROCM_ARCHITECTURES="${ROCM_ARCHITECTURES:-gfx90a}"
export ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.1}"
export OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/5.0.10}"
export HEFFTE_VERSION="${HEFFTE_VERSION:-2.4.1}"
export HEFFTE_PREFIX="${HEFFTE_PREFIX:-${HOME}/opt/heffte/${HEFFTE_VERSION}-rocm-ompi5}"
export MPI_HIP_AWARE="${MPI_HIP_AWARE:-0}"
# CTest MPI suites need 2/3/4 ranks. Slurm --ntasks=4 provides the slots;
# oversubscribe is a belt-and-suspenders fallback if PRRTE still counts 1.
export OMPI_MCA_rmaps_base_oversubscribe="${OMPI_MCA_rmaps_base_oversubscribe:-true}"
export PRTE_MCA_rmaps_default_mapping_policy="${PRTE_MCA_rmaps_default_mapping_policy:-:OVERSUBSCRIBE}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-} ntasks=${SLURM_NTASKS:-} gpus=${CUDA_VISIBLE_DEVICES:-}/${ROCR_VISIBLE_DEVICES:-}"
rocm-smi || true

if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  # shellcheck source=/dev/null
  source /usr/share/lmod/lmod/init/bash
fi
module purge
module load "${OPENMPI_MODULE}"
module load "${ROCM_MODULE}"
hash -r

command -v hipcc >/dev/null 2>&1 || { echo "ERROR: hipcc not found after ${ROCM_MODULE}" >&2; exit 1; }
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
# hipcc may be /usr/bin/hipcc; do not use dirname/.. as ROCM_PATH.
_rocm_has_hip_config() {
  [[ -f "${1}/lib/cmake/hip/hip-config.cmake" ]] ||
    [[ -f "${1}/lib64/cmake/hip/hip-config.cmake" ]] ||
    [[ -f "${1}/lib/cmake/hip/HIPConfig.cmake" ]]
}
_user_rocm="${ROCM_PATH:-}"
ROCM_PATH=""
for cand in "${_user_rocm}" /opt/rocm-7.2.1 /opt/rocm "${HIP_PATH:-}"; do
  [[ -n "${cand}" ]] || continue
  if _rocm_has_hip_config "${cand}"; then
    ROCM_PATH="${cand}"
    break
  fi
done
if [[ -z "${ROCM_PATH}" ]]; then
  echo "ERROR: could not locate ROCm prefix with hip-config.cmake (hipcc=$(command -v hipcc))" >&2
  exit 1
fi
export ROCM_PATH
export CMAKE_PREFIX_PATH="${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
echo "hipcc=$(command -v hipcc) ROCM_PATH=${ROCM_PATH}"

heffte_ok() {
  [[ -f "${HEFFTE_PREFIX}/lib64/cmake/Heffte/HeffteConfig.cmake" ]] ||
    [[ -f "${HEFFTE_PREFIX}/lib/cmake/Heffte/HeffteConfig.cmake" ]]
}

if ! heffte_ok; then
  HEFFTE_SRC="${HEFFTE_SRC:-${HOME}/src/heffte-${HEFFTE_VERSION}}"
  HEFFTE_BUILD="${HEFFTE_BUILD:-${HOME}/opt/heffte/build-rocm-${HEFFTE_VERSION}-ompi5}"
  if [[ ! -f "${HEFFTE_SRC}/CMakeLists.txt" ]]; then
    mkdir -p "$(dirname "${HEFFTE_SRC}")"
    tarball="$(dirname "${HEFFTE_SRC}")/v${HEFFTE_VERSION}.tar.gz"
    echo "Downloading HeFFTe ${HEFFTE_VERSION} to ${tarball}"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "${tarball}" \
        "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${HEFFTE_VERSION}.tar.gz"
    else
      wget -q -O "${tarball}" \
        "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${HEFFTE_VERSION}.tar.gz"
    fi
    tar xf "${tarball}" -C "$(dirname "${HEFFTE_SRC}")"
  fi
  echo "Configuring HeFFTe-ROCm -> ${HEFFTE_PREFIX} (ROCM_PATH=${ROCM_PATH})"
  rm -rf "${HEFFTE_BUILD}"
  cmake -S "${HEFFTE_SRC}" -B "${HEFFTE_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${HEFFTE_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
    -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCHITECTURES}" \
    -DHeffte_ENABLE_FFTW=ON \
    -DHeffte_ENABLE_ROCM=ON \
    -DHeffte_ENABLE_GPU_AWARE_MPI=OFF
  cmake --build "${HEFFTE_BUILD}" -j"${SLURM_CPUS_PER_TASK:-16}"
  cmake --install "${HEFFTE_BUILD}"
  heffte_ok || { echo "ERROR: HeFFTe install missing HeffteConfig.cmake under ${HEFFTE_PREFIX}" >&2; exit 1; }
fi

exec ./scripts/build.sh --machine=tohtori --with-rocm --build-type=Debug \
  --build-dir="${OPENPFC_BUILD_DIR:-${REPO_ROOT}/builds/hip-amdgpu-debug}" \
  --jobs="${SLURM_CPUS_PER_TASK:-16}"
