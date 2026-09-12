#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LUMI-C equal-node CPU control for the linear Heat3D protocol slice
# (research/articles/periodic-operator-choice).
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=heat3d-lumi-c
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-heat3d-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/results/heat3d_lumi_c}"
N=1024
STEPS=30
DT=0.01
export HEAT3D_WARMUP=5
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
HEFFTE_CMAKE="/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib/cmake/Heffte"
export LD_LIBRARY_PATH="/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib:${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# gcc 14 libstdc++ must win over the gcc 11.2 RPATH inherited from HeFFTe.
export LD_PRELOAD="/usr/lib64/libstdc++.so.6:/usr/lib64/libgcc_s.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export MPICH_GPU_SUPPORT_ENABLED=0
export CC=cc
export CXX=CC

echo "=== heat3d LUMI-C control job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "host $(hostname)  build ${BUILD}"
echo "revision $(git -C "${SRC}" rev-parse HEAD)"
module list

mkdir -p "${OUT}"
# Relink on the C node so MPI is host GTL, not HSA, and gcc 11.2 is not in RPATH.
rm -f "${BUILD}/CMakeCache.txt"
rm -rf "${BUILD}/CMakeFiles"
cmake -S "${SRC}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${SRC}/cmake/toolchains/lumi-gcc12-mpich.cmake" \
  -DOpenPFC_ENABLE_HEFFTE=ON \
  -DOpenPFC_BUILD_TESTS=OFF \
  -DOpenPFC_BUILD_APPS=ON \
  -DOpenPFC_BUILD_EXAMPLES=OFF \
  -DOpenPFC_ENABLE_HDF5=OFF \
  -DOpenPFC_ENABLE_CUDA=OFF \
  -DOpenPFC_ENABLE_HIP=OFF \
  -DOpenPFC_MPI_HIP_AWARE=OFF \
  -DMPI_C_COMPILER=cc \
  -DMPI_CXX_COMPILER=CC \
  -DHeffte_DIR="${HEFFTE_CMAKE}"
cmake --build "${BUILD}" -j "${SLURM_CPUS_ON_NODE}" \
  --target heat3d_fd heat3d_spectral
# Drop the gcc-11 RPATH that shadows gcc-14 libstdc++.
if command -v chrpath >/dev/null 2>&1; then
  chrpath -r "/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib" \
    "${BUILD}/apps/heat3d/heat3d_fd" "${BUILD}/apps/heat3d/heat3d_spectral" || true
fi

run_one() {
  local tag="$1"
  shift
  local log="${OUT}/${tag}-${SLURM_JOB_ID}.log"
  echo "=== ${tag} ==="
  srun --ntasks=8 --cpus-per-task=16 --cpu-bind=cores "$@" | tee "${log}"
}

run_one spectral "${BUILD}/apps/heat3d/heat3d_spectral" "${N}" "${STEPS}" "${DT}"
run_one fd2 "${BUILD}/apps/heat3d/heat3d_fd" "${N}" "${STEPS}" "${DT}" 2
run_one fd8 "${BUILD}/apps/heat3d/heat3d_fd" "${N}" "${STEPS}" "${DT}" 8
run_one fd12 "${BUILD}/apps/heat3d/heat3d_fd" "${N}" "${STEPS}" "${DT}" 12

echo "=== medians ==="
grep HEAT3D_CPU_WALL_STEP_MS_MEDIAN "${OUT}"/*-"${SLURM_JOB_ID}".log || true
echo "=== done ==="
