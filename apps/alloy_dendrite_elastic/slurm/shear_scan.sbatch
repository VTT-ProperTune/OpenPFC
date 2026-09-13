#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 2-D liquid-shear scan for Paper C (inviscid-limit question).
# Smaller than the 960^2 science case: nx=ny=256, t_end=200, 8 ranks.
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=dend-shear
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-heat3d-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/results/dendrite_shear}"
HEFFTE_LIB="/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib"
HEFFTE_CMAKE="${HEFFTE_LIB}/cmake/Heffte"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${HEFFTE_LIB}:${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset LD_PRELOAD
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=0
export CC=cc CXX=CC
export OMP_NUM_THREADS=16

mkdir -p "${OUT}"
cmake -S "${SRC}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${SRC}/cmake/toolchains/lumi-gcc12-mpich.cmake" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,${HEFFTE_LIB}" \
  -DOpenPFC_ENABLE_HEFFTE=ON -DOpenPFC_BUILD_TESTS=OFF \
  -DOpenPFC_BUILD_APPS=ON -DOpenPFC_BUILD_EXAMPLES=OFF \
  -DOpenPFC_ENABLE_HDF5=OFF -DOpenPFC_ENABLE_CUDA=OFF \
  -DOpenPFC_ENABLE_HIP=OFF -DOpenPFC_MPI_HIP_AWARE=OFF \
  -DMPI_C_COMPILER=cc -DMPI_CXX_COMPILER=CC \
  -DMPI_C_HEADER_DIR="${MPICH_DIR}/include" \
  -DMPI_CXX_HEADER_DIR="${MPICH_DIR}/include" \
  -DMPI_C_LIB_NAMES=mpi -DMPI_CXX_LIB_NAMES=mpi \
  -DMPI_mpi_LIBRARY="${MPICH_DIR}/lib/libmpi.so" \
  -DHeffte_DIR="${HEFFTE_CMAKE}"
cmake --build "${BUILD}" -j "${SLURM_CPUS_ON_NODE}" \
  --target alloy_dendrite_growth

BIN="${BUILD}/apps/alloy_dendrite_elastic/alloy_dendrite_growth"
COMMON=(--nx=256 --ny=256 --t-end=200 --quiet=1)

run_one() {
  local tag="$1"
  shift
  echo "=== ${tag} ==="
  srun --ntasks=8 --cpus-per-task=16 --cpu-bind=cores \
    "${BIN}" "${COMMON[@]}" \
    --run-id="${tag}" \
    --summary="${OUT}/summary-${tag}-${SLURM_JOB_ID}.csv" \
    --csv="${OUT}/series-${tag}-${SLURM_JOB_ID}.csv" \
    "$@"
}

run_one off --elastic=1 --lambda-el=0 --n-el-substep=1000000
for mu in 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.10; do
  run_one "mu${mu}" --elastic=1 --el-mu-liquid="${mu}"
done
echo "=== done ==="
ls -l "${OUT}"
