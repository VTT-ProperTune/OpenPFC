#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reserved N=128 dx=pi/3 mask-on/off wall_step (Paper A RQ2).
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=w-dealias-res
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-heat3d-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/results/tungsten_dealias}"
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
export TUNGSTEN_WARMUP=5

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
  --target tungsten_dealias_study

srun --ntasks=8 --cpus-per-task=16 --cpu-bind=cores \
  "${BUILD}/apps/tungsten/tungsten_dealias_study" --reserved \
  "${OUT}/tungsten_dealias_reserved-${SLURM_JOB_ID}.csv"
echo "=== done ==="
