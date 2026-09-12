#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --job-name=invhom-hip
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Compile and run HIP Green-operator homogenization on LUMI-G.
# Configure (FetchContent) is assumed already done on the login node.

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-hip}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/hip_${SLURM_JOB_ID}}"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}"
echo "=== invhom HIP job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} $(hostname) ==="
module list
if [[ ! -f "${BUILD}/build.ninja" && ! -f "${BUILD}/Makefile" ]]; then
  echo "No HIP build tree at ${BUILD}. Configure on the login node with"
  echo "  ./scripts/build.sh --machine=lumi --partition=standard-g --no-test --no-submit --build-dir=${BUILD}"
  echo "then re-run this sbatch. Compile stays on standard-g, not the login node."
  exit 2
fi
cd "${BUILD}"
cmake --build "${BUILD}" -j 8 --target openpfc_homogenize_hip openpfc_inverse_homogenize_hip

BIN="${BUILD}/apps/inverse_homogenization/openpfc_homogenize_hip"
srun --ntasks=1 --gpus-per-node=1 "${BIN}" \
  --shape=homogeneous --nx=16 --ny=16 --nz=16 --volume=1 \
  > "${OUT}/homogeneous.txt"
srun --ntasks=1 --gpus-per-node=1 "${BIN}" \
  --shape=rotating-squares --nx=32 --ny=32 --nz=1 \
  --E-solid=1 --E-void=0.02 --nu-solid=0.3 --nu-void=0.3 \
  > "${OUT}/rotating.txt"
srun --ntasks=1 --gpus-per-node=1 "${BIN}" \
  --shape=homogeneous --nx=64 --ny=64 --nz=64 --volume=1 \
  > "${OUT}/hom64.txt"

echo "=== done ==="
echo "OUT ${OUT}"
cat "${OUT}/homogeneous.txt"
echo "--- rotating ---"
cat "${OUT}/rotating.txt"
