#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=w-poly
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# 3x3 tungsten PFC polycrystal on HIP. Rebuild for SeedGrid Nx.

set -euo pipefail
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-hip}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/tungsten/poly_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/tungsten/tungsten_hip"

module purge
module load LUMI/25.09 partition/G cpeGNU cray-fftw lumi-CrayPath
[[ -d "${HOME}/privatemodules" ]] && module use "${HOME}/privatemodules"
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

echo "=== tungsten polycrystal ${SLURM_JOB_ID} $(hostname) ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j 8 --target tungsten_hip

mkdir -p "${OUT}/results/tungsten_polycrystal"
cd "${OUT}"
srun --ntasks=1 --gpus-per-node=1 \
  "${BIN}" "${SRC}/apps/tungsten/inputs_json/tungsten_polycrystal.json" \
  | tee "${OUT}/run.log"
ls -lh "${OUT}/results/tungsten_polycrystal" | tail
