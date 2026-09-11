#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=invhom-s5
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Stage 5 start: three target tensors on 16^3, Allen-Cahn inverse.
# Compile the inverse binary on this node if the tree is stale.

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/stage5_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

mkdir -p "${OUT}"
echo "=== stage5 job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} -> ${OUT} ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_inverse_homogenize

run() {
  local name="$1"; shift
  srun --ntasks=1 --cpus-per-task=4 \
    "${BIN}" --steps=40 --dt=0.03 --max-delta=0.03 --normalize=1 \
    --project-volume=1 --init=noise --volume=0.5 \
    --lambda-volume=1 --lambda-reg=0.08 \
    --csv="${OUT}/${name}.csv" "$@" > "${OUT}/${name}.txt"
}

# 3-D isotropic / orthotropic; 2-D high-contrast auxetic (grey two-phase
# interpolation cannot produce nu<0 — binary topology is required).
run isotropic --nx=16 --ny=16 --nz=16 --target=isotropic \
  --E-target=0.9 --nu-target=0.25
run auxetic --nx=32 --ny=32 --nz=1 --target=auxetic \
  --E-solid=2 --E-void=0.02 --E-target=0.4 --nu-target=0.4 \
  --lambda-reg=0.15 --epsilon=1.5
run orthotropic --nx=16 --ny=16 --nz=16 --target=orthotropic \
  --C11=1.3 --C22=0.6 --C12=0.2 --C66=0.25

echo "=== done ==="
echo "OUT ${OUT}"
