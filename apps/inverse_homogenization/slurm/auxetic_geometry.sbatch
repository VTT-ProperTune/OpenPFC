#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --job-name=invhom-ageo
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# 1) Catch2: rotating-square and re-entrant seeds must have C12<0.
# 2) Evaluate both seeds (steps=0) and inverse from the one that works.

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/auxgeo_${SLURM_JOB_ID}}"
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
echo "=== auxetic geometry job ${SLURM_JOB_ID} -> ${OUT} ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target openpfc_inverse_homogenize test_inverse_homogenization

ctest --output-on-failure --timeout 600 -R inverse-homogenization || true

mat=(
  --E-solid=1 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3
  --target=auxetic --E-target=0.15 --nu-target=0.4
  --project-volume=1 --volume=0.5 --lambda-volume=0.1
  --normalize=1 --dt=0.03 --max-delta=0.03 --epsilon=2.5 --W-12=6
)

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --nx=64 --ny=64 --nz=1 --steps=0 --init=rotating-squares \
  --dump-h="${OUT}/rotating_h.txt" > "${OUT}/rotating_eval.txt"

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --nx=64 --ny=64 --nz=1 --steps=0 --init=reentrant \
  --dump-h="${OUT}/reentrant_h.txt" > "${OUT}/reentrant_eval.txt"

# Inverse from rotating squares (keep volume of the seed, don't force 0.5
# if the mechanism needs a different fraction). --project-volume=0.
srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --nx=64 --ny=64 --nz=1 --steps=80 --init=rotating-squares \
  --project-volume=0 --simp=1 --simp-end=2.5 \
  --lambda-reg=0.02 --lambda-reg-end=0.25 \
  --csv="${OUT}/from_rotating.csv" --dump-h="${OUT}/from_rotating_h.txt" \
  > "${OUT}/from_rotating.txt"

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --nx=64 --ny=64 --nz=1 --steps=80 --init=reentrant \
  --project-volume=0 --simp=1 --simp-end=2.5 \
  --lambda-reg=0.02 --lambda-reg-end=0.25 \
  --csv="${OUT}/from_reentrant.csv" --dump-h="${OUT}/from_reentrant_h.txt" \
  > "${OUT}/from_reentrant.txt"

echo "=== done ==="
echo "OUT ${OUT}"
