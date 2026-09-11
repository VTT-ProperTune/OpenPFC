#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=invhom-aux
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Continue the 64^2 auxetic design and run a second seed.

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/auxetic_${SLURM_JOB_ID}}"
BIN="${BUILD}/apps/inverse_homogenization/openpfc_inverse_homogenize"
PREV="${PREV_H:-/scratch/project_462001519/juaho/inverse-homogenization/stage5_21954505/auxetic_h.txt}"

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
echo "=== auxetic job ${SLURM_JOB_ID} -> ${OUT} ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_inverse_homogenize

common=(
  --nx=64 --ny=64 --nz=1 --target=auxetic
  --E-solid=2 --E-void=0.08 --E-target=0.4 --nu-target=0.4
  --dt=0.04 --max-delta=0.04 --normalize=1 --project-volume=1
  --volume=0.5 --lambda-volume=0.2 --epsilon=2.5 --W-12=5
)

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${common[@]}" \
  --steps=150 --simp=3 --lambda-reg=0.4 --lambda-reg-end=0.75 \
  --load-h="${PREV}" --init=uniform \
  --csv="${OUT}/continue.csv" --dump-h="${OUT}/continue_h.txt" \
  > "${OUT}/continue.txt"

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${common[@]}" \
  --steps=160 --simp=1 --simp-end=3 --lambda-reg=0.05 --lambda-reg-end=0.75 \
  --init=noise --init-amp=0.35 --init-volume=0.5 --seed=7 \
  --csv="${OUT}/seed7.csv" --dump-h="${OUT}/seed7_h.txt" \
  > "${OUT}/seed7.txt"

# Evaluate thresholded C_H of the two finished designs (no extra descent).
for name in continue seed7; do
  srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${common[@]}" \
    --steps=0 --load-h="${OUT}/${name}_h.txt" --init=uniform \
    > "${OUT}/${name}_eval.txt"
done

echo "=== done ==="
echo "OUT ${OUT}"
