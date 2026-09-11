#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --job-name=invhom-s6
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Stage 6: process-parameter CH family vs rotating-square auxetic reference.

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/stage6_${SLURM_JOB_ID}}"
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
echo "=== stage6 job ${SLURM_JOB_ID} -> ${OUT} ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target openpfc_inverse_homogenize test_inverse_homogenization

ctest --output-on-failure --timeout 600 -R 'inverse-homogenization' || true

mat=(
  --nx=64 --ny=64 --nz=1
  --E-solid=1 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3
  --target=auxetic --E-target=0.15 --nu-target=0.4 --steps=0
)

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --init=rotating-squares --dump-h="${OUT}/rotating_h.txt" \
  > "${OUT}/rotating.txt"

for c0 in 0.35 0.50 0.65; do
  for kappa in 1 4; do
    for ay in 1.0 0.25; do
      name="ch_c${c0}_k${kappa}_ay${ay}"
      srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
        --init=spinodal --init-volume="${c0}" --ch-kappa="${kappa}" \
        --ch-steps=200 --ch-dt=0.2 --ch-aniso-y="${ay}" --init-amp=0.12 \
        --dump-h="${OUT}/${name}_h.txt" > "${OUT}/${name}.txt"
    done
  done
done

echo "=== done ==="
echo "OUT ${OUT}"
