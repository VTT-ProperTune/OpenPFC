#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --job-name=invhom-s7
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/stage7_${SLURM_JOB_ID}}"
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
echo "=== stage7 job ${SLURM_JOB_ID} -> ${OUT} ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target openpfc_inverse_homogenize test_inverse_homogenization

ctest --output-on-failure --timeout 600 -R 'inverse-homogenization' || true

mat=(
  --nx=64 --ny=64 --nz=1 --steps=0
  --E-solid=1 --nu-solid=0.3 --E-void=0.02 --nu-void=0.3
  --target=auxetic --E-target=0.15 --nu-target=0.4
)

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --init=rotating-squares > "${OUT}/rotating.txt"

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --init=rotating-squares --init-half=0.16 > "${OUT}/islands.txt"

srun --ntasks=1 --cpus-per-task=4 "${BIN}" "${mat[@]}" \
  --init=spinodal --init-volume=0.5 --ch-kappa=1 --ch-steps=200 --ch-dt=0.2 \
  --init-amp=0.12 > "${OUT}/spinodal.txt"

echo "=== done ==="
echo "OUT ${OUT}"
