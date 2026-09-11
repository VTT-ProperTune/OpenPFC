#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=invhom-s8
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Stage 8: genuine 3-D (not a 2-D extrusion). Device Green (#157) is still
# open — this is CPU/MPI on standard, not a GPU claim.

set -uo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/stage8_${SLURM_JOB_ID}}"
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
echo "=== stage8 job ${SLURM_JOB_ID} host $(hostname) -> ${OUT} ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_inverse_homogenize

mat=(
  --E-solid=1 --nu-solid=0.3 --E-void=0.05 --nu-void=0.3
  --target=isotropic --E-target=0.4 --nu-target=0.3
  --init=spinodal --ch-kappa=1 --ch-dt=0.2 --init-amp=0.12
  --normalize=1 --dt=0.04 --max-delta=0.04 --project-volume=1
  --lambda-reg=0.05 --lambda-reg-end=0.2 --simp=1 --simp-end=2
)

echo "=== 1 rank 64^3 inverse 12 steps ==="
srun --ntasks=1 --cpus-per-task=8 "${BIN}" "${mat[@]}" \
  --nx=64 --ny=64 --nz=64 --steps=12 --ch-steps=80 \
  --csv="${OUT}/inv64.csv" --dump-h="${OUT}/inv64_h.txt" \
  > "${OUT}/inv64.txt"

echo "=== 8 rank 64^3 inverse 8 steps ==="
srun --ntasks=8 --cpus-per-task=8 "${BIN}" "${mat[@]}" \
  --nx=64 --ny=64 --nz=64 --steps=8 --ch-steps=40 \
  --csv="${OUT}/inv64_mpi8.csv" > "${OUT}/inv64_mpi8.txt"

echo "=== 8 rank 128^3 spinodal homogenize (steps=0) ==="
srun --ntasks=8 --cpus-per-task=8 "${BIN}" "${mat[@]}" \
  --nx=128 --ny=128 --nz=128 --steps=0 --ch-steps=40 \
  > "${OUT}/hom128.txt"

echo "=== done ==="
echo "OUT ${OUT}"
