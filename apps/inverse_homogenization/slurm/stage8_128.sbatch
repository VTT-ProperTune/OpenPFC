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
#SBATCH --time=00:45:00
#SBATCH --job-name=invhom-128
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# 128^3 spinodal C_H after MPI_Finalize/HeFFTe dtor fix.

set -uo pipefail
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/inverse-homogenization/hom128_${SLURM_JOB_ID}}"
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
echo "=== 128^3 job ${SLURM_JOB_ID} $(hostname) ==="
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" --target openpfc_inverse_homogenize

srun --ntasks=8 --cpus-per-task=8 "${BIN}" \
  --nx=128 --ny=128 --nz=128 --steps=0 \
  --init=spinodal --init-volume=0.5 --ch-kappa=1 --ch-steps=40 --ch-dt=0.2 \
  --E-solid=1 --nu-solid=0.3 --E-void=0.05 --nu-void=0.3 \
  --target=isotropic --E-target=0.4 --nu-target=0.3 \
  > "${OUT}/hom128.txt"

echo "=== done ==="
tail -n 25 "${OUT}/hom128.txt"
