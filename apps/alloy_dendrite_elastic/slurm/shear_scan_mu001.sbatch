#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Single remaining Paper-C experiment (research issue #189):
# solver-qualify mu_l/mu_s = 0.001 on the same 256^2 vehicle as
# job 21995259, by raising --el-iter from 50 to 400.
#
# Not a shear sweep. Not the 960^2 science box.
#
# Admit only if el_nonconverged=0 and el_iter_max<400 at tol_el=1e-6.
# If the cap is still hit, stop; do not redesign the solver here.
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --job-name=dend-mu001
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err

set -euo pipefail
BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-heat3d-cpu}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/results/dendrite_shear_mu001}"
HEFFTE_LIB="/flash/project_462001519/juaho/build/heffte-2.3.0-milan/lib"
export LD_LIBRARY_PATH="${HEFFTE_LIB}:${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset LD_PRELOAD
export CRAYPE_LINK_TYPE=dynamic MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=16
module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath

BIN="${BUILD}/apps/alloy_dendrite_elastic/alloy_dendrite_growth"
test -x "$BIN"
mkdir -p "${OUT}"
COMMON=(--nx=256 --ny=256 --t-end=200 --quiet=1 --elastic=1)

run_one() {
  local tag="$1"; shift
  echo "=== ${tag} ==="
  srun --ntasks=8 --cpus-per-task=16 --cpu-bind=cores \
    "${BIN}" "${COMMON[@]}" --run-id="${tag}" \
    --summary="${OUT}/summary-${tag}-${SLURM_JOB_ID}.csv" \
    --csv="${OUT}/series-${tag}-${SLURM_JOB_ID}.csv" \
    "$@"
}

run_one off --lambda-el=0 --n-el-substep=1000000
run_one mu0.001 --el-mu-liquid=0.001 --el-iter=400 --el-tol=1e-6
echo "=== done ==="
