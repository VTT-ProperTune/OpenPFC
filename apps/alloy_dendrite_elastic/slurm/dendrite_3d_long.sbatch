#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=ad-3d-long
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Longer 256^3 isothermal dendrite (elastic off). t_end=220 so the <100>
# arms grow ~100 W0 — past the rounded-cross 150-tau0 512^3 snapshot.

set -euo pipefail
BIN="${ALLOY_GROWTH_BIN:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu/apps/alloy_dendrite_elastic/alloy_dendrite_growth}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/alloy-dendrite/long3d_${SLURM_JOB_ID}}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
[[ -d "${HOME}/privatemodules" ]] && module use "${HOME}/privatemodules"
module load heffte-cpu || module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"

mkdir -p "${OUT}/fields"
echo "=== 3-D dendrite long ${SLURM_JOB_ID} $(hostname) ntasks=${SLURM_NTASKS} ==="
ls -l "${BIN}"

srun "${BIN}" \
  --nx=256 --ny=256 --nz=256 --dx=1.0 \
  --t-end=220 --samples=12 --fields-every=1 \
  --seed-radius=6 --eps4=0.05 --elastic=0 \
  --fields-dir="${OUT}/fields" \
  --csv="${OUT}/tip.csv" \
  --summary="${OUT}/summary.csv" \
  --run-id=long3d \
  | tee "${OUT}/run.log"

echo "=== summary ==="
cat "${OUT}/summary.csv"
echo "=== tip tail ==="
tail -8 "${OUT}/tip.csv"
ls -lh "${OUT}/fields" | head
