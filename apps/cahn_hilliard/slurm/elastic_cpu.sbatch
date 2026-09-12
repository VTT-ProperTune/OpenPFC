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
#SBATCH --job-name=ch-elastic
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#SBATCH --error=/scratch/project_462001519/juaho/logs/%x-%j.err
#
# Compile cahn_hilliard_elastic on LUMI-C (not the login node) and run
# Catch2 plus a 64^2 science-preset smoke.

set -euo pipefail

BUILD="${BUILD_DIR:-/flash/project_462001519/juaho/build/openpfc-lumi-invhom-cpu}"
SRC="${SRC_DIR:-/pfs/lustref1/flash/project_462001519/juaho/dev/openpfc}"
OUT="${OUT_DIR:-/scratch/project_462001519/juaho/cahn-hilliard/elastic_${SLURM_JOB_ID}}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
if [[ -d "${HOME}/privatemodules" ]]; then
  module use "${HOME}/privatemodules"
fi
module load heffte-cpu || module load heffte-rocm || true
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CRAYPE_LINK_TYPE="${CRAYPE_LINK_TYPE:-dynamic}"
export CC=cc CXX=CC

echo "=== CH elastic CPU job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} $(hostname) ==="
module list
cd "${BUILD}"
cmake --build "${BUILD}" -j "${SLURM_NTASKS}" \
  --target cahn_hilliard_elastic test_cahn_hilliard

echo "=== Catch2 elastic ==="
srun --ntasks=1 --cpus-per-task=4 \
  "${BUILD}/apps/cahn_hilliard/test_cahn_hilliard" "[cahn_hilliard][elastic]"

echo "=== 64^2 fe_cr_elastic.json ==="
mkdir -p "${OUT}"
cd "${OUT}"
srun --ntasks=1 --cpus-per-task=4 \
  "${BUILD}/apps/cahn_hilliard/cahn_hilliard_elastic" \
  "${SRC}/apps/cahn_hilliard/inputs_json/fe_cr_elastic.json" \
  | tee "${OUT}/run.log"
echo "=== done ==="
ls -lh "${OUT}/results/fe_cr_elastic" | tail
tail -5 "${OUT}/results/fe_cr_elastic/diagnostics.csv"
