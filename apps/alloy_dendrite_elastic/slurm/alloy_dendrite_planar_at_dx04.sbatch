#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Follow-up to alloy_dendrite_planar_ladder.sbatch.
#
# Re-running the README's anti-trapping table at its stated `dx = 0.6 W0`
# reproduced the `at_scale = 1` column only to three figures and the
# `at_scale = 0 / -1` columns to about 1-7 %, while the resolution ladder in
# the same README reproduced to four. One entry explains it: the README's
# `at_scale = 1`, `V = 0.10` cell reads 0.15024, which is exactly the
# resolution ladder's `dx = 0.4` value, not its `dx = 0.6` value of 0.15018.
# That is a testable claim about which grid the table was run on, so this job
# tests it instead of arguing about it: same four velocities, same three
# models, `dx = 0.4 W0`, boxes scaled to keep 512 W0 (1600 W0 at V = 0.05).
#
# Required in the job environment:
#   ALLOY_PLANAR_BIN   absolute path to alloy_dendrite_planar

set -euo pipefail

: "${ALLOY_PLANAR_BIN:?set ALLOY_PLANAR_BIN to the alloy_dendrite_planar binary}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/at04_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== anti-trapping table at dx = 0.4 W0, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="

run_case() {
  local id="$1"; shift
  srun --exact --ntasks=1 --cpus-per-task=1 --mem=16G \
    "${ALLOY_PLANAR_BIN}" --summary="${OUT}/sum_${id}.csv" --run-id="${id}" "$@" \
    > "${OUT}/${id}.txt" 2>&1 &
}

for s in 1 0 m1; do
  case "${s}" in
    m1) sv=-1 ;;
    *)  sv="${s}" ;;
  esac
  run_case "at${s}_v0.05" --nx=4000 --dx=0.4 --velocity=0.05 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.10" --nx=1280 --dx=0.4 --velocity=0.10 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.20" --nx=1280 --dx=0.4 --velocity=0.20 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.40" --nx=1280 --dx=0.4 --velocity=0.40 --t-end=600 --at-scale="${sv}"
done

wait

echo "=== summary ==="
head -1 "$(ls "${OUT}"/sum_*.csv | head -1)"
for f in "${OUT}"/sum_*.csv; do tail -n +2 "${f}"; done
echo "=== done ==="
