#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Re-measure the Stage-1 planar ladder of PR #147 on a *compute* node.
#
# Why this job exists: the tables in the app README were produced on a LUMI
# login node. The physics is deterministic and single-threaded, so they ought
# to be reproducible anywhere -- but "ought to be" is not a measurement, and a
# login node is shared and throttled, so a number quoted from one is not a
# number anyone can defend. This job re-runs the same cases with the same
# options on `standard` and prints the same columns, so the two tables can be
# diffed rather than trusted.
#
# The cases run concurrently on distinct cores of one exclusive node: each is a
# single-rank pure-FD run, so the whole ladder takes as long as its slowest
# member rather than as long as their sum. Wall time is therefore meaningless
# here and no timing is quoted from this job -- only physics.
#
# Required in the job environment:
#   ALLOY_PLANAR_BIN   absolute path to alloy_dendrite_planar
# Optional:
#   ALLOY_OUT          output directory (default under /scratch)

set -euo pipefail

: "${ALLOY_PLANAR_BIN:?set ALLOY_PLANAR_BIN to the alloy_dendrite_planar binary}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/planar_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SUM="${OUT}/planar_summary.csv"

echo "=== alloy_dendrite planar ladder, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "bin=${ALLOY_PLANAR_BIN}"
echo "out=${OUT}"

# One helper per case: a single-rank srun step pinned to its own core. Each
# writes its own summary CSV; they are concatenated at the end, because a
# shared append target would interleave rows from concurrent writers.
run_case() {
  local id="$1"; shift
  srun --exact --ntasks=1 --cpus-per-task=1 --mem=8G \
    "${ALLOY_PLANAR_BIN}" --summary="${OUT}/sum_${id}.csv" --run-id="${id}" "$@" \
    > "${OUT}/${id}.txt" 2>&1 &
}

# --- resolution ladder, 512 W0 box, t_end = 600, order 4 -------------------
run_case res_dx1.6 --nx=320  --dx=1.6 --velocity=0.1 --t-end=600
run_case res_dx1.2 --nx=427  --dx=1.2 --velocity=0.1 --t-end=600
run_case res_dx0.8 --nx=640  --dx=0.8 --velocity=0.1 --t-end=600
run_case res_dx0.6 --nx=854  --dx=0.6 --velocity=0.1 --t-end=600
run_case res_dx0.4 --nx=1280 --dx=0.4 --velocity=0.1 --t-end=600

# --- FD order at a fixed coarse dx ----------------------------------------
for o in 2 4 6 8; do
  run_case "order_${o}" --nx=427 --dx=1.2 --fd-order="${o}" --velocity=0.1 --t-end=600
done

# --- anti-trapping: k_eff against velocity, three models -------------------
# The V = 0.05 row needs a 1600 W0 box so the 40 W0 boundary layer has room.
for s in 1 0 m1; do
  case "${s}" in
    m1) sv=-1 ;;
    *)  sv="${s}" ;;
  esac
  run_case "at${s}_v0.05" --nx=2667 --dx=0.6 --velocity=0.05 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.10" --nx=854  --dx=0.6 --velocity=0.10 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.20" --nx=854  --dx=0.6 --velocity=0.20 --t-end=600 --at-scale="${sv}"
  run_case "at${s}_v0.40" --nx=854  --dx=0.6 --velocity=0.40 --t-end=600 --at-scale="${sv}"
done

wait

echo "=== summary ==="
head -1 "$(ls "${OUT}"/sum_*.csv | head -1)"
for f in "${OUT}"/sum_*.csv; do
  tail -n +2 "${f}"
done
echo "=== per-case reports ==="
for f in "${OUT}"/*.txt; do
  echo "--- ${f} ---"
  cat "${f}"
done
echo "=== done ==="
