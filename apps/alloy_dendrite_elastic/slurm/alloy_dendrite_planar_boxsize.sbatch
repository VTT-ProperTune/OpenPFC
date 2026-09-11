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
# Box-size convergence of the fast rows of the anti-trapping table.
#
# Re-running the README's anti-trapping table on `standard` reproduced its
# `V = 0.05 / 0.10 / 0.20` cells but not `V = 0.40`, where it reports
# `k_eff = 0.15258` (+1.7 %) and the re-run measures 0.1358 (-9.4 %). The
# re-run's diagnostics say why rather than leaving it a mystery: at
# `V = 0.385` over `t_end = 600` each of the two fronts has travelled 231 W0
# into a 256 W0 half-box, so they have met, `U_i` has gone positive and the
# steady mass balance no longer closes. The README's own box-sizing rule --
# "the two fronts' tails must not overlap at the periodic seam" -- is the
# thing being violated, and it is stated in the README but not applied to
# that row.
#
# That is a hypothesis with an obvious test: hold everything else fixed and
# grow the box. If `k_eff` at `V = 0.4` climbs back toward the quoted value
# and the mass balance closes again, the quoted number is a large-box number
# and the table's stated `512 W0` is wrong for that row.
#
# Required in the job environment:
#   ALLOY_PLANAR_BIN   absolute path to alloy_dendrite_planar

set -euo pipefail

: "${ALLOY_PLANAR_BIN:?set ALLOY_PLANAR_BIN to the alloy_dendrite_planar binary}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/box_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== planar box-size convergence, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="

run_case() {
  local id="$1"; shift
  srun --exact --ntasks=1 --cpus-per-task=1 --mem=16G \
    "${ALLOY_PLANAR_BIN}" --summary="${OUT}/sum_${id}.csv" --run-id="${id}" "$@" \
    > "${OUT}/${id}.txt" 2>&1 &
}

# dx = 0.6 W0 throughout; nx = box / dx.
for v in 0.20 0.40; do
  run_case "v${v}_L512"  --nx=854   --dx=0.6 --velocity="${v}" --t-end=600
  run_case "v${v}_L1024" --nx=1707  --dx=0.6 --velocity="${v}" --t-end=600
  run_case "v${v}_L2048" --nx=3413  --dx=0.6 --velocity="${v}" --t-end=600
  run_case "v${v}_L4096" --nx=6827  --dx=0.6 --velocity="${v}" --t-end=600
done

wait

echo "=== summary ==="
head -1 "$(ls "${OUT}"/sum_*.csv | head -1)"
for f in "${OUT}"/sum_*.csv; do tail -n +2 "${f}"; done
echo "=== per-case reports ==="
for f in "${OUT}"/*.txt; do echo "--- ${f} ---"; cat "${f}"; done
echo "=== done ==="
