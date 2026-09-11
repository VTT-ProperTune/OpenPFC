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
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Stage 2 re-measured under the corrected anisotropy, on a compute node.
#
# Two things changed under this application since the README's Stage-2 table
# was written, and both of them move it, so it has to be re-run rather than
# re-labelled:
#
#   1. `MODEL_SPEC.md` was corrected on 2026-09-11 to the normalised
#      Karma-Rappel `a_s`, which is now the default. The shipped preset's
#      `eps4` moved from 0.2 to 0.0435 to keep the *effective* anisotropy at
#      the 4.35 % the old convention's 0.2 produced -- but `a_s` itself no
#      longer carries the `1 + 0.75 eps4` offset, so `W(n)` and `tau(n)` are
#      smaller by that factor at every orientation and the tip does not have
#      to come out the same.
#   2. The table was produced on a login node.
#
# The run with `--aniso-form=unnormalised --eps4=0.2` is included so the two
# conventions can be compared at fixed effective anisotropy in one job rather
# than across two documents.
#
# Required in the job environment:
#   ALLOY_GROWTH_BIN   absolute path to alloy_dendrite_growth

set -euo pipefail

: "${ALLOY_GROWTH_BIN:?set ALLOY_GROWTH_BIN to the alloy_dendrite_growth binary}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/stage2_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== alloy_dendrite Stage 2, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="

run_case() {
  local id="$1"; shift
  srun --exact --ntasks=1 --cpus-per-task=1 --mem=8G \
    "${ALLOY_GROWTH_BIN}" --summary="${OUT}/sum_${id}.csv" \
    --csv="${OUT}/ts_${id}.csv" --run-id="${id}" "$@" \
    > "${OUT}/${id}.txt" 2>&1 &
}

# The shipped preset, and the old convention at the same effective anisotropy.
run_case shipped
run_case legacy_eps02 --aniso-form=unnormalised --eps4=0.2

# Tip-radius sensitivity to the parabola fit window, at two anisotropies.
for w in 3 5 8 12; do
  run_case "fit${w}_e0435" --eps4=0.0435 --tip-halfwidth="${w}"
  run_case "fit${w}_e0233" --eps4=0.0233 --tip-halfwidth="${w}"
done

# 3-D, one modest brick: Stage 3's morphology check needs a machine, but a
# 96^3 run says whether the 3-D CPU path still grows an arm and conserves.
run_case dendrite3d --nx=96 --ny=96 --nz=96 --dx=1.0 --t-end=60 --seed-radius=6

wait

echo "=== summary ==="
head -1 "$(ls "${OUT}"/sum_*.csv | head -1)"
for f in "${OUT}"/sum_*.csv; do tail -n +2 "${f}"; done
echo "=== shipped time series ==="
cat "${OUT}/ts_shipped.csv"
echo "=== legacy time series ==="
cat "${OUT}/ts_legacy_eps02.csv"
echo "=== reports ==="
for f in "${OUT}"/*.txt; do echo "--- ${f} ---"; cat "${f}"; done
echo "=== done ==="
