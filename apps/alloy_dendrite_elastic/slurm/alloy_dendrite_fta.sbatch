#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
#SBATCH --account=project_462001519
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=alloy-dendrite-fta
#SBATCH --output=/scratch/project_462001519/juaho/logs/%x-%j.out
#
# Issue #155: FTA directional science campaign on alloy_dendrite_growth.
# Three required cases (aligned / misori / bicrystal) plus an optional 22.5°
# single grain, concurrent on one LUMI-C node. Pure FD; do not pass --elastic.
#
# Frozen-temperature (infinite Lewis number): evolve_theta=0, Mc=0.5.
# Elevated-G directional geometry, not an Al-Cu furnace:
#   gradient=0.02  =>  l_T = 50 W0
#   pulling=0.05   =>  l_D = D_l / V_p = 40 W0
# theta is scaled by the hypercooling L/c_p ≈ 336 K, not ΔT_0.
#
# Required in the job environment:
#   ALLOY_GROWTH_BIN   absolute path to alloy_dendrite_growth
# Optional:
#   ALLOY_OUT          output directory
#   FTA_RUN_22P5=0     skip the optional 22.5° grain (default: run it)

set -euo pipefail

: "${ALLOY_GROWTH_BIN:?set ALLOY_GROWTH_BIN to the alloy_dendrite_growth binary}"

OUT="${ALLOY_OUT:-/scratch/project_462001519/juaho/alloy-dendrite/fta_${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

module purge
module load LUMI/25.09 partition/C cpeGNU cray-fftw lumi-CrayPath
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== alloy_dendrite FTA science, job ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION} ==="
echo "bin=${ALLOY_GROWTH_BIN}"
echo "out=${OUT}"

SHARED=(
  --nx=640 --ny=256 --nz=1 --dx=0.8 --fd-order=4 --t-end=1200 --samples=240
  --eps4=0.04 --Mc=0.5 --evolve-theta=0 --Dl=2 --omega=0.55
  --gradient=0.02 --pulling=0.05 --fta-x0=24
  --seed-radius=10 --fields-every=20
)

run_case() {
  local id="$1"; shift
  mkdir -p "${OUT}/fields_${id}"
  srun --exact --ntasks=32 --cpus-per-task=1 --mem=32G \
    "${ALLOY_GROWTH_BIN}" "${SHARED[@]}" \
    --csv="${OUT}/ts_${id}.csv" --summary="${OUT}/sum_${id}.csv" \
    --fields-dir="${OUT}/fields_${id}" --run-id="${id}" "$@" \
    > "${OUT}/${id}.txt" 2>&1 &
}

# (a) aligned <100>
run_case fta-aligned --seed-x=24 --seed-y=102.4 --crystal-angle=0

# (b) misoriented, 0.2 rad ≈ 11.5°
run_case fta-misori --seed-x=24 --seed-y=102.4 --crystal-angle=0.2

# (c) two-seed bicrystal, single phi -- not a grain-boundary energy model
run_case fta-bicrystal --seed-x=24 --seed-y=64 --seed2-radius=10 \
  --seed2-x=24 --seed2-y=140.8 --crystal-angle=0.2 --crystal-angle2=-0.2

# Optional fourth grain at 22.5° (π/8). Issue #155 asked for a second
# finite angle; this is it. Set FTA_RUN_22P5=0 to skip.
if [[ "${FTA_RUN_22P5:-1}" != "0" ]]; then
  run_case fta-misori-22.5 --seed-x=24 --seed-y=102.4 --crystal-angle=0.39269908169872414
fi

wait

echo "=== summary ==="
head -1 "$(ls "${OUT}"/sum_*.csv | head -1)"
for f in "${OUT}"/sum_*.csv; do tail -n +2 "${f}"; done
echo "=== reports ==="
for f in "${OUT}"/*.txt; do echo "--- ${f} ---"; cat "${f}"; done
echo "=== done ==="
