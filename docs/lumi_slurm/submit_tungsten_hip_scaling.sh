#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Submit the first #87 LUMI-G tungsten_hip slice from a LUMI login node.
# This host (Tohtori) cannot reach the LUMI scheduler.
#
# Usage:
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip ./submit_tungsten_hip_scaling.sh size
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip TUNGSTEN_LX=512 \
#     ./submit_tungsten_hip_scaling.sh strong
#
# Optional environment:
#   PARTITION          default small-g (dev-g for bring-up)
#   TUNGSTEN_LX        cubic grid; size tries several, strong uses this (512)
#   TUNGSTEN_STEPS     accepted steps (default 20)
#   OPENPFC_SCALING_ROOT  scratch parent for run directories
#   ACCOUNT            default project_462001519

set -euo pipefail

: "${TUNGSTEN_HIP_BIN:?set TUNGSTEN_HIP_BIN to the 0.2 tungsten_hip binary}"
if [[ ! -x "${TUNGSTEN_HIP_BIN}" ]]; then
  echo "TUNGSTEN_HIP_BIN is not executable: ${TUNGSTEN_HIP_BIN}" >&2
  exit 1
fi

MODE="${1:-}"
if [[ "${MODE}" != "size" && "${MODE}" != "strong" ]]; then
  echo "usage: $0 size|strong" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/tungsten_hip_scaling.sbatch"
TEMPLATE="${SCRIPT_DIR}/tungsten_hip_scaling.toml"
PARTITION="${PARTITION:-small-g}"
ACCOUNT="${ACCOUNT:-project_462001519}"
STEPS="${TUNGSTEN_STEPS:-20}"
export TUNGSTEN_HIP_BIN
export TUNGSTEN_STEPS="${STEPS}"
export TUNGSTEN_SCALING_TEMPLATE="${TEMPLATE}"
export OPENPFC_SCALING_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"

submit_one() {
  local n="$1"
  local lx="$2"
  local job_name="$3"
  export TUNGSTEN_LX="${lx}"
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks="${n}" \
    --ntasks-per-node="${n}" \
    --gpus-per-node="${n}" \
    --job-name="${job_name}" \
    --export=ALL \
    "${SBATCH}"
}

case "${MODE}" in
  size)
    echo "Sizing 1 GCD (I/O off, ${STEPS} steps, partition=${PARTITION})"
    for lx in 256 384 512 640 768; do
      submit_one 1 "${lx}" "thip-size-${lx}"
    done
    ;;
  strong)
    LX="${TUNGSTEN_LX:-512}"
    echo "Strong scaling ${LX}³ on 1/2/4/8 GCDs (I/O off, ${STEPS} steps, partition=${PARTITION})"
    for n in 1 2 4 8; do
      submit_one "${n}" "${LX}" "thip-strong-${n}gcd-lx${LX}"
    done
    ;;
esac

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"
echo "Keep job ids next to the profiles. See docs/hpc/lumi_gpu_scaling.md."
