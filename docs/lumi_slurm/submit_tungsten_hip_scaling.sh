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
#   TUNGSTEN_HIP_BIN=/path/to/tungsten_hip TUNGSTEN_LX=768 \
#     PARTITION=standard-g ./submit_tungsten_hip_scaling.sh multinode
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
if [[ "${MODE}" != "size" && "${MODE}" != "strong" && "${MODE}" != "multinode" ]]; then
  echo "usage: $0 size|strong|multinode" >&2
  echo "  multinode: 16 (2 nodes), 24 (3 nodes), and 32 (4 nodes) GCDs" >&2
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
  local nodes="$1"
  local per_node="$2"
  local lx="$3"
  local job_name="$4"
  local ntasks=$((nodes * per_node))
  export TUNGSTEN_LX="${lx}"
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes="${nodes}" \
    --ntasks="${ntasks}" \
    --ntasks-per-node="${per_node}" \
    --gpus-per-node="${per_node}" \
    --job-name="${job_name}" \
    --export=ALL \
    "${SBATCH}"
}

case "${MODE}" in
  size)
    echo "Sizing 1 GCD (I/O off, ${STEPS} steps, partition=${PARTITION})"
    for lx in 256 384 512 640 768; do
      submit_one 1 1 "${lx}" "thip-size-${lx}"
    done
    ;;
  strong)
    LX="${TUNGSTEN_LX:-512}"
    echo "Strong scaling ${LX}³ on 1/2/4/8 GCDs (I/O off, ${STEPS} steps, partition=${PARTITION})"
    for n in 1 2 4 8; do
      submit_one 1 "${n}" "${LX}" "thip-strong-${n}gcd-lx${LX}"
    done
    ;;
  multinode)
    LX="${TUNGSTEN_LX:-768}"
    echo "Strong scaling ${LX}³ on 16/24/32 GCDs (2/3/4 nodes, 8 GCD/node, I/O off, ${STEPS} steps, partition=${PARTITION})"
    submit_one 2 8 "${LX}" "thip-strong-16gcd-lx${LX}"
    submit_one 3 8 "${LX}" "thip-strong-24gcd-lx${LX}"
    submit_one 4 8 "${LX}" "thip-strong-32gcd-lx${LX}"
    ;;
esac

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"
echo "Keep job ids next to the profiles. See docs/hpc/lumi_gpu_scaling.md."
