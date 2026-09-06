#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Submit heat3d_fd_hip LUMI-G strong-scaling jobs from a LUMI login node.
#
# Usage:
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip ./submit_heat3d_fd_hip_scaling.sh size
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip HEAT3D_N=256 \
#     ./submit_heat3d_fd_hip_scaling.sh strong
#   HEAT3D_HIP_BIN=/path/to/heat3d_fd_hip HEAT3D_N=256 \
#     PARTITION=standard-g ./submit_heat3d_fd_hip_scaling.sh multinode
#
# Optional:
#   PARTITION, HEAT3D_N, HEAT3D_STEPS, HEAT3D_DT, HEAT3D_FD_ORDER,
#   OPENPFC_SCALING_ROOT, ACCOUNT

set -euo pipefail

: "${HEAT3D_HIP_BIN:?set HEAT3D_HIP_BIN to the 0.2 heat3d_fd_hip binary}"
if [[ ! -x "${HEAT3D_HIP_BIN}" ]]; then
  echo "HEAT3D_HIP_BIN is not executable: ${HEAT3D_HIP_BIN}" >&2
  exit 1
fi

MODE="${1:-}"
if [[ "${MODE}" != "size" && "${MODE}" != "strong" && "${MODE}" != "multinode" ]]; then
  echo "usage: $0 size|strong|multinode" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/heat3d_fd_hip_scaling.sbatch"
PARTITION="${PARTITION:-small-g}"
ACCOUNT="${ACCOUNT:-project_462001519}"
STEPS="${HEAT3D_STEPS:-20}"
export HEAT3D_HIP_BIN
export HEAT3D_STEPS="${STEPS}"
export HEAT3D_DT="${HEAT3D_DT:-0.01}"
export HEAT3D_FD_ORDER="${HEAT3D_FD_ORDER:-2}"
export OPENPFC_SCALING_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling}"

submit_one() {
  local nodes="$1"
  local per_node="$2"
  local n="$3"
  local job_name="$4"
  local ntasks=$((nodes * per_node))
  export HEAT3D_N="${n}"
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
    echo "Sizing 1 GCD heat3d_fd_hip (I/O off, ${STEPS} steps, partition=${PARTITION})"
    for n in 128 256 384 512; do
      submit_one 1 1 "${n}" "h3dhip-size-${n}"
    done
    ;;
  strong)
    N="${HEAT3D_N:-256}"
    echo "Strong scaling heat3d_fd_hip ${N}³ on 1/2/4/8 GCDs (I/O off, ${STEPS} steps, partition=${PARTITION})"
    for g in 1 2 4 8; do
      submit_one 1 "${g}" "${N}" "h3dhip-strong-${g}gcd-n${N}"
    done
    ;;
  multinode)
    N="${HEAT3D_N:-256}"
    echo "Strong scaling heat3d_fd_hip ${N}³ on 16/24/32 GCDs (2/3/4 nodes, partition=${PARTITION})"
    submit_one 2 8 "${N}" "h3dhip-strong-16gcd-n${N}"
    submit_one 3 8 "${N}" "h3dhip-strong-24gcd-n${N}"
    submit_one 4 8 "${N}" "h3dhip-strong-32gcd-n${N}"
    ;;
esac

echo "Logs: /scratch/project_462001519/juaho/logs/"
echo "Runs: ${OPENPFC_SCALING_ROOT}/runs/"
echo "Keep job ids next to the profiles. See docs/hpc/lumi_gpu_scaling.md."
