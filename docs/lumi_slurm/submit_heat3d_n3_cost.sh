#!/bin/bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Issue #124: submit the frozen 1-node Heat3D GPU cost
# matrix. Bill project_462001519. Refuse project_462001245.
#
# Usage (LUMI login):
#   ./docs/lumi_slurm/submit_heat3d_n3_cost.sh check
#   ./docs/lumi_slurm/submit_heat3d_n3_cost.sh build
#   HEAT3D_SPECTRAL_HIP_BIN=... HEAT3D_HIP_BIN=... \
#     ./docs/lumi_slurm/submit_heat3d_n3_cost.sh submit
#   ./docs/lumi_slurm/submit_heat3d_n3_cost.sh collect

set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/heat3d_n3_cost.sbatch"
PY="${SCRIPT_DIR}/../../apps/heat3d/scripts/n3_cost_matrix.py"

ALLOWED_ACCOUNTS="project_462001519"
ACCOUNT="${ACCOUNT:-project_462001519}"
PARTITION="${PARTITION:-standard-g}"
CAMPAIGN_ROOT="${OPENPFC_SCALING_ROOT:-/scratch/project_462001519/juaho/openpfc-scaling/heat3d-n3-cost-592}"
LOGDIR="${OPENPFC_LOG_DIR:-/scratch/project_462001519/juaho/logs}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_BUILD="/flash/project_462001519/juaho/build/openpfc-lumi-rocm-n3-cost-592"
DEFAULT_SPEC="${DEFAULT_BUILD}/apps/heat3d/heat3d_spectral_hip"
DEFAULT_FD="${DEFAULT_BUILD}/apps/heat3d/heat3d_fd_hip"

account_allowed() {
  local a="$1" x
  for x in ${ALLOWED_ACCOUNTS}; do
    if [[ "${x}" == "${a}" ]]; then
      return 0
    fi
  done
  return 1
}
if ! account_allowed "${ACCOUNT}"; then
  echo "refusing ACCOUNT=${ACCOUNT}; allowed: ${ALLOWED_ACCOUNTS}" >&2
  exit 2
fi

case "${MODE}" in
  check|build|submit|collect) ;;
  *)
    echo "usage: $0 check|build|submit|collect" >&2
    exit 1
    ;;
esac

export OPENPFC_SCALING_ROOT="${CAMPAIGN_ROOT}"

if [[ "${MODE}" == "check" ]]; then
  python3 "${PY}" --check
  exit 0
fi
if [[ "${MODE}" == "collect" ]]; then
  python3 "${PY}" --harvest "${CAMPAIGN_ROOT}" \
    --out "${SCRIPT_DIR}/../../docs/hpc"
  exit 0
fi

SRC="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${OPENPFC_SRC:-}" ]]; then
  leaked="$(cd "${OPENPFC_SRC}" 2>/dev/null && pwd || true)"
  if [[ -n "${leaked}" && "${leaked}" != "${SRC}" ]]; then
    echo "ignoring leaked OPENPFC_SRC=${OPENPFC_SRC}; using ${SRC}" >&2
  fi
fi
export OPENPFC_SRC="${SRC}"
if command -v git >/dev/null 2>&1 &&
   git -C "${SRC}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export OPENPFC_REVISION="$(git -C "${SRC}" rev-parse HEAD)"
  export OPENPFC_DIRTY="$(git -C "${SRC}" status --porcelain | wc -l | tr -d ' ')"
fi
export OPENPFC_REVISION="${OPENPFC_REVISION:-unknown}"
export OPENPFC_DIRTY="${OPENPFC_DIRTY:-unknown}"

unset SBATCH_ACCOUNT SLURM_ACCOUNT SBATCH_PARTITION SLURM_PARTITION \
  HEAT3D_USE_PENCILS OPENPFC_FFT_PROC_GRID OPENPFC_FFT_NODE_GRID || true

if [[ -n "${HEAT3D_SPECTRAL_HIP_BIN:-}" &&
      "${HEAT3D_SPECTRAL_HIP_BIN}" == *heffte-trace* ]]; then
  echo "ignoring diagnostic HEAT3D_SPECTRAL_HIP_BIN=${HEAT3D_SPECTRAL_HIP_BIN}" >&2
  unset HEAT3D_SPECTRAL_HIP_BIN
fi
if [[ -n "${BUILD_DIR:-}" && "${BUILD_DIR}" != *n3-cost-592* ]]; then
  echo "ignoring leaked BUILD_DIR=${BUILD_DIR}" >&2
  unset BUILD_DIR
fi
if [[ -n "${HEFFTE_MODULE:-}" && "${HEFFTE_MODULE}" == *trace* ]]; then
  echo "ignoring diagnostic HEFFTE_MODULE=${HEFFTE_MODULE}" >&2
  unset HEFFTE_MODULE HEFFTE_PREFIX HEFFTE_DIR
fi
HEAT3D_SPECTRAL_HIP_BIN="${HEAT3D_SPECTRAL_HIP_BIN:-${DEFAULT_SPEC}}"
HEAT3D_HIP_BIN="${HEAT3D_HIP_BIN:-${DEFAULT_FD}}"
export HEAT3D_SPECTRAL_HIP_BIN HEAT3D_HIP_BIN
export HEAT3D_STEPS="${HEAT3D_STEPS:-30}"
export HEAT3D_WARMUP="${HEAT3D_WARMUP:-5}"
export HEAT3D_DT="${HEAT3D_DT:-0.01}"
export BUILD_DIR="${BUILD_DIR:-${DEFAULT_BUILD}}"

if [[ "${MODE}" == "build" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${SRC}/scripts/build.sh --machine=lumi --with-rocm --no-test --partition=${PARTITION} --build-dir=${BUILD_DIR}"
    echo "DRY_RUN campaign ${CAMPAIGN_ROOT}"
    echo "DRY_RUN log ${LOGDIR}"
    exit 0
  fi
  mkdir -p "${LOGDIR}"
  cd "${SRC}"
  LUMI_ACCOUNT="${ACCOUNT}" LUMI_PARTITION="${PARTITION}" \
    "${SRC}/scripts/build.sh" \
      --machine=lumi \
      --with-rocm \
      --no-test \
      --no-mpi-tests \
      --partition="${PARTITION}" \
      --build-dir="${BUILD_DIR}"
  exit 0
fi

if [[ ! -x "${HEAT3D_SPECTRAL_HIP_BIN}" ]]; then
  echo "HEAT3D_SPECTRAL_HIP_BIN is not executable: ${HEAT3D_SPECTRAL_HIP_BIN}" >&2
  exit 2
fi
if [[ ! -x "${HEAT3D_HIP_BIN}" ]]; then
  echo "HEAT3D_HIP_BIN is not executable: ${HEAT3D_HIP_BIN}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && "${OPENPFC_DIRTY}" != "0" && "${OPENPFC_DIRTY}" != "unknown" ]]; then
  echo "refusing dirty tree (OPENPFC_DIRTY=${OPENPFC_DIRTY})" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN campaign ${CAMPAIGN_ROOT}"
  echo "DRY_RUN log ${LOGDIR}"
else
  mkdir -p "${CAMPAIGN_ROOT}/runs" "${LOGDIR}"
fi
export OPENPFC_LOG_DIR="${LOGDIR}"

submit_one() {
  local repeat="$1"
  local name="h3-n3-r${repeat}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN sbatch --job-name=${name} --export HEAT3D_REPEAT=${repeat}"
    return 0
  fi
  sbatch --account="${ACCOUNT}" --partition="${PARTITION}" \
    --job-name="${name}" \
    --export=ALL,HEAT3D_REPEAT="${repeat}",HEAT3D_SPECTRAL_HIP_BIN="${HEAT3D_SPECTRAL_HIP_BIN}",HEAT3D_HIP_BIN="${HEAT3D_HIP_BIN}",OPENPFC_SCALING_ROOT="${CAMPAIGN_ROOT}",OPENPFC_REVISION="${OPENPFC_REVISION}",OPENPFC_DIRTY="${OPENPFC_DIRTY}",HEAT3D_STEPS="${HEAT3D_STEPS}",HEAT3D_WARMUP="${HEAT3D_WARMUP}",HEAT3D_DT="${HEAT3D_DT}" \
    "${SBATCH}"
}

submit_one 1
submit_one 2
submit_one 3
