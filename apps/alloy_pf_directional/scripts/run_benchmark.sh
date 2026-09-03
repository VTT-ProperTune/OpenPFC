#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LOCKED starting point: 12×3.2 µm, W0=10 nm, two grains, noise off.
# Default CLI / `start` / `benchmark` freeze physics; MAX_STEPS still applies.
#
#   ./apps/alloy_pf_directional/scripts/run_benchmark.sh
#   QUICK=1 ./apps/alloy_pf_directional/scripts/run_benchmark.sh   # 400 steps (seeds only)
#   ./apps/alloy_pf_directional/scripts/run_benchmark.sh --plot-only
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
BUILD="${BUILD:-${OPENPFC_BUILD_DIR:-${ROOT}/builds/macos-cpu-release}}"
BIN="${BIN:-${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_openmp}"
OUT="${OUT:-${ROOT}/results/alloy_pf_directional/benchmark/ly3.2_w10nm_bicrystal}"
REF="${REF:-${OUT}/reference}"
NTHREADS="${NTHREADS:-${OMP_NUM_THREADS:-8}}"
PY="${PYTHON:-/opt/homebrew/bin/python3.12}"
if [[ ! -x "${PY}" ]]; then
  PY="${PYTHON:-python3}"
fi

plot_only=0
if [[ "${1:-}" == "--plot-only" ]]; then
  plot_only=1
fi

plot_from=""
if [[ -f "${REF}/phi_final.raw" ]]; then
  plot_from="${REF}"
elif [[ -f "${OUT}/phi_final.raw" ]]; then
  plot_from="${OUT}"
fi

if [[ "${plot_only}" == 1 ]]; then
  if [[ -z "${plot_from}" ]]; then
    echo "no fields in ${REF} or ${OUT}" >&2
    exit 1
  fi
  export MPLCONFIGDIR="${ROOT}/results/.matplotlib"
  export MPLBACKEND=Agg
  mkdir -p "${MPLCONFIGDIR}" "${OUT}/figures"
  "${PY}" "${HERE}/plot_benchmark.py" "${plot_from}" --out "${OUT}/figures"
  "${PY}" "${HERE}/plot_bicrystal.py" "${plot_from}" \
    --label "ly3.2 W0=10 nm bicrystal" --out "${OUT}/figures"
  exit 0
fi

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" >&2
  exit 1
fi

unset OPENPFC_ALCU_W0 OPENPFC_ALCU_DXW OPENPFC_ALCU_LX OPENPFC_ALCU_LY OPENPFC_ALCU_LZ || true
unset OPENPFC_ALCU_G OPENPFC_ALCU_VP OPENPFC_ALCU_TEND OPENPFC_ALCU_SEED || true
unset OPENPFC_ALCU_NGRANS OPENPFC_ALCU_THETA OPENPFC_ALCU_NOISE OPENPFC_ALCU_NOISE_SEED || true
unset OPENPFC_ALCU_DT_OVER_TAU OPENPFC_ALCU_WINDOW OPENPFC_ALCU_OMEGA || true
unset OPENPFC_ALCU_NY OPENPFC_ALCU_NZ OPENPFC_ALCU_NDIM || true
unset OPENPFC_ALCU_WARMUP OPENPFC_ALCU_TIMED_STEPS || true
export OPENPFC_ALCU_SKIP_PNG="${OPENPFC_ALCU_SKIP_PNG:-1}"
export OPENPFC_ALCU_SKIP_VTK="${OPENPFC_ALCU_SKIP_VTK:-1}"
if [[ "${QUICK:-0}" == 1 ]]; then
  export OPENPFC_ALCU_MAX_STEPS="${MAX_STEPS:-400}"
  echo "QUICK=1: MAX_STEPS=${OPENPFC_ALCU_MAX_STEPS} (seeds only; not the morphology gold)"
else
  unset OPENPFC_ALCU_MAX_STEPS || true
fi
if [[ "$(uname -s)" == Darwin ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
fi
export OMP_NUM_THREADS="${NTHREADS}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

mkdir -p "${OUT}"
echo "== benchmark  nthreads=${NTHREADS}  out=${OUT} =="
"${BIN}" benchmark "${OUT}" "${NTHREADS}" | tee "${OUT}/run.log"

export MPLCONFIGDIR="${ROOT}/results/.matplotlib"
export MPLBACKEND=Agg
mkdir -p "${MPLCONFIGDIR}" "${OUT}/figures"
"${PY}" "${HERE}/plot_benchmark.py" "${OUT}" --out "${OUT}/figures"
