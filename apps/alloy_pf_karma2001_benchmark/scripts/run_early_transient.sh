#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Early-transient diagnostics vs Karma 2001 Fig. 1 (t* ≤ 2000).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-${ROOT}/builds/macos-cpu-release/apps/alloy_pf_karma2001_benchmark/alloy_pf_karma2001_benchmark_openmp}"
OUTROOT="${OUTROOT:-${ROOT}/results/alloy_pf_karma2001_benchmark/early}"
NTHREADS="${NTHREADS:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export OPENPFC_KARMA_VD="${OPENPFC_KARMA_VD:-0}"
export OPENPFC_KARMA_BETA0="${OPENPFC_KARMA_BETA0:-0}"
export OPENPFC_KARMA_EPSK="${OPENPFC_KARMA_EPSK:-0}"
export OPENPFC_KARMA_EPSC="${OPENPFC_KARMA_EPSC:-0.02}"
export OPENPFC_KARMA_K="${OPENPFC_KARMA_K:-0.15}"
export OPENPFC_KARMA_OMEGA="${OPENPFC_KARMA_OMEGA:-0.55}"
export OPENPFC_KARMA_LD0="${OPENPFC_KARMA_LD0:-500}"
export OPENPFC_KARMA_SEED_D0="${OPENPFC_KARMA_SEED_D0:-22}"
export OPENPFC_KARMA_STOP_FRAC="${OPENPFC_KARMA_STOP_FRAC:-0.90}"
export OPENPFC_KARMA_SKIP_PNG="${OPENPFC_KARMA_SKIP_PNG:-1}"
export OPENPFC_KARMA_QUIET="${OPENPFC_KARMA_QUIET:-1}"
export OPENPFC_KARMA_TSTAR="${OPENPFC_KARMA_TSTAR:-2000}"
export OPENPFC_KARMA_NHIST="${OPENPFC_KARMA_NHIST:-5}"
unset OPENPFC_KARMA_NOISE OPENPFC_KARMA_TDOT OPENPFC_KARMA_MAX_STEPS || true

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${NTHREADS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/results/.matplotlib}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PY="$(command -v python3.12 || command -v python3)"

run_complete() {
  local hist="$1/tip_history.tsv"
  [[ -f "${hist}" ]] || return 1
  local tstar
  tstar="$(awk 'NF && $1 !~ /^#/ { t=$2 } END { print t+0 }' "${hist}")"
  awk -v t="${tstar}" -v need="${OPENPFC_KARMA_TSTAR}" 'BEGIN { exit !(t >= 0.90 * need) }'
}

run_one() {
  local tag="$1" ang="$2"
  shift 2
  # remaining: env assignments as KEY=VAL
  local out="${OUTROOT}/${tag}"
  mkdir -p "${out}"
  if [[ "${SKIP_EXISTING}" == "1" ]] && run_complete "${out}"; then
    echo "=== skip ${out} ===" >&2
    printf '%s\n' "${out}"
    return 0
  fi
  unset OPENPFC_KARMA_ISO OPENPFC_KARMA_FD OPENPFC_KARMA_HALVES OPENPFC_KARMA_GLASNER \
    OPENPFC_KARMA_TAU_EU OPENPFC_KARMA_DX OPENPFC_KARMA_DT || true
  export OPENPFC_KARMA_DT=0.02
  local kv
  for kv in "$@"; do
    export "${kv}"
  done
  echo "=== ${tag}  ang=${ang}  ISO=${OPENPFC_KARMA_ISO:-1} FD=${OPENPFC_KARMA_FD:-2} HALVES=${OPENPFC_KARMA_HALVES:-1} GLASNER=${OPENPFC_KARMA_GLASNER:-1} DX=${OPENPFC_KARMA_DX:-1} TAU_EU=${OPENPFC_KARMA_TAU_EU:-1} ===" >&2
  "${BIN}" glasner 0.277 "${ang}" "${out}" "${NTHREADS}" | tee "${out}/run.log" >&2
  printf '%s\n' "${out}"
}

mkdir -p "${OUTROOT}"
ROOTS=()

ROOTS+=("$(run_one th0_dx1_iso 0 OPENPFC_KARMA_ISO=1)")
ROOTS+=("$(run_one th0_dx1_std 0 OPENPFC_KARMA_ISO=0)")
ROOTS+=("$(run_one th0_dx1_fd4 0 OPENPFC_KARMA_FD=4)")
ROOTS+=("$(run_one th45_dx1_iso 45 OPENPFC_KARMA_ISO=1)")
ROOTS+=("$(run_one th45_dx1_std 45 OPENPFC_KARMA_ISO=0)")
ROOTS+=("$(run_one th45_dx1_fd4 45 OPENPFC_KARMA_FD=4)")
ROOTS+=("$(run_one th45_dx1_iso_full 45 OPENPFC_KARMA_ISO=1 OPENPFC_KARMA_HALVES=4)")
ROOTS+=("$(run_one th0_dx0.4_paperlike 0 OPENPFC_KARMA_DX=0.4 OPENPFC_KARMA_GLASNER=0 OPENPFC_KARMA_ISO=0 OPENPFC_KARMA_TAU_EU=0)")
ROOTS+=("$(run_one th45_dx0.4_paperlike 45 OPENPFC_KARMA_DX=0.4 OPENPFC_KARMA_GLASNER=0 OPENPFC_KARMA_ISO=0 OPENPFC_KARMA_TAU_EU=0)")

FIGDIR="${ROOT}/results/alloy_pf_karma2001_benchmark/paper/figures"
mkdir -p "${FIGDIR}"
"${PY}" "${HERE}/assess_early_transient.py" "${ROOTS[@]}" --out-dir "${FIGDIR}"
echo "early runs: ${ROOTS[*]}"
echo "figures: ${FIGDIR}"
