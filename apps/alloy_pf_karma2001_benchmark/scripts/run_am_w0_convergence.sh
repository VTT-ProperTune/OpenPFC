#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Extra to the PRL present-model suite: spatially isothermal AM cooling
# (Ω=0, curvature-compensated T0), no noise. Glasner Δx=W0. Physical Al–Cu kinetics (V_D=2 m/s, β₀=0.1 s/m)
# except one no-trap companion. Exponential Ṫ(t)=Ṫ₀ e^{−t/τ} with τ=12 μs
# (ΔT_cool → 120 K), t_end=18 μs, L=7 μm (2× the old 3.5 μm box so the
# circular envelope stays off the far wall).
#
# Laptop-cheap: 88, 44, 20 nm, plus 10 nm. 5 nm is ~8× 10 nm — leave to LUMI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="${ROOT}/builds/macos-cpu-release/apps/alloy_pf_karma2001_benchmark/alloy_pf_karma2001_benchmark_openmp"
OUTROOT="${OUTROOT:-${ROOT}/results/alloy_pf_karma2001_benchmark/am_w0/tau12_L7e-6}"
NTHREADS="${NTHREADS:-8}"
PHI1="${PHI1:-45}"
export OPENPFC_KARMA_TDOT="${OPENPFC_KARMA_TDOT:-1e7}"
export OPENPFC_KARMA_TEND="${OPENPFC_KARMA_TEND:-18e-6}"
export OPENPFC_KARMA_TDECAY="${OPENPFC_KARMA_TDECAY:-12e-6}"
export OPENPFC_KARMA_L="${OPENPFC_KARMA_L:-7e-6}"
export OPENPFC_KARMA_STOP_FRAC="${OPENPFC_KARMA_STOP_FRAC:-0.80}"
export OPENPFC_KARMA_NCONTOUR="${OPENPFC_KARMA_NCONTOUR:-12}"

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${NTHREADS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/results/.matplotlib}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "${MPLCONFIGDIR}"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.12}"
export OPENPFC_KARMA_SKIP_PNG="${OPENPFC_KARMA_SKIP_PNG:-1}"
export OPENPFC_KARMA_QUIET="${OPENPFC_KARMA_QUIET:-1}"
unset OPENPFC_KARMA_VD OPENPFC_KARMA_BETA0 OPENPFC_KARMA_NOISE || true

if [[ ! -x "${BIN}" ]]; then
  echo "missing binary: ${BIN}" >&2
  exit 1
fi

mkdir -p "${OUTROOT}"

run_one() {
  local wnm="$1"
  local tag="${2:-}"
  local out="${OUTROOT}/W${wnm}nm_th${PHI1}${tag}"
  mkdir -p "${out}"
  echo "=== W0=${wnm} nm  tag=${tag:-trap}  L=${OPENPFC_KARMA_L} m  out=${out} ==="
  "${BIN}" am "${wnm}" "${PHI1}" "${out}" "${NTHREADS}" | tee "${out}/run.log"
}

WLIST="${WLIST:-88 44 20 10}"
for w in ${WLIST}; do
  run_one "${w}"
done

NOTRAP_W="${NOTRAP_W:-20}"
export OPENPFC_KARMA_VD=0 OPENPFC_KARMA_BETA0=0
run_one "${NOTRAP_W}" "_notrap"
unset OPENPFC_KARMA_VD OPENPFC_KARMA_BETA0

ROOTS=()
for w in ${WLIST}; do
  ROOTS+=("${OUTROOT}/W${w}nm_th${PHI1}")
done
ROOTS+=("${OUTROOT}/W${NOTRAP_W}nm_th${PHI1}_notrap")
"${PYTHON}" "${ROOT}/apps/alloy_pf_karma2001_benchmark/scripts/plot_figures.py" \
  "${ROOTS[@]}" \
  --out-dir "${OUTROOT}/figures"
