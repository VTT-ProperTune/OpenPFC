#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Isothermal trap dendrites at β₀=4 s/m, plus matching no-trap (A=β₀=0).
# d0/W = 0.138 is W₀≈88 nm (coarse; expected to sit off the W₀ plateau).
# Δt=0.1 τ₀ is used where stable; 88 nm and 44 nm no-trap need 0.02 τ₀.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="${ROOT}/builds/macos-cpu-release/apps/alloy_pf_karma2001_benchmark/alloy_pf_karma2001_benchmark_openmp"
OUTROOT="${OUTROOT:-${ROOT}/results/karma2001_trap_w0/beta4}"
NTHREADS="${NTHREADS:-8}"
PHI1="${PHI1:-45}"
D0WS=(0.138 0.277 0.544)

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${NTHREADS}"
export OPENPFC_KARMA_SKIP_PNG="${OPENPFC_KARMA_SKIP_PNG:-1}"
export OPENPFC_KARMA_QUIET="${OPENPFC_KARMA_QUIET:-1}"
export OPENPFC_KARMA_NCONTOUR="${OPENPFC_KARMA_NCONTOUR:-2}"
unset OPENPFC_KARMA_VD OPENPFC_KARMA_BETA0 OPENPFC_KARMA_NOISE || true

if [[ ! -x "${BIN}" ]]; then
  echo "missing binary: ${BIN}" >&2
  exit 1
fi

mkdir -p "${OUTROOT}"

run_one() {
  local d0w="$1"
  local tag="$2"
  local dt="$3"
  local out="${OUTROOT}/d0W_${d0w}_th${PHI1}${tag}"
  mkdir -p "${out}"
  export OPENPFC_KARMA_DT="${dt}"
  echo "=== d0/W=${d0w}  tag=${tag:-trap}  Δt=${dt} τ0  out=${out} ==="
  "${BIN}" glasner "${d0w}" "${PHI1}" "${out}" "${NTHREADS}" | tee "${out}/run.log"
}

# 88 nm (0.138) is unstable at 0.1 τ₀. 44 nm no-trap also needs 0.02 τ₀.
for d0w in "${D0WS[@]}"; do
  dt="0.1"
  if [[ "${d0w}" == "0.138" ]]; then dt="0.02"; fi
  run_one "${d0w}" "" "${dt}"
done

export OPENPFC_KARMA_VD=0 OPENPFC_KARMA_BETA0=0
for d0w in "${D0WS[@]}"; do
  dt="0.02"
  if [[ "${d0w}" == "0.544" ]]; then dt="0.1"; fi
  run_one "${d0w}" "_notrap" "${dt}"
done
unset OPENPFC_KARMA_VD OPENPFC_KARMA_BETA0 OPENPFC_KARMA_DT

PLOT=()
for d0w in "${D0WS[@]}"; do
  PLOT+=("${OUTROOT}/d0W_${d0w}_th${PHI1}" "${OUTROOT}/d0W_${d0w}_th${PHI1}_notrap")
done

PY="${ROOT}/.venv/bin/python3"
if ! command -v "${PY}" >/dev/null 2>&1; then
  PY="$(command -v python3.12 || command -v python3)"
fi
"${PY}" "${ROOT}/apps/alloy_pf_karma2001_benchmark/scripts/plot_figures.py" \
  "${PLOT[@]}" \
  --out-dir "${OUTROOT}/figures"
