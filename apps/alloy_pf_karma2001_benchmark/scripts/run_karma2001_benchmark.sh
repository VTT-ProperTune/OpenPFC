#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Karma 2001 present-model isothermal dendrite.
# Matches PRL 87, 115701 (2001): A = 0, β₀ = 0, k = 0.15, ε_c = 0.02,
# isothermal Ω = 0.55, seed 22 d0. ε_k is off (β₀ = 0).
#
# Default paper suite (SUITE=paper): three [100] cases, dense tip history,
# centered LS velocity in compare_karma2001.py (Δt* = 80).
#   1. fast Glasner:     d0/W=0.277, Δx=W0,  Δt=0.09 τ0  (max stable with e^u)
#   2. thicker W:        d0/W=0.544, Δx=W0,  Δt=0.09 τ0
#   3. 2001-like mesh:   d0/W=0.277, Δx=0.4 W0, no Glasner, 5-pt, τ at e^u=1,
#                        Δt/τ0=0.02 → Δt=0.008 τ0 (paper)
#
# Optional: GRID=1 adds one 45° Glasner check. DT_SCAN=1 restores the old
# 45° Δt family (not part of the paper comparison). DX_PINNING=0 skips the
# extra d0/W=0.277 Δx=0.6 and 0.8 W0 runs used for the pinning figure.
#
#   ./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh
#   QUICK=1 ./apps/alloy_pf_karma2001_benchmark/scripts/run_karma2001_benchmark.sh
#   --plot-only existing_run_dir ...
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-${ROOT}/builds/macos-cpu-release/apps/alloy_pf_karma2001_benchmark/alloy_pf_karma2001_benchmark_openmp}"
USER_OUTROOT="${OUTROOT:-}"
NTHREADS="${NTHREADS:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GRID="${GRID:-0}"
DT_SCAN="${DT_SCAN:-0}"
DX_PINNING="${DX_PINNING:-1}"

# Present-model paper protocol. Driver-level overrides use KARMA_* (not leftover
# OPENPFC_KARMA_* from diagnostic scripts). Trapping/AM have their own scripts.
export OPENPFC_KARMA_VD=0
export OPENPFC_KARMA_BETA0=0
export OPENPFC_KARMA_EPSK=0
export OPENPFC_KARMA_EPSC="${KARMA_EPSC:-0.02}"
export OPENPFC_KARMA_K="${KARMA_K:-0.15}"
export OPENPFC_KARMA_OMEGA="${KARMA_OMEGA:-0.55}"
export OPENPFC_KARMA_LD0="${KARMA_LD0:-1000}"
export OPENPFC_KARMA_SEED_D0="${KARMA_SEED_D0:-22}"
export OPENPFC_KARMA_STOP_FRAC="${KARMA_STOP_FRAC:-0.90}"
export OPENPFC_KARMA_SKIP_PNG="${OPENPFC_KARMA_SKIP_PNG:-1}"
export OPENPFC_KARMA_QUIET="${OPENPFC_KARMA_QUIET:-1}"
export OPENPFC_KARMA_NCONTOUR="${KARMA_NCONTOUR:-2}"
unset OPENPFC_KARMA_NOISE OPENPFC_KARMA_TDOT || true
# Leftover diagnostic env (early-transient, full-plane, Δt probes) must not
# silently change the advertised suite.
unset OPENPFC_KARMA_MAX_STEPS OPENPFC_KARMA_HALVES OPENPFC_KARMA_NHIST || true
unset OPENPFC_KARMA_ISO OPENPFC_KARMA_GLASNER OPENPFC_KARMA_TAU_EU || true
unset OPENPFC_KARMA_DX OPENPFC_KARMA_DT OPENPFC_KARMA_FD || true

if [[ "${QUICK:-0}" == "1" ]]; then
  # Ignore leftover OPENPFC_KARMA_TSTAR / LD0 from diagnostic scripts.
  export OPENPFC_KARMA_TSTAR="${KARMA_QUICK_TSTAR:-80}"
  export OPENPFC_KARMA_LD0="${KARMA_QUICK_LD0:-1000}"
  OUTROOT="${USER_OUTROOT:-${ROOT}/results/alloy_pf_karma2001_benchmark/quick}"
  echo "QUICK=1: t*=${OPENPFC_KARMA_TSTAR}, L/d0=${OPENPFC_KARMA_LD0}, OUTROOT=${OUTROOT} (not a paper-length run)"
else
  export OPENPFC_KARMA_TSTAR="${KARMA_TSTAR:-10000}"
  OUTROOT="${USER_OUTROOT:-${ROOT}/results/alloy_pf_karma2001_benchmark/paper}"
fi

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${NTHREADS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/results/.matplotlib}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "${MPLCONFIGDIR}"

PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python3" ]]; then
    PY="${ROOT}/.venv/bin/python3"
  else
    PY="$(command -v python3.12 || command -v python3)"
  fi
fi

COMPARE="${HERE}/compare_karma2001.py"

if [[ "${1:-}" == "--plot-only" ]]; then
  shift
  if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 --plot-only RUN_DIR [RUN_DIR ...] [--dx-scan DIR ...] [--dt-scan DIR ...]" >&2
    exit 2
  fi
  exec "${PY}" "${COMPARE}" "$@" --out-dir "${OUTROOT}/figures"
fi

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" >&2
  echo "build: cmake --build builds/macos-cpu-release --target alloy_pf_karma2001_benchmark_openmp" >&2
  exit 1
fi

echo "Karma 2001 present model: A_trap from VD=${OPENPFC_KARMA_VD}  beta0=${OPENPFC_KARMA_BETA0}  k=${OPENPFC_KARMA_K}  eps_c=${OPENPFC_KARMA_EPSC}  eps_k=${OPENPFC_KARMA_EPSK}  Omega=${OPENPFC_KARMA_OMEGA}  Tdot=0  t*=${OPENPFC_KARMA_TSTAR}  L/d0=${OPENPFC_KARMA_LD0}  R_seed/d0=${OPENPFC_KARMA_SEED_D0}  default τ uses local e^u"

run_complete() {
  local hist="$1/tip_history.tsv"
  [[ -f "${hist}" ]] || return 1
  local tstar
  tstar="$(awk 'NF && $1 !~ /^#/ { t=$2 } END { print t+0 }' "${hist}")"
  awk -v t="${tstar}" -v need="${OPENPFC_KARMA_TSTAR}" 'BEGIN { exit !(t >= 0.95 * need) }'
}

# History stride so Δt* between samples is ≲ 8 (centered LS window is 80).
nhist_for() {
  local dtt="$1" dxw="$2" d0w="$3"
  "${PY}" -c "dtt=float('${dtt}'); dxw=float('${dxw}'); d0w=float('${d0w}'); ts=dtt*dxw*0.8839*0.6267/(d0w**3); print(max(1, int(round(8.0/max(ts, 1e-9)))))"
}

run_case() {
  local d0w="$1" ang="$2" dxw="$3" dtt="$4"
  local tau_eu="${5:-1}"
  local glasner="${6:-1}"
  local iso="${7:-1}"
  local tag out nhist
  tag="d0W_${d0w}_th${ang}_dx${dxw}"
  if [[ "${dtt}" != "0.02" || "${ang}" != "0" ]]; then
    tag="${tag}_dt${dtt}"
  fi
  if [[ "${tau_eu}" != "1" ]]; then
    tag="${tag}_notauEU"
  fi
  if [[ "${glasner}" != "1" ]]; then
    tag="${tag}_paperlike"
  fi
  out="${OUTROOT}/${tag}"
  mkdir -p "${out}"
  if [[ "${SKIP_EXISTING}" == "1" ]] && run_complete "${out}"; then
    echo "=== skip complete ${out} ===" >&2
    printf '%s\n' "${out}"
    return 0
  fi
  nhist="${OPENPFC_KARMA_NHIST:-$(nhist_for "${dtt}" "${dxw}" "${d0w}")}"
  echo "=== d0/W=${d0w}  phi1=${ang} deg  Δx/W0=${dxw}  dt/τ0=${dtt}  glasner=${glasner} iso=${iso} tau_eu=${tau_eu} n_hist=${nhist}  out=${out} ===" >&2
  OPENPFC_KARMA_DX="${dxw}" \
    OPENPFC_KARMA_DT="${dtt}" \
    OPENPFC_KARMA_TAU_EU="${tau_eu}" \
    OPENPFC_KARMA_GLASNER="${glasner}" \
    OPENPFC_KARMA_ISO="${iso}" \
    OPENPFC_KARMA_NHIST="${nhist}" \
    "${BIN}" glasner "${d0w}" "${ang}" "${out}" "${NTHREADS}" | tee "${out}/run.log" >&2
  printf '%s\n' "${out}"
}

probe_dt_stable() {
  local dtt="$1"
  local tau_eu="${2:-1}"
  local tmp log
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/karma_dtprobe.XXXXXX")"
  log="${tmp}/run.log"
  echo "probe Δt/τ0=${dtt} tau_eu=${tau_eu} (2000 steps, 45°, Δx=W0) ..." >&2
  set +e
  OPENPFC_KARMA_MAX_STEPS=2000 OPENPFC_KARMA_DX=1.0 OPENPFC_KARMA_DT="${dtt}" \
    OPENPFC_KARMA_TAU_EU="${tau_eu}" \
    "${BIN}" glasner 0.277 45 "${tmp}" "${NTHREADS}" >"${log}" 2>&1
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "  unstable (binary exit ${rc})" >&2
    rm -rf "${tmp}"
    return 1
  fi
  if grep -Eq 'min_phi=nan|max_phi=nan|min_c=nan|mass1=nan' "${log}"; then
    echo "  unstable (NaN in KARMA_VERIFY)" >&2
    rm -rf "${tmp}"
    return 1
  fi
  # KARMA_VERIFY ... min_phi=... max_phi=... min_c=... max_c=...
  "${PY}" - "${log}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(
    r"min_phi=([-\d.eE+]+) max_phi=([-\d.eE+]+) min_c=([-\d.eE+]+) max_c=([-\d.eE+]+)",
    text,
)
if not m:
    sys.exit(1)
vals = [float(x) for x in m.groups()]
mn_p, mx_p, mn_c, mx_c = vals
ok = (
    all(abs(v) < 1.0e30 for v in vals)
    and mn_p >= -1.05
    and mx_p <= 1.05
    and mn_c > 0.0
    and mx_c < 5.0
)
sys.exit(0 if ok else 1)
PY
  local prc=$?
  if [[ "${prc}" -ne 0 ]]; then
    echo "  unstable (phi/c bounds)" >&2
    rm -rf "${tmp}"
    return 1
  fi
  echo "  stable" >&2
  rm -rf "${tmp}"
  return 0
}

mkdir -p "${OUTROOT}"
ROOTS=()
DT_ROOTS=()
DX_ROOTS=()

# Paper comparison: two Glasner widths at max stable Δt, plus 2001-like Δx=0.4 W.
# Dense n_hist (Δt* ≲ 8) + centered LS — do not shrink Δt just to smooth V*(t).
ROOTS+=("$(run_case 0.277 0 1.0 0.09)")
ROOTS+=("$(run_case 0.544 0 1.0 0.09)")
ROOTS+=("$(run_case 0.277 0 0.4 0.02 0 0 0)")

if [[ "${QUICK:-0}" != "1" && "${DX_PINNING}" == "1" ]]; then
  echo "DX_PINNING=1: Glasner d0/W=0.277 at Δx/W0 = 0.6 and 0.8 (Fig. 1 stays the 3-case set)"
  DX_ROOTS+=("$(run_case 0.277 0 0.6 0.09)")
  DX_ROOTS+=("$(run_case 0.277 0 0.8 0.09)")
  for extra in \
    "${OUTROOT}/d0W_0.277_th0_dx0.4_notauEU_paperlike" \
    "${OUTROOT}/d0W_0.277_th0_dx0.5" \
    "${OUTROOT}/d0W_0.277_th0_dx1.0_dt0.09"; do
    if run_complete "${extra}"; then
      DX_ROOTS+=("${extra}")
    fi
  done
fi

if [[ "${QUICK:-0}" != "1" && "${GRID}" == "1" ]]; then
  echo "GRID=1: 45° Glasner check at Δx=W0, Δt=0.09 τ0"
  ROOTS+=("$(run_case 0.277 45 1.0 0.09)")
fi

if [[ "${QUICK:-0}" != "1" && "${DT_SCAN}" == "1" ]]; then
  echo "von Neumann (2D iso. diffusion): Δt/τ0 ≤ 0.1875 at Δx=W0; user cap 0.2"
  if run_complete "${OUTROOT}/d0W_0.277_th45_dx1.0_dt0.09"; then
    DT_LIST=(0.02 0.05 0.06 0.07 0.08 0.09)
    echo "reuse existing 45° Δt family (max Δt/τ0=0.09 with local e^u)"
  else
    DT_LIST=(0.02 0.05)
    for dtt in 0.06 0.07 0.08 0.09; do
      if probe_dt_stable "${dtt}"; then
        DT_LIST+=("${dtt}")
      else
        echo "Δt/τ0=${dtt} is unstable in the full scheme; not used"
        break
      fi
    done
  fi
  echo "45° Δt/τ0 values: ${DT_LIST[*]}"
  for dtt in "${DT_LIST[@]}"; do
    DT_ROOTS+=("$(run_case 0.277 45 1.0 "${dtt}")")
  done
fi

compare_args=("${ROOTS[@]}")
if [[ "${#DX_ROOTS[@]}" -gt 0 ]]; then
  compare_args+=(--dx-scan "${DX_ROOTS[@]}")
fi
if [[ "${#DT_ROOTS[@]}" -gt 0 ]]; then
  compare_args+=(--dt-scan "${DT_ROOTS[@]}")
fi
"${PY}" "${COMPARE}" "${compare_args[@]}" --out-dir "${OUTROOT}/figures"

echo "runs: ${ROOTS[*]}"
echo "dx-pinning: ${DX_ROOTS[@]+${DX_ROOTS[*]}}"
echo "dt-scan: ${DT_ROOTS[@]+${DT_ROOTS[*]}}"
echo "figures: ${OUTROOT}/figures"
