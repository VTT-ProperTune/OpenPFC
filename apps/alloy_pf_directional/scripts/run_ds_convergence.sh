#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Sequential laptop launcher for the single-grain Al-Cu directional cases:
#   - same physical box 6.40 μm × 0.80 μm
#   - planar seed at x = 0.20 μm (near left wall), one Gaussian bump
#   - T = Tl + G(x − xs − Vp t) Bridgman (cold solid left, hot liquid right; Tl on the initial solidus)
#   - y periodic (z periodic in 3D); no-flux on the left/right x walls
#   - Glasner, Ji isotropic FD, new dt = min(½ Δx²/(2 n D_L), ½ Δx²/(2 n W0²/τ0), 0.05 τ0)
#   - stop when φ > 0 on any right-boundary cell (t_end = 80 μs is only a cap)
#   - I/O PNG+VTK (4th arg of run_one; log defaults to 1/10 of that):
#       20 nm → 1000    10 nm → 5000    5 nm → 20000    2.5 nm → 50000
#     Edit those numbers on the run_one lines, or override:
#       --save-every N / --log-every N     (all cases)
#       OPENPFC_ALCU_SAVE_EVERY_<case>     (one case, e.g. OPENPFC_ALCU_SAVE_EVERY_w0_5nm_dx1=30000)
#       binary: --save-every N --log-every N
#
# Usage:
#   ./apps/alloy_pf_directional/scripts/run_ds_convergence.sh
#   ./apps/alloy_pf_directional/scripts/run_ds_convergence.sh --force
#   ./apps/alloy_pf_directional/scripts/run_ds_convergence.sh --only w0_5nm_dx1
#   NTHREADS=8 ./apps/alloy_pf_directional/scripts/run_ds_convergence.sh --out /Users/tptatu/Data/OpenPFC/alloy_pf_directional_ds/my_run
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="${ROOT}/builds/macos-cpu-release/apps/alloy_pf_directional/alloy_pf_directional_openmp"
OUTROOT="${OUTROOT:-/Users/tptatu/Data/OpenPFC/alloy_pf_directional_ds/convergence}"
#NTHREADS="${NTHREADS:-16}"
NTHREADS="${NTHREADS:-8}"
FORCE=0
ONLY=""
SAVE_EVERY_ALL=""
LOG_EVERY_ALL=""

usage() {
  cat <<EOF
Run the Al-Cu single-grain directional solidification matrix on this laptop.

Cases (same physical Lx, Ly, seed, G, Vp):
  w0_20nm_dx1     W0=20 nm   Δx/W0=1.0
  w0_10nm_dx1     W0=10 nm   Δx/W0=1.0
  w0_5nm_dx1      W0=5 nm    Δx/W0=1.0     (Δx-reference and W0-reference)
  w0_5nm_dx0.4    W0=5 nm    Δx/W0=0.4     (Δx convergence)
  w0_2.5nm_dx1    W0=2.5 nm  Δx/W0=1.0     (finest W0)

Stops when solid reaches the right wall. Outputs go under:
  ${OUTROOT}

PNG/VTK stride is the 4th argument on each run_one line below (log = 1/10 unless a 5th
argument or OPENPFC_ALCU_LOG_EVERY_<case> is set).

Options:
  --out DIR         output root (default: ${OUTROOT})
  --threads N       OpenMP threads (default: ${NTHREADS})
  --only NAME       run a single case name from the list above
  --save-every N    PNG+VTK stride for every case (overrides the table)
  --log-every N     fields.log stride for every case
  --force           overwrite a case that already has phi_final.raw
  --help            this text

Environment (optional overrides):
  OPENPFC_ALCU_QUIET  OPENPFC_ALCU_SKIP_PNG  OPENPFC_ALCU_SKIP_VTK  OPENPFC_ALCU_ISO
  OPENPFC_ALCU_SAVE_EVERY_<case>  OPENPFC_ALCU_LOG_EVERY_<case>
  OPENPFC_ALCU_LX  OPENPFC_ALCU_LY  OPENPFC_ALCU_TEND  OPENPFC_ALCU_SEED
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUTROOT="$2"
      shift 2
      ;;
    --threads)
      NTHREADS="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY_ALL="$2"
      shift 2
      ;;
    --log-every)
      LOG_EVERY_ALL="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${BIN}" ]]; then
  echo "missing binary: ${BIN}" >&2
  echo "build with: cmake --build ${ROOT}/builds/macos-cpu-release -j${NTHREADS} --target alloy_pf_directional_openmp" >&2
  exit 1
fi

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${NTHREADS}"
export OPENPFC_ALCU_QUIET="${OPENPFC_ALCU_QUIET:-1}"
export OPENPFC_ALCU_LX="${OPENPFC_ALCU_LX:-6.40e-6}"
export OPENPFC_ALCU_LY="${OPENPFC_ALCU_LY:-0.80e-6}"
export OPENPFC_ALCU_TEND="${OPENPFC_ALCU_TEND:-80.0e-6}"
export OPENPFC_ALCU_SEED="${OPENPFC_ALCU_SEED:-0.20e-6}"
export OPENPFC_ALCU_NGRANS=1
unset OPENPFC_ALCU_MAX_STEPS || true

mkdir -p "${OUTROOT}"
MASTER="${OUTROOT}/campaign.log"
{
  echo "===== campaign $(date) ====="
  echo "bin=${BIN}"
  echo "out=${OUTROOT}"
  echo "threads=${NTHREADS}"
  echo "Lx=${OPENPFC_ALCU_LX} Ly=${OPENPFC_ALCU_LY} t_end=${OPENPFC_ALCU_TEND} seed=${OPENPFC_ALCU_SEED}"
} | tee -a "${MASTER}"

run_one() {
  local name="$1"
  local w0="$2"
  local dxw="$3"
  local table_save="${4:-}"
  local table_log="${5:-}"
  local rc=0
  if [[ -n "${ONLY}" && "${ONLY}" != "${name}" ]]; then
    return 0
  fi
  local dest="${OUTROOT}/${name}"
  mkdir -p "${dest}"
  if [[ -f "${dest}/phi_final.raw" && ! -f "${dest}/abort.txt" && "${FORCE}" -eq 0 ]]; then
    echo "===== SKIP ${name} (phi_final.raw exists; pass --force to overwrite) =====" | tee -a "${MASTER}"
    return 0
  fi

  local case_save_var="OPENPFC_ALCU_SAVE_EVERY_${name}"
  local case_log_var="OPENPFC_ALCU_LOG_EVERY_${name}"
  local save=""
  local log=""
  if [[ -n "${!case_save_var+x}" ]]; then
    save="${!case_save_var}"
  elif [[ -n "${SAVE_EVERY_ALL}" ]]; then
    save="${SAVE_EVERY_ALL}"
  elif [[ -n "${table_save}" ]]; then
    save="${table_save}"
  fi
  if [[ -n "${!case_log_var+x}" ]]; then
    log="${!case_log_var}"
  elif [[ -n "${LOG_EVERY_ALL}" ]]; then
    log="${LOG_EVERY_ALL}"
  elif [[ -n "${table_log}" ]]; then
    log="${table_log}"
  fi

  local -a extra=()
  if [[ -n "${save}" ]]; then
    extra+=(--save-every "${save}")
  fi
  if [[ -n "${log}" ]]; then
    extra+=(--log-every "${log}")
  fi

  echo "===== START ${name} W0=${w0} dx/W0=${dxw} save_every=${save:-W0-default} log_every=${log:-save/10} $(date) =====" | tee -a "${MASTER}"
  set +e
  if [[ ${#extra[@]} -gt 0 ]]; then
    OPENPFC_ALCU_W0="${w0}" OPENPFC_ALCU_DXW="${dxw}" \
      "${BIN}" ds "${dest}" "${NTHREADS}" "${extra[@]}" 2>&1 | tee "${dest}/run.log" | tee -a "${MASTER}"
  else
    OPENPFC_ALCU_W0="${w0}" OPENPFC_ALCU_DXW="${dxw}" \
      "${BIN}" ds "${dest}" "${NTHREADS}" 2>&1 | tee "${dest}/run.log" | tee -a "${MASTER}"
  fi
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "===== ABORT ${name} exit=${rc} $(date) — see ${dest}/abort.txt =====" | tee -a "${MASTER}"
    exit "${rc}"
  fi
  echo "===== DONE ${name} $(date) =====" | tee -a "${MASTER}"
}

cd "${ROOT}"
#                    name            W0     dx/W0  PNG+VTK  [log]
run_one "w0_20nm_dx1"  20e-9  1.0    1000
run_one "w0_10nm_dx1"  10e-9  1.0    5000
run_one "w0_5nm_dx1"    5e-9  1.0   20000
run_one "w0_5nm_dx0.4"  5e-9  0.4   20000
run_one "w0_2.5nm_dx1" 2.5e-9  1.0   50000

echo "===== all requested ds runs complete $(date) =====" | tee -a "${MASTER}"
