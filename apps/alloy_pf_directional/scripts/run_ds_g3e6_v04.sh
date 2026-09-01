#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Sequential laptop FTA directional series:
#   G = 3e6 K/m, Vp = 0.4 m/s, Ly = 0.80 μm periodic, static Lx = 12 μm
#   No moving window (integer shifts are an extra discrete advection).
#   e^u τ correction (engine default), α = 0.38 (engine default)
#   noise off; stop when any right-face liquid pixel leaves c_∞ (1%)
#   W0 = 40, 20, 10 nm on the laptop; 5 nm and 2.5 nm on LUMI-C
#   (see lumi_submit_g3e6_v04.sh). 2.5 nm is a long OpenMP job.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="${ROOT}/builds/macos-cpu-release/apps/alloy_pf_directional/alloy_pf_directional_openmp"
OUTROOT="${OUTROOT:-${ROOT}/results/alloy_pf_directional_ds/G3e6_V0.4_static}"
NTHREADS="${NTHREADS:-8}"
FORCE=0
ONLY=""

usage() {
  cat <<EOF
Run the G=3e6 K/m, Vp=0.4 m/s Al-Cu FTA W0 series on this laptop.

  ${OUTROOT}

Options:
  --out DIR      output root
  --threads N    OpenMP threads (default ${NTHREADS})
  --only NAME    w0_40nm_dx1 | w0_20nm_dx1 | w0_10nm_dx1 | w0_5nm_dx1
  --force        overwrite a finished case
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUTROOT="$2"; shift 2 ;;
    --threads) NTHREADS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
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
export OPENPFC_ALCU_G=3.0e6
export OPENPFC_ALCU_VP=0.4
export OPENPFC_ALCU_LX=12.0e-6
export OPENPFC_ALCU_LY=0.80e-6
export OPENPFC_ALCU_TEND=120.0e-6
export OPENPFC_ALCU_SEED=0.20e-6
export OPENPFC_ALCU_NGRANS=1
export OPENPFC_ALCU_NOISE=0
export OPENPFC_ALCU_STOP_RIGHT=0
export OPENPFC_ALCU_STOP_FAR_C=1
export OPENPFC_ALCU_PERIODIC_Y=1
export OPENPFC_ALCU_SKIP_VTK=1
export OPENPFC_ALCU_WINDOW=0
unset OPENPFC_ALCU_WINDOW_NX || true
unset OPENPFC_ALCU_WINDOW_LEFT || true
unset OPENPFC_ALCU_WINDOW_RIGHT || true
unset OPENPFC_ALCU_MAX_STEPS || true

mkdir -p "${OUTROOT}"
MASTER="${OUTROOT}/campaign.log"
{
  echo "===== campaign $(date) ====="
  echo "bin=${BIN}"
  echo "out=${OUTROOT}"
  echo "threads=${NTHREADS}"
  echo "G=${OPENPFC_ALCU_G} Vp=${OPENPFC_ALCU_VP} Lx=${OPENPFC_ALCU_LX} Ly=${OPENPFC_ALCU_LY} t_end=${OPENPFC_ALCU_TEND}"
} | tee -a "${MASTER}"

run_one() {
  local name="$1"
  local w0="$2"
  local dxw="$3"
  local table_save="${4:-}"
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

  echo "===== START ${name} W0=${w0} dx/W0=${dxw} static-box $(date) =====" | tee -a "${MASTER}"
  set +e
  OPENPFC_ALCU_W0="${w0}" OPENPFC_ALCU_DXW="${dxw}" \
    "${BIN}" ds "${dest}" "${NTHREADS}" --save-every "${table_save}" 2>&1 \
    | tee "${dest}/run.log" | tee -a "${MASTER}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "===== ABORT ${name} exit=${rc} $(date) — see ${dest}/abort.txt =====" | tee -a "${MASTER}"
    exit "${rc}"
  fi
  echo "===== DONE ${name} $(date) =====" | tee -a "${MASTER}"
}

cd "${ROOT}"
run_one "w0_40nm_dx1"  40e-9  1.0     500
run_one "w0_20nm_dx1"  20e-9  1.0    1000
run_one "w0_10nm_dx1"  10e-9  1.0    5000
run_one "w0_5nm_dx1"    5e-9  1.0   20000

echo "===== all requested ds runs complete $(date) =====" | tee -a "${MASTER}"
