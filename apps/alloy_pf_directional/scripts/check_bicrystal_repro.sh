#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Bit-reproduce a noisy twin of the advertised 12×3.2 µm W=10 nm box
# (CLI `benchmark` is that box with noise off, full length).
# Same G, Vp, ±30°, dt=0.2 τ0, with F0=1e-3 and seed=1. Caps Euler steps
# so it finishes on a laptop; 400 steps is still the two seeds, not the gold.
#
#   ./apps/alloy_pf_directional/scripts/check_bicrystal_repro.sh
#   STEPS=400 NTHREADS=8 ./apps/alloy_pf_directional/scripts/check_bicrystal_repro.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
BUILD="${BUILD:-${OPENPFC_BUILD_DIR:-${ROOT}/builds/macos-cpu-release}}"
BIN="${BIN:-${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_openmp}"
OUT="${OUT:-${ROOT}/results/alloy_pf_directional_bicrystal_repro}"
STEPS="${STEPS:-400}"
NTHREADS="${NTHREADS:-${OMP_NUM_THREADS:-8}}"

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" >&2
  exit 1
fi

export OPENPFC_ALCU_W0=10e-9
export OPENPFC_ALCU_DXW=1.0
export OPENPFC_ALCU_G=3.0e6
export OPENPFC_ALCU_VP=0.4
export OPENPFC_ALCU_LX=12.0e-6
export OPENPFC_ALCU_LY=3.20e-6
export OPENPFC_ALCU_TEND=120.0e-6
export OPENPFC_ALCU_SEED=0.20e-6
export OPENPFC_ALCU_NGRANS=2
export OPENPFC_ALCU_THETA=30
export OPENPFC_ALCU_NOISE=1e-3
export OPENPFC_ALCU_NOISE_SEED=1
export OPENPFC_ALCU_DT_OVER_TAU=0.2
export OPENPFC_ALCU_MAX_STEPS="${STEPS}"
export OPENPFC_ALCU_WINDOW=0
export OPENPFC_ALCU_STOP_FAR_C=0
export OPENPFC_ALCU_STOP_RIGHT=0
export OPENPFC_ALCU_PERIODIC_Y=1
export OPENPFC_ALCU_SKIP_PNG=1
export OPENPFC_ALCU_SKIP_VTK=1
export OPENPFC_ALCU_QUIET=1
unset OPENPFC_ALCU_WARMUP OPENPFC_ALCU_TIMED_STEPS || true
if [[ "$(uname -s)" == Darwin ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
fi
export OMP_NUM_THREADS="${NTHREADS}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

rm -rf "${OUT}/a" "${OUT}/b"
mkdir -p "${OUT}/a" "${OUT}/b"

echo "== bicrystal W=10 nm noise  STEPS=${STEPS}  nthreads=${NTHREADS}  A =="
"${BIN}" bicrystal "${OUT}/a" "${NTHREADS}" | tee "${OUT}/a.log"
echo "== B =="
"${BIN}" bicrystal "${OUT}/b" "${NTHREADS}" | tee "${OUT}/b.log"

for f in phi_final.raw phi2_final.raw c_final.raw; do
  cmp "${OUT}/a/${f}" "${OUT}/b/${f}"
done
echo "PASS: bit-identical φ1, φ2, c"

python3 - "${OUT}/a.log" "${OUT}/b.log" <<'PY'
import pathlib, re, sys
skip = {"wall_loop_s", "time_per_step_s"}

def grab(p):
    t = pathlib.Path(p).read_text()
    m = re.search(r"^ALCU_VERIFY (.*)$", t, re.M)
    if not m:
        raise SystemExit(f"no ALCU_VERIFY in {p}")
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d

a, b = grab(sys.argv[1]), grab(sys.argv[2])
ok = True
for k in sorted(set(a) | set(b)):
    if k in skip:
        continue
    if a.get(k) != b.get(k):
        print(f"FAIL: {k} A={a.get(k)} B={b.get(k)}")
        ok = False
    else:
        print(f"  {k}={a.get(k)}")
if not ok:
    sys.exit(2)
print("PASS: ALCU_VERIFY matches (excluding wall times)")
print("OpenMP A", {k: a.get(k) for k in ("n_steps_done", "mass1", "x_tip", "sum_phi", "sum_c", "nthreads")})
PY
