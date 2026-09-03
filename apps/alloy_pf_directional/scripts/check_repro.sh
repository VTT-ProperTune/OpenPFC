#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Frozen last-bit CI (CLI `repro`), not the morphology product:
#   G=3e6 K/m, Vp=0.4 m/s, ±30°, F0=1e-3, seed=1, dt=0.2 τ0, 128×64, 40 steps.
# Runs twice and requires bit-identical φ1, φ2, c. Then checks ALCU_VERIFY
# against data/repro_alcu_verify.txt (wall times excluded).
#
#   ./apps/alloy_pf_directional/scripts/check_repro.sh
#   BUILD=builds/macos-cpu-release ./apps/alloy_pf_directional/scripts/check_repro.sh
#   check_repro.sh /path/to/alloy_pf_directional_openmp /tmp/out [nthreads]
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
BUILD="${BUILD:-${OPENPFC_BUILD_DIR:-${ROOT}/builds/macos-cpu-release}}"
BIN="${1:-${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_openmp}"
OUT="${2:-${ROOT}/results/alloy_pf_directional/repro}"
NTHREADS="${3:-${OMP_NUM_THREADS:-1}}"
EXPECT="${HERE}/../data/repro_alcu_verify.txt"

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" >&2
  exit 1
fi

# Leftover campaign env must not change the frozen case.
unset OPENPFC_ALCU_W0 OPENPFC_ALCU_DXW OPENPFC_ALCU_LX OPENPFC_ALCU_LY OPENPFC_ALCU_LZ || true
unset OPENPFC_ALCU_G OPENPFC_ALCU_VP OPENPFC_ALCU_TEND OPENPFC_ALCU_SEED || true
unset OPENPFC_ALCU_NGRANS OPENPFC_ALCU_THETA OPENPFC_ALCU_NOISE OPENPFC_ALCU_NOISE_SEED || true
unset OPENPFC_ALCU_DT_OVER_TAU OPENPFC_ALCU_MAX_STEPS OPENPFC_ALCU_WINDOW || true
unset OPENPFC_ALCU_WARMUP OPENPFC_ALCU_TIMED_STEPS OPENPFC_ALCU_OMEGA || true
unset OPENPFC_ALCU_NY OPENPFC_ALCU_NZ OPENPFC_ALCU_NDIM || true
export OPENPFC_ALCU_SKIP_PNG=1 OPENPFC_ALCU_SKIP_VTK=1 OPENPFC_ALCU_QUIET=1
if [[ "$(uname -s)" == Darwin ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${HOME}/opt/heffte/2.4.1-cpu/lib:/opt/homebrew/opt/fftw/lib:${DYLD_LIBRARY_PATH:-}"
fi
export OMP_NUM_THREADS="${NTHREADS}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

rm -rf "${OUT}/a" "${OUT}/b"
mkdir -p "${OUT}/a" "${OUT}/b"

echo "== repro A  nthreads=${NTHREADS} =="
"${BIN}" repro "${OUT}/a" "${NTHREADS}" | tee "${OUT}/a.log"
echo "== repro B =="
"${BIN}" repro "${OUT}/b" "${NTHREADS}" | tee "${OUT}/b.log"

for f in phi_final.raw phi2_final.raw c_final.raw; do
  cmp "${OUT}/a/${f}" "${OUT}/b/${f}"
done
echo "PASS: bit-identical φ1, φ2, c (two runs, seed=1)"

python3 - "${OUT}/a.log" "${EXPECT}" <<'PY'
import pathlib, re, sys
log, expect = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
line = ""
for raw in log.read_text().splitlines():
    if raw.startswith("ALCU_VERIFY "):
        line = raw
        break
if not line:
    print("FAIL: no ALCU_VERIFY"); sys.exit(1)
got = {}
for tok in line.split()[1:]:
    if "=" in tok:
        k, v = tok.split("=", 1)
        got[k] = v
skip = {"wall_loop_s", "time_per_step_s", "nthreads"}
want = {}
for raw in expect.read_text().splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        continue
    k, v = raw.split("=", 1)
    want[k] = v
ok = True
for k, v in want.items():
    if k in skip:
        continue
    if k not in got:
        print(f"FAIL: missing {k}"); ok = False
        continue
    if got[k] != v:
        print(f"FAIL: {k} got={got[k]} expected={v}")
        ok = False
    else:
        print(f"  {k}={got[k]}")
if not ok:
    sys.exit(2)
print("PASS: ALCU_VERIFY matches committed checksums")
PY
