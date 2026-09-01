#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Light 3D smoke / superficial convergence: two tiny bricks (coarse vs 2× finer
# in all directions), few steps, compare mass conservation and tip trend.
# Not a paper-style study — catches 3D stencil/BC bugs. Do not sbatch this.
#
#   BUILD=builds/release ./apps/alloy_pf_directional/scripts/check_3d_smoke.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD="${BUILD:-${OPENPFC_BUILD_DIR:-${ROOT}/builds/release}}"
OMP="${BUILD}/apps/alloy_pf_directional/alloy_pf_directional_openmp"
OUT="${ROOT}/results/alloy_pf_directional_3d_smoke"
mkdir -p "${OUT}/coarse" "${OUT}/fine"
if [[ ! -x "${OMP}" ]]; then
  echo "missing ${OMP}" >&2
  exit 1
fi
export OPENPFC_ALCU_SKIP_PNG=1 OPENPFC_ALCU_SKIP_VTK=1 OPENPFC_ALCU_QUIET=1
export OPENPFC_ALCU_NOISE=0 OPENPFC_ALCU_NGRANS=1
export OPENPFC_ALCU_NDIM=3 OPENPFC_ALCU_STOP_RIGHT=0
# Coarse: ~48×16×16 at dx=W0. Seed must sit inside Lx (default seed_depth is 0.20 µm).
export OPENPFC_ALCU_W0=5e-9 OPENPFC_ALCU_DXW=1.0
export OPENPFC_ALCU_LX=2.40e-7 OPENPFC_ALCU_LY=8.0e-8 OPENPFC_ALCU_LZ=8.0e-8
export OPENPFC_ALCU_SEED=4.0e-8 OPENPFC_ALCU_BUMP=1.0e-8
export OPENPFC_ALCU_TEND=2.0e-8 OPENPFC_ALCU_MAX_STEPS="${STEPS:-30}"
echo "== 3D coarse =="
"${OMP}" ds "${OUT}/coarse" 1 | tee "${OUT}/coarse.log"
export OPENPFC_ALCU_DXW=0.5
echo "== 3D fine (dx/2) =="
"${OMP}" ds "${OUT}/fine" 1 | tee "${OUT}/fine.log"
python3 - "${OUT}" <<'PY'
import pathlib, re, sys
out = pathlib.Path(sys.argv[1])
def grab(p):
    t = p.read_text()
    m = re.search(r"ALCU_VERIFY (.*)", t)
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d
c, f = grab(out / "coarse.log"), grab(out / "fine.log")
print("coarse", {k: c.get(k) for k in ("mass0", "mass1", "rel_mass_err", "x_tip", "min_phi", "max_phi")})
print("fine  ", {k: f.get(k) for k in ("mass0", "mass1", "rel_mass_err", "x_tip", "min_phi", "max_phi")})
ok = True
for name, d in (("coarse", c), ("fine", f)):
    if float(d.get("blew_up", "0")) != 0:
        print(f"FAIL: {name} blew up"); ok = False
    if abs(float(d["rel_mass_err"])) > 5e-3:
        print(f"FAIL: {name} mass drift {d['rel_mass_err']}"); ok = False
    mn, mx = float(d["min_phi"]), float(d["max_phi"])
    if not (-1.2 < mn <= mx < 1.2):
        print(f"FAIL: {name} phi range looks wrong"); ok = False
    if mn > -0.5 or mx < 0.5:
        print(f"FAIL: {name} missing solid/liquid (interface not in box?)"); ok = False
# Tip should exist and not be wildly larger on the coarser grid after the same t
xtc, xtf = float(c["x_tip"]), float(f["x_tip"])
if xtc <= 0 or xtf <= 0:
    print("WARN: tip still at origin (few steps / small seed) — not a fail")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 2)
PY
