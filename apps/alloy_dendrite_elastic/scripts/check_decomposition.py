#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare two field-snapshot directories written at different rank counts.

Why this is a script and not `cmp`
----------------------------------
The finite-difference fields of this application *are* bitwise
rank-independent: the stencil is evaluated in the same order on every rank
and the halo carries exact copies, so `phi`, `U` and `theta` from a one-rank
run are byte-for-byte the files from a four-rank run.  The elastic fields are
not, and cannot be: the Green-operator solve is an FFT, and HeFFTe composes a
different sequence of transforms and transposes for each process grid.  The
result is a different rounding path to the same answer.

So `cmp` reports every elastic file as differing and says nothing about
whether the coupling is correct.  What matters is the *size* of the
difference, and the distinction between the two regimes is the point:

    round-off       ~ 1e-14 relative, growing slowly with step count
    a real bug      ~ 1e-3 and up, or localised at a rank boundary

The second line of defence is the *location*: a decomposition bug puts its
error on the subdomain seams, so this script reports where the maximum sits
as well as how big it is.

Usage
-----
    check_decomposition.py REF_DIR TEST_DIR [--rtol 1e-10]

Both directories must contain a manifest written by the same `--run-id`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_manifest(d: Path) -> dict:
    hits = sorted(d.glob("*_manifest.json"))
    if not hits:
        raise SystemExit(f"{d}: no *_manifest.json (was --fields-dir set?)")
    if len(hits) > 1:
        raise SystemExit(f"{d}: {len(hits)} manifests; expected one run per directory")
    return json.loads(hits[0].read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-10,
                    help="fail above this relative max-norm difference (1e-10)")
    args = ap.parse_args()

    ma, mb = load_manifest(args.ref), load_manifest(args.test)
    for key in ("run_id", "nx", "ny", "nz", "fields"):
        if ma[key] != mb[key]:
            raise SystemExit(f"manifests disagree on {key!r}: {ma[key]} vs {mb[key]}")
    shape = (ma["nx"], ma["ny"], ma["nz"])
    nsnap = min(len(ma["times"]), len(mb["times"]))
    if nsnap == 0:
        raise SystemExit("no snapshots in common")

    worst = 0.0
    rows = []
    for name in ma["fields"]:
        for idx in range(nsnap):
            pat = ma["pattern"].replace("{field}", name)
            fn = pat.replace("{index:04d}", f"{idx:04d}")
            a = np.fromfile(args.ref / fn).reshape(shape, order="F")
            b = np.fromfile(args.test / fn).reshape(shape, order="F")
            d = np.abs(a - b)
            scale = max(float(np.abs(a).max()), 1e-300)
            rel = float(d.max()) / scale
            worst = max(worst, rel)
            at = np.unravel_index(int(np.argmax(d)), shape)
            rows.append((name, idx, rel, at, bool(np.array_equal(a, b))))

    print(f"{'field':12s} {'snap':>4s} {'rel max-norm':>14s} {'argmax (i,j,k)':>18s}  bitwise")
    for name, idx, rel, at, same in rows:
        print(f"{name:12s} {idx:4d} {rel:14.3e} {str(at):>18s}  {'yes' if same else 'no'}")
    print(f"\nworst relative difference: {worst:.3e}  (tolerance {args.rtol:.1e})")
    if worst > args.rtol:
        print("FAIL: larger than round-off. Check the subdomain seams in argmax.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
