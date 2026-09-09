#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gradient-elasticity misfitting-inclusion size sweep (issue #117, science case A).

Runs the `gradient_elasticity` binary once per inclusion radius `R` (fixed
internal length `ell`, fixed Lame parameters and eigenstrain amplitude, box
size L = box_to_radius * max(R, ell) so the periodic-image ratio stays fixed
relative to *both* controlling length scales, not just R -- ell can be
larger than R at the small-R end of the sweep), parses the
`GRADIENT_ELASTICITY_SUMMARY` line each run prints on rank 0, and writes one
aggregated CSV: R, ell, R/ell, box L, peak |hydrostatic stress|, peak von
Mises stress, total elastic energy.

Physical direction (see the app README "Classical and clamped closed forms"
section): for this loading (a smooth, *finite* inclusion, no classical
singularity), peak stress *decreases* as R/ell grows, from a closed-form
"clamped" bound (R/ell -> 0, the gradient penalty suppresses essentially all
elastic relaxation) down to the closed-form classical "relaxed" bound
(R/ell -> infinity). This is the opposite direction from the familiar
"gradient elasticity regularizes a classical singularity" story, which
applies to sharp/singular defect loadings, not this smooth inclusion.

This is a *science preset* (roadmap #120): it is meant to be run by hand on
real allocations to produce the size-effect table for the report/PR, not to
run inside CI. `apps/gradient_elasticity/tests/test_gradient_elasticity.cpp`
carries the fast, CI-safe analytical/monotonicity/box-size regression tests
that check the same physics on tiny grids.

Example (see apps/gradient_elasticity/README.md for the full recipe):

    python3 size_sweep.py --binary build/apps/gradient_elasticity/gradient_elasticity \\
        --outdir results/gradient_elasticity/size_sweep --csv size_sweep.csv \\
        --launcher 'srun --overlap -n 1'
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
from pathlib import Path

SUMMARY_RE = re.compile(
    r"GRADIENT_ELASTICITY_SUMMARY ell=(?P<ell>\S+) ell4=(?P<ell4>\S+) "
    r"order=(?P<order>\S+) peak_abs_hydrostatic_stress=(?P<peak_hydro>\S+) "
    r"peak_von_mises_stress=(?P<peak_vm>\S+) total_elastic_energy=(?P<energy>\S+)"
)


def make_settings(*, N, R, w, ell, mu, lam, eps0, order, x0=None, y0=None,
                   line_profile=None):
    x0 = N / 2.0 if x0 is None else x0
    y0 = N / 2.0 if y0 is None else y0
    settings = {
        "model": {
            "name": "gradient_elasticity",
            "params": {"mu": mu, "lambda": lam, "ell": ell, "eps0": eps0,
                       "order": order},
        },
        "domain": {"Lx": N, "Ly": N, "Lz": 1, "dx": 1.0, "dy": 1.0, "dz": 1.0,
                  "origin": "corner"},
        "timestepping": {"t0": 0.0, "t1": 1.0, "dt": 1.0, "saveat": -1.0},
        "initial_conditions": [
            {"target": "g", "type": "circular_inclusion", "g0": 0.0,
             "amplitude": 1.0, "radius": R, "interface_width": w, "x0": x0,
             "y0": y0}
        ],
    }
    if line_profile is not None:
        settings["line_profile"] = {"path": line_profile, "x0": x0, "y0": y0}
    return settings


def run_case(binary, launcher, workdir, settings):
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / "case.json"
    input_path.write_text(json.dumps(settings, indent=2))
    cmd = [*launcher, str(binary), str(input_path)]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                            timeout=300, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed (rc={result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    match = SUMMARY_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"no GRADIENT_ELASTICITY_SUMMARY line in output of "
                           f"{' '.join(cmd)}:\n{result.stdout}")
    return {k: float(v) for k, v in match.groupdict().items()}


def interface_width(R, dx=1.0):
    """Smoothed-inclusion interface half-width: max(1 grid spacing, R/8)."""
    return max(dx, R / 8.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--ell", type=float, default=8.0,
                        help="Fixed Helmholtz internal length (default: 8.0).")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--eps0", type=float, default=0.01)
    parser.add_argument("--box-to-radius", type=float, default=16.0,
                        help="L = box_to_radius * max(R, ell) (default: 16; "
                             "the box_size unit test measured ~4-5%% image "
                             "contamination at ratio 8 and <2%% doubling from "
                             "16 to 32, so 16 is the default here). Scaling "
                             "by max(R, ell), not just R, matters once ell "
                             "is comparable to or larger than R.")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
                        help="Inclusion radii to sweep (default spans "
                             "R/ell = 0.5 .. 16 at ell=8).")
    parser.add_argument("--launcher", type=str, default="",
                        help="Command prefix as one shell-quoted string, e.g. "
                             "--launcher 'srun --overlap -n 1' (argparse cannot "
                             "collect a prefix containing its own -n/--flags as "
                             "nargs='*', so this takes a single string, split "
                             "with shlex).")
    parser.add_argument("--line-profile-radii", type=float, nargs="*", default=[],
                        help="Also write a line-profile CSV for these radii.")
    parser.add_argument("--box-check-radius", type=float, default=None,
                        help="If set, also runs this radius at the sweep's "
                             "box-to-radius ratio and at double that ratio, "
                             "and reports the relative peak-stress change "
                             "(periodic-image control evidence).")
    args = parser.parse_args()
    launcher = shlex.split(args.launcher)

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for R in args.radii:
        N = int(round(args.box_to_radius * max(R, args.ell)))
        w = interface_width(R)
        line_profile = None
        if R in args.line_profile_radii:
            line_profile = str((args.outdir / f"line_profile_R{R:g}.csv").resolve())
        settings = make_settings(N=N, R=R, w=w, ell=args.ell, mu=args.mu,
                                 lam=args.lam, eps0=args.eps0, order=4,
                                 line_profile=line_profile)
        summary = run_case(args.binary, launcher, args.outdir / f"R{R:g}",
                           settings)
        rows.append({"R": R, "ell": args.ell, "R_over_ell": R / args.ell, "N": N,
                    "box_L": N, "box_L_over_R": N / R,
                    "interface_width": w, **summary})
        print(f"R={R:g} R/ell={R/args.ell:.3g} peak_hydro={summary['peak_hydro']:.6g} "
             f"peak_vm={summary['peak_vm']:.6g} energy={summary['energy']:.6g}")

    with args.csv.open("w", newline="") as stream:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv} ({len(rows)} rows)")

    if args.box_check_radius is not None:
        R = args.box_check_radius
        w = interface_width(R)
        base = max(R, args.ell)
        small_N = int(round(args.box_to_radius * base))
        large_N = int(round(2.0 * args.box_to_radius * base))
        small = run_case(args.binary, launcher, args.outdir / "box_check_small",
                         make_settings(N=small_N, R=R, w=w, ell=args.ell, mu=args.mu,
                                      lam=args.lam, eps0=args.eps0, order=4))
        large = run_case(args.binary, launcher, args.outdir / "box_check_large",
                         make_settings(N=large_N, R=R, w=w, ell=args.ell, mu=args.mu,
                                      lam=args.lam, eps0=args.eps0, order=4))
        rel_hydro = abs(large["peak_hydro"] - small["peak_hydro"]) / small["peak_hydro"]
        rel_vm = abs(large["peak_vm"] - small["peak_vm"]) / small["peak_vm"]
        print(f"box check R={R:g} ell={args.ell:g}: L/max(R,ell) "
             f"{args.box_to_radius:g} -> {2*args.box_to_radius:g} (N={small_N} -> "
             f"{large_N}): peak_hydro rel. change={rel_hydro:.3%}, peak_vm rel. "
             f"change={rel_vm:.3%}")


if __name__ == "__main__":
    main()
