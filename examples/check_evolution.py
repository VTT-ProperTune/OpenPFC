#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent complete-field checks for waves, KdV and Allen–Cahn."""
from __future__ import annotations

import argparse
import cmath
import math
from pathlib import Path
import tempfile

from check_foundations import run, table
from check_spectra import lagrange_second


def snapshots(path, kind=None):
    data = table(path)
    if kind:
        data = [row for row in data if row["kind"] == kind]
    return {
        step: [float(row["u"]) for row in data if int(row["step"]) == step]
        for step in sorted({int(row["step"]) for row in data})
    }


def close(actual, expected, tolerance=2e-10):
    error = max(abs(a - b) for a, b in zip(actual, expected))
    assert len(actual) == len(expected) and error < tolerance, error


def apply(matrix, values):
    return [sum(a * b for a, b in zip(row, values)) for row in matrix]


def check_line(binary, root):
    folder = root / "line-reference"
    n, dt = 12, 0.001
    run(binary, folder, parameters={"n": n, "dt": dt, "end": 0.03, "saveat": 0.01})
    saved = snapshots(folder / "results/wave.csv")
    x, d2 = lagrange_second(n)
    u = [math.exp(-200 * p * p) for p in x]
    old = [math.exp(-200 * (p - dt) ** 2) for p in x]
    close(saved[0], u)
    for step in range(1, 31):
        acceleration = apply(d2, u)
        acceleration[0] = acceleration[-1] = 0
        u, old = [2 * a - b + dt * dt * c for a, b, c in zip(u, old, acceleration)], u
        if step in saved:
            close(saved[step], u)
    errors = []
    for dt in (0.004, 0.002):
        folder = root / f"line-mode-{dt}"
        run(
            binary,
            folder,
            parameters={
                "n": 32,
                "dt": dt,
                "end": 0.4,
                "saveat": 0.4,
                "initial": "mode",
                "startup": "taylor",
            },
        )
        data = table(folder / "results/wave.csv")
        final = [r for r in data if float(r["time"]) > 0.39]
        errors.append(
            max(
                abs(
                    float(r["u"])
                    - math.cos(math.pi * float(r["x"]) / 2)
                    * math.cos(math.pi * 0.4 / 2)
                )
                for r in final
            )
        )
    assert 3.9 < errors[0] / errors[1] < 4.1, errors
    run(binary, root / "bad-line", parameters={"n": 32, "dt": 1}, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--case", choices=["chebyshev_wave_line"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bin_dir = args.bin_dir.resolve()
    ranks = args.ranks
    launch = [args.launcher, args.rank_flag, str(ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(
        prefix="example-check-", dir=args.output
    ) as temporary:
        base = Path(temporary).resolve()
        if args.case in (None, "chebyshev_wave_line"):
            root = base / "chebyshev_wave_line"
            root.mkdir()
            check_line(bin_dir / "chebyshev_wave_line", root)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
