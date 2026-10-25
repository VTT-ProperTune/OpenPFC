#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check transport against independently traced characteristics and RK stability polynomials."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import tempfile

from check_foundations import run


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def characteristic(x, t, variable):
    """Integrate backward in time independently of the C++ phase transformation."""
    if not variable:
        return x - t

    def rate(value):
        return -(0.2 + math.sin(value - 1) ** 2)

    h = t / 1000
    for _ in range(1000):
        k1 = rate(x)
        k2 = rate(x + h * k1 / 2)
        k3 = rate(x + h * k2 / 2)
        k4 = rate(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def check_transport(binary, root, launch):
    for variable in (False, True):
        errors = []
        for steps, dt in ((50, 0.002), (100, 0.001)):
            directory = root / f"transport-{variable}-{steps}"
            run(
                binary,
                directory,
                launch,
                {
                    "n": 32,
                    "steps": steps,
                    "dt": dt,
                    "coefficient": "variable" if variable else "constant",
                    "initial": "cosine",
                },
            )
            samples = [
                row
                for path in (directory / "results").glob("transport_rank_*.csv")
                for row in rows(path)
                if int(row["step"]) == steps
            ]
            assert len(samples) == 32
            error = 0
            for row in samples:
                x, t, value = (float(row[key]) for key in ("x", "time", "u"))
                assert math.isfinite(value)
                expected = math.cos(characteristic(x, t, variable))
                assert abs(float(row["exact"]) - expected) < 2e-12
                error = max(error, abs(value - expected))
            assert 0 < error < 1e-4
            errors.append(error)
        assert errors[1] < errors[0] / 2.5, (variable, errors)
    run(binary, root / "unstable-transport", launch, {"n": 32, "dt": 1}, success=False)
    run(
        binary,
        root / "book-startup",
        launch,
        {"n": 128, "dt": 0.0125, "steps": 2, "startup": "book"},
    )


def check_stability(binary, directory):
    run(binary, directory)
    grid = rows(directory / "results/rk.csv")
    assert len(grid) == 141**2
    for row in grid[::37]:
        z = complex(float(row["real"]), float(row["imag"]))
        for order in range(1, 5):
            expected = abs(sum(z**k / math.factorial(k) for k in range(order + 1)))
            assert abs(float(row[f"rk{order}"]) - expected) < 3e-13
    boundary = rows(directory / "results/multistep.csv")
    for family, order, expected in (
        ("AB", 1, -2),
        ("AB", 2, -1),
        ("AB", 3, -6 / 11),
        ("AM", 3, -6),
        ("AM", 4, -3),
        ("BDF", 1, 2),
        ("BDF", 6, sum(2**k / k for k in range(1, 7))),
    ):
        sample = next(
            row
            for row in boundary
            if row["family"] == family
            and int(row["order"]) == order
            and abs(float(row["theta"]) - math.pi) < 1e-12
        )
        assert abs(float(sample["real"]) - expected) < 1e-11
        assert abs(float(sample["imag"])) < 1e-11
    assert len({(row["family"], row["order"]) for row in boundary}) == 13


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--case", choices=["variable_transport", "stability_regions"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bin_dir = args.bin_dir.resolve()
    ranks = args.ranks
    launch = [args.launcher, args.rank_flag, str(ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(
        prefix="example-check-", dir=args.output
    ) as temporary:
        base = Path(temporary).resolve()
        if args.case in (None, "variable_transport"):
            root = base / "variable_transport"
            root.mkdir()
            check_transport(bin_dir / "variable_transport", root, launch)
        if args.case in (None, "stability_regions"):
            root = base / "stability_regions"
            root.mkdir()
            check_stability(bin_dir / "stability_regions", root)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
