#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check polar-defined owned fields using an independent Cartesian polynomial."""

import argparse
import csv
import math
from pathlib import Path
import tempfile

from check_foundations import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.bin_dir.resolve() / "17_custom_coordinate_system"
    launch = [args.launcher, args.rank_flag, str(args.ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(prefix="coordinates-", dir=args.output) as name:
        root = Path(name).resolve()
        for n, harmonic, width in (
            (16, 0, 0.4),
            (16, 3, 0.3),
            (9, 1, 0.25),
            (16, 7, 0.5),
        ):
            directory = root / f"{n}-{harmonic}"
            run(
                binary,
                directory,
                launch,
                parameters={"n": n, "harmonic": harmonic, "width": width},
            )
            paths = sorted((directory / "results").glob("polar_rank_*.csv"))
            assert len(paths) == args.ranks
            seen = set()
            origins = 0
            for path in paths:
                with path.open() as file:
                    data = list(csv.DictReader(file))
                assert data
                for row in data:
                    i, j = int(row["i"]), int(row["j"])
                    assert 0 <= i < n and 0 <= j < n and (i, j) not in seen
                    seen.add((i, j))
                    x, y = -1 + 2 * i / n, -1 + 2 * j / n
                    assert abs(float(row["x"]) - x) < 1e-14
                    assert abs(float(row["y"]) - y) < 1e-14
                    r = math.hypot(x, y)
                    assert abs(float(row["r"]) - r) < 1e-14
                    if r == 0:
                        origins += 1
                        assert math.isnan(float(row["theta"]))
                        assert float(row["u"]) == (1 if harmonic == 0 else 0)
                    else:
                        theta = float(row["theta"])
                        assert abs(theta - math.atan2(y, x)) < 1e-14
                        assert abs(r * math.cos(theta) - x) < 1e-14
                        assert abs(r * math.sin(theta) - y) < 1e-14
                    expected = ((complex(x, y) / width) ** harmonic).real * math.exp(
                        -(x * x + y * y) / (2 * width * width)
                    )
                    assert abs(float(row["u"]) - expected) < 2e-12 * max(
                        1, abs(expected)
                    )
            assert len(seen) == n * n and origins == (1 if n % 2 == 0 else 0)
        for parameters in ({"n": 7}, {"harmonic": -1}, {"harmonic": 9}, {"width": 0}):
            run(binary, root / "bad", launch, parameters=parameters, success=False)
    print(
        f"Polar-defined fields: complete ownership, origin limits and Cartesian references passed ({args.ranks} rank(s))."
    )


if __name__ == "__main__":
    main()
