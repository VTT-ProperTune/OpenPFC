#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the actual generated-law consumer against a continuous mode and source."""

import argparse
import math
from pathlib import Path

from check_foundations import run, table


def check(binary, root):
    for name, n, time, rate in (
        ("forced", 16, 0.75, 0.1),
        ("homogeneous", 16, 0.75, 0.0),
        ("negative-source", 9, -0.125, -0.3),
        ("initial-time", 8, 0.0, 0.2),
    ):
        folder = root / name
        run(binary, folder, parameters={"n": n, "time": time, "source_rate": rate})
        rows = table(folder / "results/rates.csv")
        seen = set()
        for row in rows:
            index = tuple(int(row[key]) for key in ("i", "j", "k"))
            assert all(0 <= q < n for q in index) and index not in seen
            seen.add(index)
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            assert all(
                abs(q - i * 2 * math.pi / n) < 2e-14 for q, i in zip((x, y, z), index)
            )
            mode = math.cos(x + 2 * y + 3 * z)
            source = rate * time * (x + y + z)
            expected = (
                -math.sin(time) * mode + source,
                (-math.cos(time) - 0.6 * math.sin(time)) * mode + source,
            )
            for key, value in zip(("du", "dv"), expected):
                assert abs(float(row[key]) - value) < 3e-13, (name, row, value)
        assert len(seen) == n**3
    for name, parameters in (
        ("bad-n", {"n": 1}),
        ("bad-source", {"source_rate": 11}),
        ("bad-time", {"time": 11}),
    ):
        run(binary, root / name, parameters=parameters, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(
        args.bin_dir.resolve() / "symbolic_coupled_rhs",
        args.output.resolve() / "symbolic",
    )
    print(
        "Generated-law consumer: complete fields, homogeneous rates and explicit source pass."
    )


if __name__ == "__main__":
    main()
