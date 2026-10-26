#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check actual generic stencil responses using affine and polynomial moments."""

import argparse
from pathlib import Path

from check_foundations import run, table


def check(binary, root):
    errors = []
    for name, profile, n, h in (
        ("affine", "affine", 8, 0.3),
        ("edge", "edge", 17, 0.25),
        ("polynomial", "polynomial", 9, 0.25),
        ("polynomial-fine", "polynomial", 17, 0.125),
    ):
        folder = root / name
        run(binary, folder, parameters={"n": n, "spacing": h, "profile": profile})
        data = table(folder / "results/responses.csv")
        seen = set()
        error = 0
        for row in data:
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            index = tuple(round(q / h + (n - 1) / 2) for q in (x, y, z))
            assert all(1 <= i < n - 1 for i in index) and index not in seen
            seen.add(index)
            assert all(
                abs(q - h * (i - (n - 1) / 2)) < 2e-14 for q, i in zip((x, y, z), index)
            )
            if profile == "affine":
                value, central, smoothed = x + 2 * y - 3 * z, 1, 1
            elif profile == "polynomial":
                value = x**3 + x * y**2 + 0.5 * x * z**2
                true_derivative = 3 * x**2 + y**2 + 0.5 * z**2
                central = true_derivative + h**2
                smoothed = true_derivative + 1.75 * h**2
                error = max(error, abs(float(row["separable"]) - true_derivative))
            else:
                amplitude = 1 + 0.2 * y**2 + 0.1 * z**2
                jump = (int(x + h >= 0) - int(x - h >= 0)) / (2 * h)
                value = amplitude if x >= 0 else 0
                central, smoothed = jump * amplitude, jump * (amplitude + 0.15 * h**2)
            assert abs(float(row["u"]) - value) < 2e-13
            assert abs(float(row["central"]) - central) < 2e-12, (name, row)
            assert abs(float(row["separable"]) - smoothed) < 2e-12, (name, row)
            assert abs(float(row["dense"]) - smoothed) < 2e-12, (name, row)
        assert len(seen) == (n - 2) ** 3
        if profile == "polynomial":
            errors.append(error)
    assert abs(errors[0] / errors[1] - 4) < 1e-10, errors
    for name, parameters in (
        ("bad-size", {"n": 2}),
        ("bad-spacing", {"spacing": 0}),
        ("bad-profile", {"profile": "missing"}),
    ):
        run(binary, root / name, parameters=parameters, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(
        args.bin_dir.resolve() / "18_sobel_edge_detection",
        args.output.resolve() / "stencils",
    )
    print("Stencil normalization, polynomial moments and complete interiors pass.")


if __name__ == "__main__":
    main()
