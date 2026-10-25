#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify the polynomial experiments with analytic and independent basis formulas."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import tempfile

from check_foundations import run


def rows(path):
    with path.open() as file:
        data = list(csv.DictReader(file))
    for row in data:
        for key, value in row.items():
            if key not in ("grid", "kind", "boundary"):
                assert math.isfinite(float(value)), (path, key, value)
    return data


def lagrange(nodes, values, x, derivative=False):
    """Direct Lagrange basis, independent of barycentric weights and D entries."""
    total = 0
    for j, (node, value) in enumerate(zip(nodes, values)):
        others = [k for k in range(len(nodes)) if k != j]
        denominator = math.prod(node - nodes[k] for k in others)
        if derivative:
            basis = sum(
                math.prod(x - nodes[k] for k in others if k != m) for m in others
            )
        else:
            basis = math.prod(x - nodes[k] for k in others)
        total += value * basis / denominator
    return total


def check_interpolation(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/interpolation.csv")
    assert len(data) == 2 * (17 + 405)
    errors = []
    for grid in ("equispaced", "chebyshev"):
        samples = [r for r in data if r["grid"] == grid and r["kind"] == "node"]
        x = [float(r["x"]) for r in samples]
        values = [1 / (1 + 16 * p * p) for p in x]
        error = 0
        for row in (r for r in data if r["grid"] == grid and r["kind"] == "curve"):
            point, value = float(row["x"]), float(row["value"])
            assert abs(value - lagrange(x, values, point)) < 2e-9
            if abs(point) <= 1:
                error = max(error, abs(value - 1 / (1 + 16 * point**2)))
        errors.append(error)
    assert errors[0] > 1 and errors[1] < 0.03, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--case", choices=["polynomial_interpolation"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bin_dir = args.bin_dir.resolve()
    ranks = args.ranks
    launch = [args.launcher, args.rank_flag, str(ranks)] if args.launcher else []
    with tempfile.TemporaryDirectory(
        prefix="example-check-", dir=args.output
    ) as temporary:
        base = Path(temporary).resolve()
        if args.case in (None, "polynomial_interpolation"):
            root = base / "polynomial_interpolation"
            root.mkdir()
            check_interpolation(bin_dir / "polynomial_interpolation", root / "valid")
            run(
                bin_dir / "polynomial_interpolation",
                root / "invalid",
                parameters={"n": 0, "sizes": [0]},
                success=False,
            )
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
