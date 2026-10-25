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


def check_nodal(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/potential.csv")
    assert len(data) == 2 * 141 * 113
    for row in data:
        if row["grid"] != "chebyshev":
            continue
        z = complex(float(row["real"]), float(row["imag"]))
        # U_15 from its three-term recurrence, not a product over the nodes.
        previous, current = 1, 2 * z
        for _ in range(2, 16):
            previous, current = current, 2 * z * current - previous
        exact = abs(2**-15 * (z * z - 1) * current)
        assert abs(float(row["magnitude"]) - exact) < 2e-12 * max(1, exact)
    line = rows(directory / "results/nodal.csv")
    assert len(line) == 802
    assert all(abs(float(r["value"])) < 1e-14 for r in line if abs(float(r["x"])) == 1)


def check_derivative(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/derivatives.csv")
    assert len(data) == 32
    errors = []
    for n in (10, 20):
        error = 0
        for row in (r for r in data if int(r["n"]) == n):
            x = float(row["x"])
            exact = math.exp(x) * (math.sin(5 * x) + 5 * math.cos(5 * x))
            dense, fourier = float(row["matrix"]), float(row["fourier"])
            assert abs(float(row["exact"]) - exact) < 1e-13
            assert abs(dense - fourier) < 2e-11
            error = max(error, abs(dense - exact), abs(fourier - exact))
        errors.append(error)
    assert errors[1] < 1e-8 and errors[1] < errors[0] / 1000, errors


def sample(kind, x):
    if kind == 0:
        return abs(x) ** 3, 3 * x * abs(x)
    if kind == 1:
        if abs(x) < 1e-14:
            return 0, 0
        value = math.exp(-1 / x**2)
        return value, 2 * value / x**3
    if kind == 2:
        return 1 / (1 + x * x), -2 * x / (1 + x * x) ** 2
    return x**10, 10 * x**9


def check_regularity(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/regularity.csv")
    assert len(data) == 200
    for row in data:
        n, kind, error = int(row["n"]), int(row["profile"]), float(row["error"])
        if n <= 12:
            x = [math.cos(math.pi * j / n) for j in range(n + 1)]
            values = [sample(kind, p)[0] for p in x]
            expected = max(
                abs(lagrange(x, values, p, derivative=True) - sample(kind, p)[1])
                for p in x
            )
            assert abs(error - expected) < 2e-10
        if kind == 3 and n >= 10:
            assert error < 1e-9
        if kind == 2 and n == 50:
            assert error < 1e-10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument(
        "--case",
        choices=[
            "polynomial_interpolation",
            "polynomial_nodal_polynomials",
            "polynomial_differentiation",
            "polynomial_regularity",
        ],
    )
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
        if args.case in (None, "polynomial_nodal_polynomials"):
            root = base / "polynomial_nodal_polynomials"
            root.mkdir()
            check_nodal(bin_dir / "polynomial_nodal_polynomials", root / "valid")
            run(
                bin_dir / "polynomial_nodal_polynomials",
                root / "invalid",
                parameters={"n": 0, "sizes": [0]},
                success=False,
            )
        if args.case in (None, "polynomial_differentiation"):
            root = base / "polynomial_differentiation"
            root.mkdir()
            check_derivative(bin_dir / "polynomial_differentiation", root / "valid")
            run(
                bin_dir / "polynomial_differentiation",
                root / "invalid",
                parameters={"n": 0, "sizes": [0]},
                success=False,
            )
        if args.case in (None, "polynomial_regularity"):
            root = base / "polynomial_regularity"
            root.mkdir()
            check_regularity(bin_dir / "polynomial_regularity", root / "valid")
            run(
                bin_dir / "polynomial_regularity",
                root / "invalid",
                parameters={"n": 0, "sizes": [0]},
                success=False,
            )
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
