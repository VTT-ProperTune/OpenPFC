#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check complete native coupled rates against analytic and moment-based oracles."""

import argparse
import cmath
from fractions import Fraction
import math
from pathlib import Path

from check_foundations import run, table


def moment_weights(derivative):
    """Derive seven-point weights by exact polynomial moments, not library code."""
    offsets = list(range(-3, 4))
    system = [
        [Fraction(j**power) for j in offsets]
        + [Fraction(math.factorial(power) if power == derivative else 0)]
        for power in range(7)
    ]
    for col in range(7):
        pivot = next(row for row in range(col, 7) if system[row][col])
        system[col], system[pivot] = system[pivot], system[col]
        scale = system[col][col]
        system[col] = [value / scale for value in system[col]]
        for row in range(7):
            if row != col:
                scale = system[row][col]
                system[row] = [a - scale * b for a, b in zip(system[row], system[col])]
    return [float(row[-1]) for row in system]


def check(binary, root, spectral):
    weights = {order: moment_weights(order) for order in (1, 2)}
    errors = []
    cases = [(f"fd-{n}", n, 0.75, "fd") for n in (12, 16, 24, 32, 48)]
    cases += [("fd-odd-time", 25, 0.125, "fd")]
    if spectral:
        cases += [
            ("spectral", 24, 0.75, "spectral"),
            ("spectral-odd-time", 25, 0.125, "spectral"),
        ]
    for name, n, time, method in cases:
        folder = root / name
        run(binary, folder, parameters={"n": n, "time": time, "method": method})
        h = 2 * math.pi / n

        def symbol(k, order):
            return (
                sum(
                    w * cmath.exp(1j * k * j * h)
                    for w, j in zip(weights[order], range(-3, 4))
                )
                / h**order
            )

        first = [symbol(k, 1).imag for k in (1, 2, 3)]
        second = sum(symbol(k, 2).real for k in (1, 2, 3))
        a, b = math.cos(time), -math.sin(time) + 0.6 * math.cos(time)
        if method == "fd":
            rate_u = b - 0.3 * first[0] * first[1] * a
            rate_v = (0.04 * second - 2 / 15 * first[1] * first[2]) * a - 0.2 * (
                -first[0] * first[2]
            ) * b
        else:
            rate_u, rate_v = -math.sin(time), -math.cos(time) - 0.6 * math.sin(time)
        rows = table(folder / "results/rates.csv")
        seen = set()
        maximum = 0
        for row in rows:
            index = tuple(int(row[key]) for key in ("i", "j", "k"))
            assert all(0 <= q < n for q in index) and index not in seen
            seen.add(index)
            x, y, z = (float(row[key]) for key in ("x", "y", "z"))
            assert all(abs(q - i * h) < 2e-14 for q, i in zip((x, y, z), index))
            mode = math.cos(x + 2 * y + 3 * z)
            expected = {
                "u": a * mode,
                "v": b * mode,
                "du": rate_u * mode,
                "dv": rate_v * mode,
                "exact_du": -math.sin(time) * mode,
                "exact_dv": (-math.cos(time) - 0.6 * math.sin(time)) * mode,
            }
            for key, value in expected.items():
                assert abs(float(row[key]) - value) < 2e-11, (name, key, row, value)
            maximum = max(
                maximum,
                abs(float(row["du"]) - expected["exact_du"]),
                abs(float(row["dv"]) - expected["exact_dv"]),
            )
        assert len(seen) == n**3
        if method == "spectral":
            assert maximum < 1e-10, maximum
        elif time == 0.75:
            errors.append((n, maximum))
    ratio = errors[2][1] / errors[4][1]
    assert 45 < ratio < 75, (errors, ratio)
    print("FD24/FD48 continuum RHS error ratio:", ratio)
    for name, parameters in (
        ("bad-n", {"n": 6}),
        ("bad-method", {"method": "missing"}),
        ("bad-time", {"time": 11}),
    ):
        run(binary, root / name, parameters=parameters, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spectral", action="store_true")
    args = parser.parse_args()
    check(
        args.bin_dir.resolve() / "coupled_fields_example",
        args.output.resolve() / "coupled",
        args.spectral,
    )
    print("Coupled fields, mixed derivatives, tuple outputs and refinement pass.")


if __name__ == "__main__":
    main()
