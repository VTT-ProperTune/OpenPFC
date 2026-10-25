#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check plate modes using directly differentiated Lagrange products."""

import argparse
import math
from pathlib import Path
import tempfile

from check_spectra import lowest, rows, run


def operators(n):
    x = [math.cos(math.pi * j / n) for j in range(n + 1)]
    second, fourth = [], []
    for point in x[1:-1]:
        row2, row4 = [], []
        for j in range(1, n):
            coefficients = [1.0, 0.0, 0.0, 0.0, 0.0]
            denominator = 1.0
            for k, other in enumerate(x):
                if k == j:
                    continue
                denominator *= x[j] - other
                for order in range(4, 0, -1):
                    coefficients[order] = (point - other) * coefficients[
                        order
                    ] + coefficients[order - 1]
                coefficients[0] *= point - other
            row2.append(2 * coefficients[2] / denominator)
            # Differentiate (1-x^2)*ell_j(x)/(1-x_j^2) directly.
            row4.append(
                24
                * (
                    (1 - point * point) * coefficients[4]
                    - 2 * point * coefficients[3]
                    - coefficients[2]
                )
                / denominator
                / (1 - x[j] ** 2)
            )
        second.append(row2)
        fourth.append(row4)
    return x, second, fourth


def matrix(second, fourth):
    m = len(second)
    return [
        [
            (fourth[i][a] if j == b else 0)
            + (fourth[j][b] if i == a else 0)
            + 2 * second[i][a] * second[j][b]
            for b in range(m)
            for a in range(m)
        ]
        for j in range(m)
        for i in range(m)
    ]


def reconstruction(x, point):
    values, slopes = [], []
    for j in range(1, len(x) - 1):
        factors = [point - other for k, other in enumerate(x) if k != j]
        denominator = math.prod(x[j] - other for k, other in enumerate(x) if k != j)
        value = math.prod(factors) / denominator
        derivative = (
            sum(math.prod(factors[:k] + factors[k + 1 :]) for k in range(len(factors)))
            / denominator
        )
        values.append((1 - point**2) * value / (1 - x[j] ** 2))
        slopes.append(
            (-2 * point * value + (1 - point**2) * derivative) / (1 - x[j] ** 2)
        )
    return values, slopes


def check(binary, root, n):
    directory = root / f"plate-{n}"
    run(binary, directory, parameters={"n": n, "modes": 25})
    spectrum = rows(directory / "results/spectrum.csv")
    shapes = rows(directory / "results/shapes.csv")
    x, second, fourth = operators(n)
    m = n - 1
    a = matrix(second, fourth)
    eigenvalues = [float(row["eigenvalue"]) for row in spectrum]
    assert len(spectrum) == 25 and eigenvalues == sorted(eigenvalues)
    fundamental = lowest(a)[0]
    assert abs(eigenvalues[0] - fundamental) < 2e-8 * fundamental
    if n == 6:  # All 25 eigenvalues: multiplicity matters to the trace.
        assert abs(sum(eigenvalues) - sum(a[i][i] for i in range(m * m))) < 1e-8
    independent = []
    for index, row in enumerate(spectrum):
        mode = index + 1
        samples = [r for r in shapes if int(r["mode"]) == mode]
        nodes = [r for r in samples if r["kind"] == "node"]
        assert len(nodes) == (n + 1) ** 2
        assert len(samples) == (n + 1) ** 2 + 65**2
        vector = [
            float(r["u"]) for r in nodes if 0 < int(r["i"]) < n and 0 < int(r["j"]) < n
        ]
        image = [sum(value * u for value, u in zip(line, vector)) for line in a]
        residual = math.sqrt(
            sum(
                (value - eigenvalues[index] * u) ** 2 for value, u in zip(image, vector)
            )
            / sum(value * value for value in image)
        )
        assert residual < 2e-8, (n, mode, residual)
        assert float(row["residual"]) < 1e-8
        assert (
            abs(
                float(row["frequency_ratio"])
                - math.sqrt(eigenvalues[index] / eigenvalues[0])
            )
            < 1e-13
        )
        indexed = {(int(r["i"]), int(r["j"])): float(r["u"]) for r in nodes}
        for sample in nodes:
            assert abs(float(sample["x"]) - x[int(sample["i"])]) < 1e-14
            assert abs(float(sample["y"]) - x[int(sample["j"])]) < 1e-14
        for j in range(n + 1):
            for i in range(n + 1):
                u = indexed[i, j]
                assert abs(indexed[n - i, j] - int(row["parity_x"]) * u) < 1e-10
                assert abs(indexed[i, n - j] - int(row["parity_y"]) * u) < 1e-10
                if int(row["exchange"]):
                    assert abs(indexed[j, i] - int(row["exchange"]) * u) < 1e-10
        for sample in samples:
            px, py = float(sample["x"]), float(sample["y"])
            assert all(
                math.isfinite(float(sample[key])) for key in ("u", "du_dx", "du_dy")
            )
            if abs(px) == 1 or abs(py) == 1:
                assert float(sample["u"]) == 0
            if abs(px) == 1:
                assert float(sample["du_dx"]) == 0
            if abs(py) == 1:
                assert float(sample["du_dy"]) == 0
            if (
                sample["kind"] == "curve"
                and int(sample["i"]) in (13, 32, 49)
                and int(sample["j"]) in (17, 37)
            ):
                vx, dx = reconstruction(x, px)
                vy, dy = reconstruction(x, py)
                for key, bx, by in (
                    ("u", vx, vy),
                    ("du_dx", dx, vy),
                    ("du_dy", vx, dy),
                ):
                    expected = sum(
                        bx[i] * by[j] * indexed[i + 1, j + 1]
                        for j in range(m)
                        for i in range(m)
                    )
                    assert abs(float(sample[key]) - expected) < 1e-9, (n, mode, key)
        # An eigenvalue list alone could hide duplicated vectors at a double root.
        norm = math.sqrt(sum(u * u for u in vector))
        vector = [u / norm for u in vector]
        for _ in range(2):
            for previous in independent:
                dot = sum(u * v for u, v in zip(vector, previous))
                vector = [u - dot * v for u, v in zip(vector, previous)]
        remainder = math.sqrt(sum(u * u for u in vector))
        assert remainder > 0.1, (n, mode, remainder)
        independent.append([u / remainder for u in vector])
    # The first cross-parity pair is a double eigenvalue with two shapes.
    assert abs(eigenvalues[1] - eigenvalues[2]) < 1e-10
    return eigenvalues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.bin_dir.resolve() / "clamped_plate"
    with tempfile.TemporaryDirectory(prefix="clamped-", dir=args.output) as temporary:
        root = Path(temporary).resolve()
        check(binary, root, 6)
        check(binary, root, 17)
        for parameters in ({"n": 5}, {"n": 21}, {"n": 8, "modes": 26}):
            run(binary, root / "bad-plate", parameters=parameters, success=False)
    print(
        "Clamped plate: complete fields, independent residuals, edges and multiplicities passed."
    )


if __name__ == "__main__":
    main()
