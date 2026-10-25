#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check singular values by shifted solves, without forming normal equations."""

import argparse
import math
from pathlib import Path
import tempfile

from check_spectra import factor, lagrange_second, rows, run, substitute


def oscillator(n, length):
    x, d2 = lagrange_second(n)
    return [
        [
            -d2[i][j] / length**2 + ((1 + 3j) * (length * x[i]) ** 2 if i == j else 0)
            for j in range(1, n)
        ]
        for i in range(1, n)
    ]


def inverse_singular(b):
    n = len(b)
    lu = factor(b)
    adjoint = factor([[b[j][i].conjugate() for j in range(n)] for i in range(n)])
    v = [complex(math.sin(0.37 * (j + 1)), math.cos(0.61 * j)) for j in range(n)]
    previous = None
    stable = 0
    for _ in range(2000):
        v = substitute(lu, substitute(adjoint, v))
        norm = math.sqrt(sum(abs(value) ** 2 for value in v))
        v = [value / norm for value in v]
        image = [sum(value * u for value, u in zip(row, v)) for row in b]
        sigma = math.sqrt(sum(abs(value) ** 2 for value in image))
        change = abs(sigma - previous) if previous is not None else math.inf
        if change < 1e-13 * max(1, sigma):
            stable += 1
        else:
            stable = 0
        if stable == 6:
            return sigma
        previous = sigma
    raise AssertionError(
        ("Independent singular inverse iteration did not converge", sigma, change)
    )


def smallest_by_solves(b):
    # The even potential commutes with reflection. Opposite-parity singular
    # values can be nearly equal, making scalar inverse iteration very slow.
    n = len(b)
    reflection_error = 0.5 * math.sqrt(
        sum(
            abs(b[i][j] - b[n - 1 - i][n - 1 - j]) ** 2
            for i in range(n)
            for j in range(n)
        )
    )
    assert reflection_error < 5e-9, reflection_error
    minima = []
    for sign in (1, -1):
        basis = [
            ((i, 1 / math.sqrt(2)), (n - 1 - i, sign / math.sqrt(2)))
            for i in range(n // 2)
        ]
        if n % 2 and sign == 1:
            basis.append(((n // 2, 1.0),))
        block = [
            [
                sum(ci * cj * b[i][j] for i, ci in row for j, cj in column)
                for column in basis
            ]
            for row in basis
        ]
        minima.append(inverse_singular(block))
    return min(minima)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.bin_dir.resolve() / "davies_pseudospectra"
    with tempfile.TemporaryDirectory(prefix="pseudospectra-", dir=args.output) as name:
        root = Path(name).resolve()
        for n in (12, 70):
            directory = root / str(n)
            run(binary, directory, parameters={"n": n, "length": 6, "step": 10})
            data = rows(directory / "results/pseudospectrum.csv")
            spectrum = rows(directory / "results/spectrum.csv")
            assert len(data) == 30 and len(spectrum) == n - 1
            assert {(float(r["real"]), float(r["imag"])) for r in data} == {
                (10 * i, 10 * j) for j in range(5) for i in range(6)
            }
            a = oscillator(n, 6)
            eigenvalues = [
                complex(float(row["real"]), float(row["imag"])) for row in spectrum
            ]
            assert all(math.isfinite(abs(value)) for value in eigenvalues)
            assert all(float(row["residual"]) < 1e-9 for row in spectrum)
            assert abs(
                sum(eigenvalues) - sum(a[i][i] for i in range(n - 1))
            ) < 1e-9 * abs(sum(eigenvalues))
            for row in data:
                z = complex(float(row["real"]), float(row["imag"]))
                shifted = [
                    [(z if i == j else 0) - value for j, value in enumerate(line)]
                    for i, line in enumerate(a)
                ]
                try:
                    expected = smallest_by_solves(shifted)
                except AssertionError as error:
                    raise AssertionError((n, z)) from error
                actual = float(row["sigma_min"])
                assert math.isfinite(actual) and actual >= 0
                assert abs(actual - expected) < 2e-8 * max(1, expected), (
                    n,
                    z,
                    actual,
                    expected,
                )
                assert actual <= min(abs(z - value) for value in eigenvalues) + 1e-8
            if n == 70:
                branch = [(2 * j + 1) * (1 + 3j) ** 0.5 for j in range(4)]
                for expected in branch:
                    assert min(abs(value - expected) for value in eigenvalues) < 1e-7
        for parameters in ({"n": 5}, {"n": 81}, {"length": 0}, {"step": 0.1}):
            run(binary, root / "bad", parameters=parameters, success=False)
    print(
        "Pseudospectra: complete coarse grids, direct shifted-solve singular values and spectral checks passed."
    )


if __name__ == "__main__":
    main()
