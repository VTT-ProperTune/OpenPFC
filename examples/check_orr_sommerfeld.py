#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the original Orr–Sommerfeld pencil using Lagrange products."""

import argparse
import math
from pathlib import Path
import tempfile

from check_clamped import operators
from check_spectra import factor, rows, run, substitute


def pencil(n, reynolds):
    x, d2, d4 = operators(n)
    m = n - 1
    b = [[d2[i][j] - (i == j) for j in range(m)] for i in range(m)]
    a = [
        [
            (d4[i][j] - 2 * d2[i][j] + (i == j)) / reynolds
            - 1j * (2 * (i == j) + (1 - x[i + 1] ** 2) * b[i][j])
            for j in range(m)
        ]
        for i in range(m)
    ]
    return x, a, b


def leading_near(a, b, shift=-0.26j):
    # Generalized inverse iteration, independent of the C++ QR eigenproblem.
    n = len(a)
    lu = factor([[a[i][j] - shift * b[i][j] for j in range(n)] for i in range(n)])
    v = [complex(math.cos(0.73 * j)) for j in range(n)]
    for _ in range(60):
        rhs = [sum(value * u for value, u in zip(row, v)) for row in b]
        v = substitute(lu, rhs)
        norm = math.sqrt(sum(abs(value) ** 2 for value in v))
        v = [value / norm for value in v]
    av = [sum(value * u for value, u in zip(row, v)) for row in a]
    bv = [sum(value * u for value, u in zip(row, v)) for row in b]
    return sum(u.conjugate() * w for u, w in zip(bv, av)) / sum(abs(u) ** 2 for u in bv)


def check(binary, directory, degrees, reynolds):
    run(binary, directory, parameters={"degrees": degrees, "reynolds": reynolds})
    spectrum = rows(directory / "results/spectrum.csv")
    vectors = rows(directory / "results/vectors.csv")
    growth = rows(directory / "results/growth.csv")
    assert len(spectrum) == sum(n - 1 for n in degrees)
    assert len(vectors) == sum((n - 1) * (n + 1) for n in degrees)
    assert [int(row["n"]) for row in growth] == degrees
    for n in degrees:
        x, a, b = pencil(n, reynolds)
        m = n - 1
        eigenpairs = [row for row in spectrum if int(row["n"]) == n]
        modes = [row for row in vectors if int(row["n"]) == n]
        eigenvalues = [
            complex(float(row["real"]), float(row["imag"])) for row in eigenpairs
        ]
        assert len(eigenpairs) == m
        assert [value.real for value in eigenvalues] == sorted(
            value.real for value in eigenvalues
        )
        norm_a = math.sqrt(sum(abs(value) ** 2 for row in a for value in row))
        norm_b = math.sqrt(sum(value * value for row in b for value in row))
        for index, row in enumerate(eigenpairs):
            samples = [sample for sample in modes if int(sample["mode"]) == index + 1]
            assert len(samples) == n + 1
            for j, sample in enumerate(samples):
                assert int(sample["j"]) == j and abs(float(sample["y"]) - x[j]) < 1e-14
            values = [
                complex(float(sample["real"]), float(sample["imag"]))
                for sample in samples
            ]
            assert values[0] == values[-1] == 0
            v = values[1:-1]
            lam = eigenvalues[index]
            assert math.isfinite(abs(lam)) and all(math.isfinite(abs(u)) for u in v)
            assert abs(max(abs(u) for u in v) - 1) < 1e-12
            residual = [
                sum((a[i][j] - lam * b[i][j]) * v[j] for j in range(m))
                for i in range(m)
            ]
            scaled = math.sqrt(
                sum(abs(u) ** 2 for u in residual) / sum(abs(u) ** 2 for u in v)
            ) / (norm_a + abs(lam) * norm_b)
            assert scaled < 2e-9, (n, index, scaled)
            assert float(row["pencil_residual"]) < 1e-9
            assert float(row["transformed_residual"]) < 1e-9
        # The full trace checks multiplicity, rather than just the leading root.
        lu = factor(b)
        trace = sum(substitute(lu, [a[i][j] for i in range(m)])[j] for j in range(m))
        assert abs(sum(eigenvalues) - trace) < 1e-8 * max(1, abs(trace))
        expected = leading_near(a, b)
        assert abs(eigenvalues[-1] - expected) < 1e-8, (n, eigenvalues[-1], expected)
        summary = next(row for row in growth if int(row["n"]) == n)
        assert float(summary["reynolds"]) == reynolds
        assert (
            complex(float(summary["real"]), float(summary["imag"])) == eigenvalues[-1]
        )
        assert (
            (eigenvalues[-1].real < 0)
            if reynolds == 5772
            else (eigenvalues[-1].real > 0)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.bin_dir.resolve() / "orr_sommerfeld"
    with tempfile.TemporaryDirectory(prefix="orr-", dir=args.output) as name:
        root = Path(name).resolve()
        check(binary, root / "book", [40, 60, 80, 100], 5772)
        check(binary, root / "unstable", [80], 10000)
        for parameters in (
            {"degrees": []},
            {"degrees": [13]},
            {"degrees": [102]},
            {"reynolds": 0},
            {"reynolds": 1000001},
        ):
            run(binary, root / "bad", parameters=parameters, success=False)
    print(
        "Orr–Sommerfeld: all four spectra, generalized eigenvectors, trace and growth signs passed."
    )


if __name__ == "__main__":
    main()
