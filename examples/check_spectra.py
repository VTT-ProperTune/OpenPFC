#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent eigenproblem, tensor-boundary, contour and multigrid checks."""
from __future__ import annotations

import argparse
import cmath
import csv
import math
from pathlib import Path
import tempfile

from check_foundations import run


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def factor(matrix):
    a = [list(row) for row in matrix]
    pivots = []
    for k in range(len(a)):
        p = max(range(k, len(a)), key=lambda i: abs(a[i][k]))
        assert abs(a[p][k]) > 1e-18
        pivots.append(p)
        a[k], a[p] = a[p], a[k]
        for i in range(k + 1, len(a)):
            a[i][k] /= a[k][k]
            for j in range(k + 1, len(a)):
                a[i][j] -= a[i][k] * a[k][j]
    return a, pivots


def substitute(lu, rhs):
    a, pivots = lu
    x = list(rhs)
    for k, p in enumerate(pivots):
        x[k], x[p] = x[p], x[k]
    for i in range(len(x)):
        x[i] -= sum(a[i][j] * x[j] for j in range(i))
    for i in reversed(range(len(x))):
        x[i] = (x[i] - sum(a[i][j] * x[j] for j in range(i + 1, len(x)))) / a[i][i]
    return x


def lowest(matrix, count=1, shift=0):
    """Inverse iteration and orthogonal deflation, not Jacobi or shifted QR."""
    n = len(matrix)
    a = [
        [v + (shift if i == j else 0) for j, v in enumerate(row)]
        for i, row in enumerate(matrix)
    ]
    lu = factor(a)
    vectors, values = [], []
    for mode in range(count):
        v = [
            math.sin(0.73 * (j + 1) * (mode + 1)) + 0.1 * math.cos(0.31 * j)
            for j in range(n)
        ]
        for _ in range(180):
            v = substitute(lu, v)
            for _ in range(2):
                for previous in vectors:
                    dot = sum(x * y for x, y in zip(v, previous))
                    v = [x - dot * y for x, y in zip(v, previous)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / norm for x in v]
        values.append(
            sum(v[i] * sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n))
        )
        vectors.append(v)
    return values


def lagrange_second(n):
    """Differentiate Lagrange products directly; do not square a D matrix."""
    x = [math.cos(math.pi * j / n) for j in range(n + 1)]
    a = [[0.0] * (n + 1) for _ in x]
    denominator = [
        math.prod(p - other for k, other in enumerate(x) if k != j)
        for j, p in enumerate(x)
    ]
    for i, point in enumerate(x):
        for j in range(n + 1):
            if i == j:
                inverses = [1 / (point - other) for k, other in enumerate(x) if k != i]
                a[i][j] = sum(inverses) ** 2 - sum(v * v for v in inverses)
            else:
                others = [other for k, other in enumerate(x) if k not in (i, j)]
                a[i][j] = (
                    2
                    * math.prod(point - other for other in others)
                    * sum(1 / (point - other) for other in others)
                    / denominator[j]
                )
    return x, a


def check_harmonic(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/energies.csv")
    assert len(data) == 24
    fine = [float(r["eigenvalue"]) for r in data if int(r["n"]) == 36]
    assert max(abs(value - (2 * j + 1)) for j, value in enumerate(fine)) < 1e-9
    n = 36
    k = [j if j <= n // 2 else j - n for j in range(n)]
    # Direct DFT sum for -d_xx, independent of FFTW.
    a = [
        [
            sum(
                (math.pi * mode / 8) ** 2 * cmath.exp(2j * math.pi * mode * (i - j) / n)
                for mode in k
            ).real
            / n
            for j in range(n)
        ]
        for i in range(n)
    ]
    for i in range(n):
        a[i][i] += (-8 + 16 * (i + 1) / n) ** 2
    expected = lowest(a, 4)
    assert (
        max(abs(value - reference) for value, reference in zip(fine, expected)) < 2e-10
    )
    run(
        binary,
        directory.parent / "bad-harmonic",
        parameters={"sizes": [7]},
        success=False,
    )


def check_mathieu(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/mathieu.csv")
    assert len(data) == 76 * 11
    free = [0, 1, 1, 4, 4, 9, 9, 16, 16, 25, 25]
    for step in range(76):
        subset = [r for r in data if abs(float(r["q"]) - 0.2 * step) < 1e-12]
        values = [float(r["eigenvalue"]) for r in subset]
        assert len(values) == 11 and values == sorted(values)
        assert all(math.isfinite(v) for v in values)
        assert all(abs(v - base) <= 0.4 * step + 1e-10 for v, base in zip(values, free))
        if step == 0:
            assert max(abs(v - base) for v, base in zip(values, free)) < 1e-10
    # Fourier-mode potential couples k to k+-2; solve each parity sector.
    ground = []
    for parity in (0, 1):
        indices = list(range(parity, 42, 2))
        m = len(indices)
        a = [[0.0] * m for _ in indices]
        for i, index in enumerate(indices):
            k = index if index <= 21 else index - 42
            a[i][i] = k * k
            a[i][(i - 1) % m] += 15
            a[i][(i + 1) % m] += 15
        ground.extend(lowest(a, shift=31))
    native = next(
        float(r["eigenvalue"])
        for r in data
        if float(r["q"]) == 15 and int(r["mode"]) == 1
    )
    assert abs(native - min(ground)) < 1e-10
    run(binary, directory.parent / "bad-mathieu", parameters={"n": 11}, success=False)


def check_eigenmodes(binary, root):
    maxima = []
    for n, modes in ((36, [5, 10, 15, 20, 25, 30]), (60, [15, 59])):
        directory = root / f"eigen-{n}"
        run(binary, directory, parameters={"n": n, "modes": modes})
        data = rows(directory / "results/spectrum.csv")
        vectors = rows(directory / "results/vectors.csv")
        assert len(data) == n - 1 and len(vectors) == (n - 1) * (n + 1)
        x, a = lagrange_second(n)
        norm_a = math.sqrt(sum(a[i][j] ** 2 for i in range(1, n) for j in range(1, n)))
        for row in data:
            mode = int(row["mode"])
            eigenvalue = complex(float(row["eigenvalue"]), float(row["imag"]))
            v = [
                complex(float(r["real"]), float(r["imag"]))
                for r in vectors
                if int(r["mode"]) == mode
            ]
            assert v[0] == v[-1] == 0 and all(math.isfinite(abs(value)) for value in v)
            residual = [
                -sum(a[i][j] * v[j] for j in range(1, n)) - eigenvalue * v[i]
                for i in range(1, n)
            ]
            relative = math.sqrt(
                sum(abs(r) ** 2 for r in residual) / sum(abs(z) ** 2 for z in v)
            ) / (norm_a + abs(eigenvalue))
            assert relative < 1e-10, (n, mode, relative)
            if mode <= 5:
                assert abs(eigenvalue - (mode * math.pi / 2) ** 2) < 1e-7
        trace = -sum(a[i][i] for i in range(1, n))
        assert abs(sum(float(r["eigenvalue"]) for r in data) - trace) < 2e-10 * abs(
            trace
        )
        maxima.append(float(data[-1]["eigenvalue"]))
    assert 0.9 < maxima[1] / maxima[0] / (60 / 36) ** 4 < 1.1
    run(binary, root / "bad-eigen", parameters={"n": 8, "modes": [9]}, success=False)


def check_airy(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/eigenvalues.csv")
    fields = rows(directory / "results/airy.csv")
    assert len(data) == 4
    for row in data:
        n = int(row["n"])
        lam = float(row["eigenvalue"])
        x, a = lagrange_second(n)
        u = [float(r["u"]) for r in fields if int(r["n"]) == n and r["kind"] == "node"]
        assert len(u) == n + 1 and u[0] == u[-1] == 0
        assert abs(u[n // 2] - 0.35502805388781723926) < 1e-13
        residual = [
            sum(a[i][j] * u[j] for j in range(n + 1)) - lam * x[i] * u[i]
            for i in range(1, n)
        ]
        scale = max(abs(a[i][j]) for i in range(1, n) for j in range(n + 1)) * max(
            abs(v) for v in u
        )
        assert max(abs(r) for r in residual) < 1e-9 * scale
    # Independent Airy boundary determinant, verified by the reference job.
    assert abs(float(data[-1]["eigenvalue"]) - 501.3483797111) < 1e-7
    assert abs(float(data[0]["eigenvalue"]) - 501.3483797111) > 1
    run(
        binary, directory.parent / "bad-airy", parameters={"sizes": [13]}, success=False
    )


def check_membrane(binary, directory):
    run(binary, directory)
    data = rows(directory / "results/spectrum.csv")
    fields = rows(directory / "results/modes.csv")
    assert len(data) == 4 and len(fields) == 4 * 17**2
    x, a = lagrange_second(16)
    for row in data:
        mode = int(row["mode"])
        lam = float(row["eigenvalue"])
        u = [[0.0] * 17 for _ in x]
        for r in fields:
            if int(r["mode"]) == mode:
                u[int(r["j"])][int(r["i"])] = float(r["u"])
        assert all(u[0][i] == u[-1][i] == u[i][0] == u[i][-1] == 0 for i in range(17))
        error = 0
        for j in range(1, 16):
            for i in range(1, 16):
                value = -sum(a[i][k] * u[j][k] + a[j][k] * u[k][i] for k in range(17))
                value += (math.exp(20 * (x[j] - x[i] - 1)) - lam) * u[j][i]
                error = max(error, abs(value))
        assert error < 1e-7, (mode, error)
    assert float(data[0]["eigenvalue"]) > math.pi**2 / 2
    run(binary, directory.parent / "bad-membrane", parameters={"n": 21}, success=False)


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
            "harmonic_oscillator",
            "mathieu_spectrum",
            "chebyshev_eigenmodes",
            "airy_eigenproblem",
            "membrane_eigenmodes",
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
        if args.case in (None, "harmonic_oscillator"):
            root = base / "harmonic_oscillator"
            root.mkdir()
            check_harmonic(bin_dir / "harmonic_oscillator", root)
        if args.case in (None, "mathieu_spectrum"):
            root = base / "mathieu_spectrum"
            root.mkdir()
            check_mathieu(bin_dir / "mathieu_spectrum", root)
        if args.case in (None, "chebyshev_eigenmodes"):
            root = base / "chebyshev_eigenmodes"
            root.mkdir()
            check_eigenmodes(bin_dir / "chebyshev_eigenmodes", root)
        if args.case in (None, "airy_eigenproblem"):
            root = base / "airy_eigenproblem"
            root.mkdir()
            check_airy(bin_dir / "airy_eigenproblem", root)
        if args.case in (None, "membrane_eigenmodes"):
            root = base / "membrane_eigenmodes"
            root.mkdir()
            check_membrane(bin_dir / "membrane_eigenmodes", root)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
