#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check polar operators with Lagrange products, direct Fourier sums and Bessel series."""

import argparse
import cmath
import math
from pathlib import Path
import tempfile

from check_spectra import rows, run


def derivatives(n):
    r = [math.cos(math.pi * j / n) for j in range(n + 1)]
    first, second = [], []
    for point in r:
        d, dd = [], []
        for j in range(n + 1):
            c = [1.0, 0.0, 0.0]
            denominator = 1.0
            for k, other in enumerate(r):
                if k == j:
                    continue
                denominator *= r[j] - other
                c = [
                    (point - other) * c[0],
                    (point - other) * c[1] + c[0],
                    (point - other) * c[2] + c[1],
                ]
            d.append(c[1] / denominator)
            dd.append(2 * c[2] / denominator)
        first.append(d)
        second.append(dd)
    return r, first, second


def radial_block(n, k):
    r, d, dd = derivatives(n)
    p = (n - 1) // 2
    return [
        [
            dd[i][j]
            + d[i][j] / r[i]
            + (-1) ** k * (dd[i][n - j] + d[i][n - j] / r[i])
            - (k * k / r[i] ** 2 if i == j else 0)
            for j in range(1, p + 1)
        ]
        for i in range(1, p + 1)
    ]


def polar_action(n, m, field):
    r, d, dd = derivatives(n)
    p = (n - 1) // 2
    angular = [
        [
            sum(
                -k * k * cmath.exp(2j * math.pi * k * (t - s) / m)
                for k in range(-m // 2, m // 2)
            ).real
            / m
            for s in range(m)
        ]
        for t in range(m)
    ]
    out = []
    for i in range(1, p + 1):
        for t in range(m):
            value = (
                sum(angular[t][s] * field[s + m * (i - 1)] for s in range(m))
                / r[i] ** 2
            )
            value += sum(
                (dd[i][j] + d[i][j] / r[i]) * field[t + m * (j - 1)]
                + (dd[i][n - j] + d[i][n - j] / r[i])
                * field[(t + m // 2) % m + m * (j - 1)]
                for j in range(1, p + 1)
            )
            out.append(value)
    return out


def bessel(k, x):
    term = (x / 2) ** k / math.factorial(k)
    value = term
    for j in range(1, 100):
        term *= -x * x / (4 * j * (j + k))
        value += term
        if abs(term) < 1e-19:
            break
    return value


def bessel_root(k, index):
    low = 0.01
    found = 0
    while low < 20:
        high = low + 0.1
        if bessel(k, low) * bessel(k, high) < 0:
            found += 1
            if found == index:
                for _ in range(50):
                    middle = (low + high) / 2
                    if bessel(k, low) * bessel(k, middle) <= 0:
                        high = middle
                    else:
                        low = middle
                return (low + high) / 2
        low = high
    raise AssertionError("Bessel root was not bracketed")


def check_modes(binary, root):
    n, m = 25, 20
    run(binary, root, parameters={"n": n, "m": m})
    spectrum = rows(root / "results/spectrum.csv")
    radial = rows(root / "results/radial.csv")
    fields = rows(root / "results/fields.csv")
    selected = rows(root / "results/selected.csv")
    p = (n - 1) // 2
    assert len(spectrum) == p * m and len(radial) == p * p * m
    assert [int(row["mode"]) for row in selected] == [1, 3, 6, 10]
    for k in range(m // 2 + 1):
        block = radial_block(n, k)
        samples = [row for row in spectrum if int(row["k"]) == k]
        assert len(samples) == p * (1 if k in (0, m // 2) else 2)
        for row in samples:
            values = [
                complex(float(q["real"]), float(q["imag"]))
                for q in radial
                if q["mode"] == row["mode"]
            ]
            lam = complex(float(row["real"]), float(row["imag"]))
            assert len(values) == p and math.isfinite(abs(lam))
            error = [
                -sum(a * u for a, u in zip(line, values)) - lam * v
                for line, v in zip(block, values)
            ]
            norm = math.sqrt(
                sum(abs(e) ** 2 for e in error) / sum(abs(v) ** 2 for v in values)
            )
            assert norm < 1e-7 * max(1, abs(lam)), (k, lam, norm)
            assert float(row["residual"]) < 1e-9
    fundamental = bessel_root(0, 1)
    for row in selected:
        index = int(row["mode"])
        k = int(row["k"])
        order = int(row["radial_mode"])
        alpha = bessel_root(k, order)
        assert abs(float(row["eigenvalue"]) - alpha * alpha) < 2e-7, (index, alpha, row)
        assert abs(float(row["frequency_ratio"]) - alpha / fundamental) < 1e-8
        samples = [q for q in fields if int(q["mode"]) == index]
        nodes = [q for q in samples if q["kind"] == "node"]
        assert len(nodes) == (p + 1) * m and len(samples) == (p + 1) * m + 65 * 129
        angular = math.sin if row["branch"] == "sin" else math.cos
        expected = [
            bessel(k, alpha * float(q["r"])) * angular(k * float(q["theta"]))
            for q in nodes
        ]
        actual = [float(q["u"]) for q in nodes]
        scale = sum(a * b for a, b in zip(actual, expected)) / sum(
            v * v for v in expected
        )
        assert max(abs(a - scale * b) for a, b in zip(actual, expected)) < 1e-8
        interior = [float(q["u"]) for q in nodes if int(q["ir"]) > 0]
        image = polar_action(n, m, interior)
        assert (
            max(abs(v + float(row["eigenvalue"]) * u) for v, u in zip(image, interior))
            < 1e-7
        )
        for q in samples:
            r, theta = float(q["r"]), float(q["theta"])
            assert abs(float(q["x"]) - r * math.cos(theta)) < 1e-14
            assert abs(float(q["y"]) - r * math.sin(theta)) < 1e-14
            assert (
                abs(float(q["u"]) - scale * bessel(k, alpha * r) * angular(k * theta))
                < 2e-8
            )
            if r == 1:
                assert float(q["u"]) == 0
            if r == 0 and k > 0:
                assert float(q["u"]) == 0


def reference(r, theta, manufactured):
    x, y = r * math.cos(theta), r * math.sin(theta)
    if manufactured:
        return (1 - x * x - y * y) * (1 + x + 0.25 * (x * x - y * y))
    # Integrate each forcing Fourier term: r^q ->
    # (r^(q+2)-r^k)/((q+2)^2-k^2), with the boundary value subtracted.
    terms = [
        (0, 2, -3 / 8, math.cos),
        (1, 2, 0.5, math.cos),
        (2, 2, -1 / 8, math.cos),
        (6, 0, 0.5, math.sin),
        (5, 0, 0.25, math.sin),
        (7, 0, 0.25, math.sin),
    ]
    return sum(
        c * (r ** (q + 2) - r**k) / ((q + 2) ** 2 - k * k) * trig(k * theta)
        for k, q, c, trig in terms
    )


def check_poisson(binary, root, n, manufactured=False):
    m = 16 if manufactured else 40
    run(binary, root, parameters={"n": n, "m": m, "manufactured": manufactured})
    samples = rows(root / "results/solution.csv")
    diagnostics = rows(root / "results/diagnostics.csv")[0]
    nodes = [q for q in samples if q["kind"] == "node"]
    p = (n - 1) // 2
    assert len(nodes) == (p + 1) * m and len(samples) == (p + 1) * m + 65 * 129
    assert float(diagnostics["max_residual"]) < 1e-8
    interior = [float(q["u"]) for q in nodes if int(q["ir"]) > 0]
    image = polar_action(n, m, interior)
    forcing = []
    error = 0
    for q in samples:
        r, theta = float(q["r"]), float(q["theta"])
        x = float(q["x"])
        y = float(q["y"])
        assert (
            abs(x - r * math.cos(theta)) < 1e-14
            and abs(y - r * math.sin(theta)) < 1e-14
        )
        expected = reference(r, theta, manufactured)
        f = (
            (-4 - 8 * x - 3 * (x * x - y * y))
            if manufactured
            else (
                -r * r * math.sin(theta / 2) ** 4
                + math.sin(6 * theta) * math.cos(theta / 2) ** 2
            )
        )
        assert abs(float(q["exact"]) - expected) < 1e-14
        assert abs(float(q["f"]) - f) < 1e-13
        assert math.isfinite(float(q["u"]))
        if r == 1:
            assert float(q["u"]) == 0
        if manufactured:
            assert abs(float(q["u"]) - expected) < 2e-10
        if q["kind"] == "node" and int(q["ir"]) > 0:
            forcing.append(f)
            error = max(error, abs(float(q["u"]) - expected))
    assert max(abs(v - f) for v, f in zip(image, forcing)) < 1e-8
    assert abs(float(diagnostics["max_error"]) - error) < 1e-14
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binaries = args.bin_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="disk-", dir=args.output) as name:
        root = Path(name).resolve()
        check_modes(binaries / "disk_eigenmodes", root / "modes")
        check_poisson(binaries / "disk_poisson", root / "manufactured", 15, True)
        coarse = check_poisson(binaries / "disk_poisson", root / "book", 31)
        fine = check_poisson(binaries / "disk_poisson", root / "fine", 63)
        assert coarse < 1e-3 and fine < coarse / 2, (coarse, fine)
        for parameters in ({"n": 24}, {"m": 19}, {"n": 65}):
            run(
                binaries / "disk_eigenmodes",
                root / "bad",
                parameters=parameters,
                success=False,
            )
        run(
            binaries / "disk_poisson",
            root / "bad-poisson",
            parameters={"m": 12},
            success=False,
        )
    print(
        "Disk modes and Poisson: complete fields, Bessel spectra, direct polar residuals and refinement passed."
    )


if __name__ == "__main__":
    main()
