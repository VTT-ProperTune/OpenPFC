#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent complete-field checks for waves, KdV and Allen–Cahn."""
from __future__ import annotations

import argparse
import cmath
import math
from pathlib import Path
import tempfile

from check_foundations import run, table
from check_spectra import lagrange_second


def snapshots(path, kind=None):
    data = table(path)
    if kind:
        data = [row for row in data if row["kind"] == kind]
    return {
        step: [float(row["u"]) for row in data if int(row["step"]) == step]
        for step in sorted({int(row["step"]) for row in data})
    }


def close(actual, expected, tolerance=2e-10):
    error = max(abs(a - b) for a, b in zip(actual, expected))
    assert len(actual) == len(expected) and error < tolerance, error


def apply(matrix, values):
    return [sum(a * b for a, b in zip(row, values)) for row in matrix]


def lagrange_first(x):
    """Differentiate the nodal Lagrange products, without FFT coefficients."""
    weights = [
        1 / math.prod(point - other for k, other in enumerate(x) if k != j)
        for j, point in enumerate(x)
    ]
    return [
        [
            (
                sum(1 / (point - other) for k, other in enumerate(x) if k != i)
                if i == j
                else weights[j] / weights[i] / (point - x[j])
            )
            for j in range(len(x))
        ]
        for i, point in enumerate(x)
    ]


def check_line(binary, root):
    folder = root / "line-reference"
    n, dt = 12, 0.001
    run(binary, folder, parameters={"n": n, "dt": dt, "end": 0.03, "saveat": 0.01})
    saved = snapshots(folder / "results/wave.csv")
    x, d2 = lagrange_second(n)
    u = [math.exp(-200 * p * p) for p in x]
    old = [math.exp(-200 * (p - dt) ** 2) for p in x]
    close(saved[0], u)
    for step in range(1, 31):
        acceleration = apply(d2, u)
        acceleration[0] = acceleration[-1] = 0
        u, old = [2 * a - b + dt * dt * c for a, b, c in zip(u, old, acceleration)], u
        if step in saved:
            close(saved[step], u)
    errors = []
    for dt in (0.004, 0.002):
        folder = root / f"line-mode-{dt}"
        run(
            binary,
            folder,
            parameters={
                "n": 32,
                "dt": dt,
                "end": 0.4,
                "saveat": 0.4,
                "initial": "mode",
                "startup": "taylor",
            },
        )
        data = table(folder / "results/wave.csv")
        final = [r for r in data if float(r["time"]) > 0.39]
        errors.append(
            max(
                abs(
                    float(r["u"])
                    - math.cos(math.pi * float(r["x"]) / 2)
                    * math.cos(math.pi * 0.4 / 2)
                )
                for r in final
            )
        )
    assert 3.9 < errors[0] / errors[1] < 4.1, errors
    run(binary, root / "bad-line", parameters={"n": 32, "dt": 1}, success=False)


def square_action(d2, values):
    width = len(d2)
    out = [0.0] * len(values)
    for j in range(1, width - 1):
        for i in range(1, width - 1):
            out[i + width * j] = sum(
                d2[i][k] * values[k + width * j] + d2[j][k] * values[i + width * k]
                for k in range(width)
            )
    return out


def check_square(binary, root):
    folder = root / "square-reference"
    n, dt = 8, 0.001
    run(binary, folder, parameters={"n": n, "dt": dt, "end": 0.02, "saveat": 0.01})
    saved = snapshots(folder / "results/wave.csv")
    x, d2 = lagrange_second(n)
    u = [math.exp(-40 * ((a - 0.4) ** 2 + b * b)) for b in x for a in x]
    old = u[:]
    close(saved[0], u)
    for step in range(1, 21):
        acceleration = square_action(d2, u)
        u, old = [2 * a - b + dt * dt * c for a, b, c in zip(u, old, acceleration)], u
        if step in saved:
            close(saved[step], u)
    for startup, expected_ratio in (("book", 2), ("taylor", 4)):
        errors = []
        for dt in (0.004, 0.002):
            folder = root / f"square-mode-{startup}-{dt}"
            run(
                binary,
                folder,
                parameters={
                    "n": 24,
                    "dt": dt,
                    "end": 0.4,
                    "saveat": 0.4,
                    "initial": "mode",
                    "startup": startup,
                },
            )
            data = table(folder / "results/wave.csv")
            errors.append(
                max(
                    abs(
                        float(r["u"])
                        - math.cos(math.pi * float(r["x"]) / 2)
                        * math.cos(math.pi * float(r["y"]) / 2)
                        * math.cos(math.pi * 0.4 / math.sqrt(2))
                    )
                    for r in data
                    if float(r["time"]) > 0.39
                )
            )
        assert (
            0.95 * expected_ratio < errors[0] / errors[1] < 1.05 * expected_ratio
        ), errors
    run(binary, root / "bad-square", parameters={"startup": "unknown"}, success=False)


def check_tank(binary, root):
    folder = root / "tank-reference"
    nx, ny, length, dt = 12, 8, 3, 0.001
    run(
        binary,
        folder,
        parameters={"nx": nx, "ny": ny, "dt": dt, "end": 0.02, "saveat": 0.01},
    )
    saved = snapshots(folder / "results/wave.csv")
    y, d2 = lagrange_second(ny)
    dy = lagrange_first(y)
    x = [-length + 2 * length * (i + 1) / nx for i in range(nx)]
    frequencies = [j if j <= nx // 2 else j - nx for j in range(nx)]
    dx2 = [
        [
            sum(
                -((math.pi * k / length) ** 2)
                * cmath.exp(2j * math.pi * k * (i - j) / nx)
                for k in frequencies
            ).real
            / nx
            for j in range(nx)
        ]
        for i in range(nx)
    ]
    determinant = dy[0][0] * dy[-1][-1] - dy[0][-1] * dy[-1][0]

    def project(v):
        for i in range(nx):
            a = -sum(dy[0][j] * v[i + nx * j] for j in range(1, ny))
            b = -sum(dy[-1][j] * v[i + nx * j] for j in range(1, ny))
            v[i] = (a * dy[-1][-1] - dy[0][-1] * b) / determinant
            v[i + nx * ny] = (dy[0][0] * b - dy[-1][0] * a) / determinant

    u = [math.exp(-8 * ((a + 1.5) ** 2 + b * b)) for b in y for a in x]
    old = [math.exp(-8 * ((a + dt + 1.5) ** 2 + b * b)) for b in y for a in x]
    close(saved[0], u)
    for step in range(1, 21):
        acceleration = [
            sum(dx2[i][k] * u[k + nx * j] for k in range(nx))
            + sum(d2[j][k] * u[i + nx * k] for k in range(ny + 1))
            for j in range(ny + 1)
            for i in range(nx)
        ]
        u, old = [2 * a - b + dt * dt * c for a, b, c in zip(u, old, acceleration)], u
        project(u)
        if step in saved:
            close(saved[step], u)
            for i in range(nx):
                for endpoint in (0, ny):
                    assert (
                        abs(sum(dy[endpoint][j] * u[i + nx * j] for j in range(ny + 1)))
                        < 1e-10
                    )
    errors = []
    for dt in (0.004, 0.002):
        folder = root / f"tank-mode-{dt}"
        run(
            binary,
            folder,
            parameters={
                "nx": 24,
                "ny": 18,
                "half_length": 2,
                "dt": dt,
                "end": 0.4,
                "saveat": 0.4,
                "initial": "mode",
                "startup": "taylor",
            },
        )
        data = table(folder / "results/wave.csv")
        omega = math.sqrt((math.pi / 2) ** 2 + math.pi**2)
        errors.append(
            max(
                abs(
                    float(r["u"])
                    - math.cos(math.pi * float(r["x"]) / 2)
                    * math.cos(math.pi * float(r["y"]))
                    * math.cos(omega * 0.4)
                )
                for r in data
                if float(r["time"]) > 0.39
            )
        )
    assert 3.9 < errors[0] / errors[1] < 4.1, errors
    run(binary, root / "bad-tank", parameters={"nx": 13}, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument(
        "--case",
        choices=["chebyshev_wave_line", "chebyshev_wave_square", "mixed_wave_tank"],
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
        if args.case in (None, "chebyshev_wave_line"):
            root = base / "chebyshev_wave_line"
            root.mkdir()
            check_line(bin_dir / "chebyshev_wave_line", root)
        if args.case in (None, "chebyshev_wave_square"):
            root = base / "chebyshev_wave_square"
            root.mkdir()
            check_square(bin_dir / "chebyshev_wave_square", root)
        if args.case in (None, "mixed_wave_tank"):
            root = base / "mixed_wave_tank"
            root.mkdir()
            check_tank(bin_dir / "mixed_wave_tank", root)
    print(f"Selected teaching checks passed ({ranks} rank(s)).")


if __name__ == "__main__":
    main()
