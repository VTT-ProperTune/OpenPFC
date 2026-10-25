#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the actual Fourier utility entry points against physical basis functions."""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
from pathlib import Path

from check_foundations import run, table


def signed_modes(n):
    """Enumerate the centered integer basis, then label it by its DFT residue."""
    return {mode % n: mode for mode in range(-((n - 1) // 2), n // 2 + 1)}


def check_symbols(binary, root):
    for case, shape, length in (
        ("mixed", [8, 9, 6], [2 * math.pi, 3 * math.pi, math.pi]),
        ("odd", [7, 5, 9], [3.2, 4.1, 8.3]),
        ("scaled", [8, 9, 6], [4 * math.pi, 6 * math.pi, 2 * math.pi]),
    ):
        folder = root / case
        dt = 0.13
        run(binary, folder, parameters={"shape": shape, "length": length, "dt": dt})
        axes = table(folder / "results/axes.csv")
        assert len(axes) == sum(shape)
        seen = set()
        for row in axes:
            axis, index = int(row["axis"]), int(row["index"])
            assert (axis, index) not in seen
            seen.add((axis, index))
            n, period = shape[axis], length[axis]
            wave = 2 * math.pi * signed_modes(n)[index] / period
            assert abs(float(row["k"]) - wave) < 2e-14
            # Check that the wave labels reproduce the samples of the DFT basis.
            for j in range(n):
                physical = cmath.exp(1j * float(row["k"]) * period * j / n)
                indexed = cmath.exp(2j * math.pi * index * j / n)
                assert abs(physical - indexed) < 3e-14
            nyquist = n % 2 == 0 and index == n // 2
            assert abs(float(row["k_odd"]) - (0 if nyquist else wave)) < 2e-14
            assert abs(float(row["laplacian"]) + wave**2) < 2e-13

        data = table(folder / "results/symbols.csv")
        nx = shape[0] // 2 + 1
        assert len(data) == nx * shape[1] * shape[2]
        seen = set()
        for row in data:
            index = tuple(int(row[key]) for key in ("i", "j", "k"))
            assert index not in seen
            seen.add(index)
            i, j, k = index
            assert 0 <= i < nx and 0 <= j < shape[1] and 0 <= k < shape[2]
            assert int(row["offset"]) == i + nx * (j + shape[1] * k)
            waves = [
                2 * math.pi * signed_modes(n)[q] / period
                for q, n, period in zip(index, shape, length)
            ]
            for key, wave in zip(("kx", "ky", "kz"), waves):
                assert abs(float(row[key]) - wave) < 2e-14
            lap = -sum(wave**2 for wave in waves)
            assert abs(float(row["laplacian"]) - lap) < 4e-13
            implicit, exponential = float(row["implicit"]), float(row["exponential"])
            assert abs(implicit - 1 / (1 - dt * lap)) < 3e-15
            assert abs(exponential - math.exp(dt * lap)) < 3e-15
            assert 0 <= exponential <= implicit <= 1
    for name, parameters in (
        ("bad-shape", {"shape": [2, 8, 8]}),
        ("bad-length", {"length": [1, 0, 1]}),
        ("bad-step", {"dt": -1}),
    ):
        run(binary, root / name, parameters=parameters, success=False)


def periodic_fourier(x, time, p):
    """Continuous Fourier series, independent of the model's image sum."""
    length, width, diffusivity = p["length"], p["width"], p["diffusivity"]
    variance = width**2 + 2 * diffusivity * time
    cutoff = math.ceil(length / (2 * math.pi * math.sqrt(variance)) * 10)
    scale = math.sqrt(2 * math.pi) * width / length
    return scale * (
        1
        + 2
        * sum(
            math.exp(-variance * (2 * math.pi * m / length) ** 2 / 2)
            * math.cos(2 * math.pi * m * x / length)
            for m in range(1, cutoff + 1)
        )
    )


def backward_euler_reference(p):
    """Defining DFT sums in three tensor stages; no FFT library or image sum."""
    n, length = p["n"], p["length"]
    line = [periodic_fourier(-length / 2 + length * j / n, 0, p) for j in range(n)]
    phase = [[cmath.exp(2j * math.pi * j * m / n) for m in range(n)] for j in range(n)]
    coefficient = [
        sum(line[j] * phase[j][m].conjugate() for j in range(n)) / n for m in range(n)
    ]
    waves = {m: 2 * math.pi * q / length for m, q in signed_modes(n).items()}
    indices = list(itertools.product(range(n), repeat=3))
    values = {
        index: math.prod(coefficient[m] for m in index)
        / (1 + p["dt"] * p["diffusivity"] * sum(waves[m] ** 2 for m in index))
        ** p["steps"]
        for index in indices
    }
    for axis in range(3):
        values = {
            index: sum(
                values[index[:axis] + (m,) + index[axis + 1 :]] * phase[index[axis]][m]
                for m in range(n)
            )
            for index in indices
        }
    assert max(abs(value.imag) for value in values.values()) < 3e-13
    return {index: value.real for index, value in values.items()}


def check_gaussian(binary, root, launch, ranks):
    cases = [
        (
            "even",
            {
                "n": 8,
                "length": 6,
                "width": 0.8,
                "diffusivity": 0.7,
                "dt": 0.03,
                "steps": 6,
            },
        ),
        (
            "odd",
            {
                "n": 9,
                "length": 7,
                "width": 0.75,
                "diffusivity": 1.3,
                "dt": 0.02,
                "steps": 7,
            },
        ),
    ]
    for dt in (0.02, 0.01, 0.005):
        cases.append(
            (
                f"refine-{dt}",
                {
                    "n": 24,
                    "length": 8,
                    "width": 0.75,
                    "diffusivity": 1,
                    "dt": dt,
                    "steps": round(0.2 / dt),
                },
            )
        )
    errors = []
    for name, p in cases:
        folder = root / name
        run(binary, folder, launch, p)
        expected = backward_euler_reference(p)
        paths = sorted((folder / "results").glob("field_rank_*.csv"))
        assert len(paths) == ranks
        rows = [row for path in paths for row in table(path)]
        n, length = p["n"], p["length"]
        time, h = p["dt"] * p["steps"], length / n
        seen = set()
        initial_mass = mass = squared_error = 0
        max_error = 0
        for row in rows:
            xyz = [float(row[key]) for key in ("x", "y", "z")]
            index = tuple(round((x + length / 2) / h) for x in xyz)
            assert index in expected and index not in seen, index
            assert all(
                abs(x - (-length / 2 + j * h)) < 2e-14 for x, j in zip(xyz, index)
            )
            seen.add(index)
            initial = math.prod(periodic_fourier(x, 0, p) for x in xyz)
            continuum = math.prod(periodic_fourier(x, time, p) for x in xyz)
            value = float(row["u"])
            assert abs(float(row["initial"]) - initial) < 2e-14
            assert abs(float(row["exact"]) - continuum) < 2e-14
            assert abs(value - expected[index]) < 2e-13, (
                name,
                index,
                value,
                expected[index],
            )
            initial_mass += initial * h**3
            mass += value * h**3
            squared_error += (value - continuum) ** 2
            max_error = max(max_error, abs(value - continuum))
        assert len(seen) == n**3
        assert abs(mass - initial_mass) < 4e-12 * initial_mass
        summary = json.loads((folder / "results/summary.json").read_text())
        assert summary["ranks"] == ranks and abs(summary["time"] - time) < 1e-15
        assert abs(summary["mass"] - mass) < 4e-12 * mass
        assert abs(summary["initial_mass"] - initial_mass) < 4e-12 * initial_mass
        assert abs(summary["rms_error"] - math.sqrt(squared_error / n**3)) < 3e-14
        assert abs(summary["max_error"] - max_error) < 3e-14
        if name.startswith("refine"):
            errors.append(max_error)
    for coarse, fine in itertools.pairwise(errors):
        assert 1.8 < coarse / fine < 2.2, errors
    for name, parameters in (
        ("bad-width", {"width": 0}),
        ("bad-time", {"dt": 100}),
        ("bad-size", {"n": 2}),
    ):
        run(binary, root / name, launch, parameters, success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("symbols", "gaussian"), required=True)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    binaries, output = args.bin_dir.resolve(), args.output.resolve()
    if args.case == "symbols":
        check_symbols(
            binaries / "fft_kspace_helpers_example", output / "fourier-symbols"
        )
    else:
        launch = (
            [args.launcher, args.rank_flag, str(args.ranks)] if args.launcher else []
        )
        check_gaussian(
            binaries / "diffusion_model_with_custom_initial_condition",
            output / f"gaussian-{args.ranks}ranks",
            launch,
            args.ranks,
        )
    print(args.case, "physical-basis reference checks pass.")


if __name__ == "__main__":
    main()
