# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify the teaching model using direct Fourier convolution and classical RK4.

This reference uses neither FFTs nor exponential time differencing. It checks
one or two results/ directories against a refined ODE solution at the SAME
retained cutoff K, not an exact solution of the infinite-dimensional PDE.
Python 3 standard library only. Default error budget 2e-5 suits input.json.
"""

import argparse
import cmath
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def convolution(left, right):
    """Full, non-circular coefficient convolution, without discarding intermediates."""
    result = [0j] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def initial(settings):
    cutoff = settings["cutoff"]
    values = [0j] * (2 * cutoff + 1)
    values[cutoff] = settings["mean"]
    for mode in [1, 3]:
        values[cutoff - mode] = values[cutoff + mode] = settings[f"a{mode}"] / 2
    return values


def reference(settings, steps=2048):
    """RK4 for the projected Fourier ODE; steps can be doubled to check the oracle."""
    cutoff = settings["cutoff"]
    dt = settings["steps"] * settings["dt"] / steps
    linear = [
        1 - settings["diffusivity"] * mode**2 for mode in range(-cutoff, cutoff + 1)
    ]

    def rhs(values):
        # Keep the full quadratic intermediate: early truncation changes u^3.
        cubic = convolution(convolution(values, values), values)
        return [
            symbol * value - cubic[2 * cutoff + i]
            for i, (symbol, value) in enumerate(zip(linear, values))
        ]

    def add(values, rate, factor):
        return [value + factor * delta for value, delta in zip(values, rate)]

    values = initial(settings)
    for _ in range(steps):
        a = rhs(values)
        b = rhs(add(values, a, dt / 2))
        c = rhs(add(values, b, dt / 2))
        d = rhs(add(values, c, dt))
        values = [
            value + dt * (ra + 2 * rb + 2 * rc + rd) / 6
            for value, ra, rb, rc, rd in zip(values, a, b, c, d)
        ]
    return values


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_owned(results, prefix, columns, shape, ranks):
    values = {}
    files = list(results.glob(f"{prefix}_rank_*.csv"))
    require(
        {p.name for p in files}
        == {f"{prefix}_rank_{rank}.csv" for rank in range(ranks)},
        f"missing/extra {prefix} rank files",
    )
    for rank in range(ranks):
        layout = json.loads((results / f"layout_rank_{rank}.json").read_text())
        name = "inbox" if prefix == "field" else "outbox"
        require(layout["rank"] == rank, "wrong layout rank")
        low, high = layout[f"{name}_low"], layout[f"{name}_high"]
        expected = set(itertools.product(*(range(a, b + 1) for a, b in zip(low, high))))
        actual = set()
        for row in read_csv(results / f"{prefix}_rank_{rank}.csv"):
            point = tuple(int(row[axis]) for axis in ["i", "j", "k"])
            require(
                all(0 <= index < size for index, size in zip(point, shape)),
                "coordinate outside global grid",
            )
            require(point not in values, "duplicate owned coordinate")
            parts = tuple(float(row[column]) for column in columns)
            require(all(math.isfinite(value) for value in parts), "non-finite output")
            values[point] = parts[0] if len(parts) == 1 else complex(*parts)
            actual.add(point)
        require(actual == expected, f"{name} coordinates disagree with layout")
    require(len(values) == math.prod(shape), f"incomplete {prefix} ownership")
    return values


def check(settings, results, oracle, tolerance=2e-5):
    n, cutoff = settings["n"], settings["cutoff"]
    summary = json.loads((results / "summary.json").read_text())
    ranks = summary["ranks"]
    require(summary["n"] == n and summary["cutoff"] == cutoff, "wrong grid/cutoff")
    require(summary["steps"] == settings["steps"], "wrong steps")
    require(
        abs(summary["time"] - settings["steps"] * settings["dt"]) < 1e-12,
        "wrong final time",
    )
    field = read_owned(results, "field", ["u"], [n, 4, 4], ranks)
    modes = read_owned(results, "modes", ["re", "im"], [n // 2 + 1, 4, 4], ranks)
    require(
        max(
            (abs(value) for (i, j, k), value in modes.items() if i > cutoff or j or k),
            default=0,
        )
        < 1e-12,
        "unprojected modes",
    )
    require(abs(modes[(0, 0, 0)].imag) < 1e-12, "nonreal DC coefficient")
    coefficients = [
        modes[(abs(mode), 0, 0)].conjugate() if mode < 0 else modes[(mode, 0, 0)]
        for mode in range(-cutoff, cutoff + 1)
    ]
    reconstruction_error, reference_error = 0.0, 0.0
    for (i, _, _), value in field.items():
        phase = [
            cmath.exp(2j * math.pi * mode * i / n)
            for mode in range(-cutoff, cutoff + 1)
        ]
        reconstructed = sum(a * b for a, b in zip(coefficients, phase))
        expected = sum(a * b for a, b in zip(oracle, phase))
        reconstruction_error = max(reconstruction_error, abs(value - reconstructed))
        reference_error = max(reference_error, abs(value - expected))
    require(reconstruction_error < 2e-11, "FFT normalization/reconstruction mismatch")
    require(
        reference_error < tolerance,
        f"reference error {reference_error:.6g} exceeds chosen budget {tolerance:.6g}",
    )
    mean = math.fsum(field.values()) / len(field)
    rms = math.sqrt(math.fsum(value**2 for value in field.values()) / len(field))
    require(
        abs(mean - summary["mean"]) < 2e-11 and abs(rms - summary["rms"]) < 2e-11,
        "summary/global field mismatch",
    )
    history = read_csv(results / "diagnostics.csv")
    require(len(history) == settings["steps"] + 1, "wrong history length")
    for step, row in enumerate(history):
        require(
            int(row["step"]) == step
            and abs(float(row["time"]) - step * settings["dt"]) < 1e-12,
            "wrong diagnostic time/step",
        )
        require(
            all(
                math.isfinite(float(row[key]))
                for key in ["mean", "rms", "energy_density"]
            ),
            "non-finite diagnostic",
        )
    parseval = sum(abs(value) ** 2 for value in coefficients)
    require(abs(parseval - rms**2) < 2e-11, "Parseval mismatch")
    gradient_energy = sum(
        mode**2 * abs(value) ** 2
        for mode, value in zip(range(-cutoff, cutoff + 1), coefficients)
    )
    potential_energy = math.fsum(
        (value**2 - 1) ** 2 / 4 for value in field.values()
    ) / len(field)
    energy = settings["diffusivity"] * gradient_energy / 2 + potential_energy
    require(
        abs(energy - summary["energy_density"]) < 2e-11, "energy normalization mismatch"
    )
    return field, {
        "ranks": ranks,
        "owned_cells": len(field),
        "complex_cells": len(modes),
        "mean": mean,
        "rms": rms,
        "energy_density": energy,
        "max_reconstruction_error": reconstruction_error,
        "max_reference_error": reference_error,
        "max_coefficient_reference_error": max(
            abs(a - b) for a, b in zip(coefficients, oracle)
        ),
        "mean_change": mean - settings["mean"],
        "energy_change": float(history[-1]["energy_density"])
        - float(history[0]["energy_density"]),
    }


def self_test(executable, ranks, mpiexec):
    """Run real entry points: cubic dynamics, a zero linear symbol and a uniform oracle."""
    settings = {
        "n": 16,
        "cutoff": 3,
        "steps": 20,
        "dt": 0.01,
        "diffusivity": 0.1,
        "mean": 0.2,
        "a1": 0.35,
        "a3": 0.15,
    }
    Path("results").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spectral-check-", dir="results") as temp:
        for case in ["nonlinear", "zero-symbol", "uniform"]:
            current = dict(settings)
            if case == "zero-symbol":
                current["diffusivity"] = 1.0
            if case == "uniform":
                current.update(a1=0.0, a3=0.0)
            directory = Path(temp).resolve() / case
            directory.mkdir()
            (directory / "input.json").write_text(json.dumps(current))
            command = ([mpiexec, "-n", str(ranks)] if ranks > 1 else []) + [
                str(executable),
                "input.json",
            ]
            subprocess.run(command, cwd=directory, check=True, timeout=25)
            oracle = reference(current, steps=512)
            _, report = check(current, directory / "results", oracle)
            if case == "uniform":
                time = current["dt"] * current["steps"]
                exact = current["mean"] / math.sqrt(
                    current["mean"] ** 2
                    + (1 - current["mean"] ** 2) * math.exp(-2 * time)
                )
                require(
                    abs(oracle[current["cutoff"]] - exact) < 1e-12,
                    "independent uniform ODE oracle failed",
                )
            print(case, json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("results", type=Path, nargs="*")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-5,
        help="absolute field-error budget versus refined RK4; this is time-discretization error, not roundoff",
    )
    parser.add_argument("--reference-steps", type=int, default=2048)
    parser.add_argument("--self-test", type=Path, metavar="EXECUTABLE")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.self_test.resolve(), args.ranks, args.mpiexec)
        return
    require(
        args.input is not None and 1 <= len(args.results) <= 2,
        "supply input.json and one or two results/ directories",
    )
    require(
        args.reference_steps >= 256
        and math.isfinite(args.tolerance)
        and args.tolerance > 0,
        "invalid reference steps/tolerance",
    )
    settings = json.loads(args.input.read_text())
    require(
        3 <= settings["cutoff"] <= 31 and 4 * settings["cutoff"] < settings["n"],
        "invalid retained cutoff",
    )
    coarse = reference(settings, args.reference_steps)
    oracle = reference(settings, 2 * args.reference_steps)
    oracle_change = max(abs(a - b) for a, b in zip(coarse, oracle))
    require(
        oracle_change < min(1e-9, args.tolerance / 100), "refine the RK4 oracle further"
    )
    checked = [
        check(settings, directory, oracle, args.tolerance) for directory in args.results
    ]
    report = {
        "reference": "direct non-circular cubic convolution + refined RK4 at fixed K",
        "oracle_step_doubling_change": oracle_change,
        "reference_steps": 2 * args.reference_steps,
        "field_error_budget": args.tolerance,
        "runs": [item[1] for item in checked],
    }
    if len(checked) == 2:
        left, right = checked[0][0], checked[1][0]
        require(left.keys() == right.keys(), "ownership differs across rank counts")
        error = max(abs(left[point] - right[point]) for point in left)
        require(error < 2e-11, "fields differ across rank counts")
        report["max_rank_count_difference"] = error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
