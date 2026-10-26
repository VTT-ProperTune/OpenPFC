#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check or plot native SBP diffusion output; plotting uses Matplotlib 3.11.2."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def check(folder):
    """Check every node and budget row against the analytic quadratic solution."""
    run = folder / "results"
    p = json.loads((run / "run.json").read_text())
    n, diffusion, amplitude, final_time = (
        p[key] for key in ("n", "diffusivity", "amplitude", "final_time")
    )
    assert p["boundary"] in ("flux", "derivative")
    assert p["steps"] > 0 and math.isclose(p["steps"] * p["dt"], final_time)
    fields, budget = table(run / "field.csv"), table(run / "budget.csv")
    assert len(fields) == n and len(budget) == p["steps"] + 1
    weights = (17 / 48, 59 / 48, 43 / 48, 49 / 48)
    measured_mass = 0.0
    values = []
    for i, row in enumerate(fields):
        assert int(row["i"]) == i
        x, weight, initial, value = (
            float(row[key]) for key in ("x", "weight", "initial", "u")
        )
        assert all(math.isfinite(v) for v in (x, weight, initial, value))
        edge = min(i, n - 1 - i)
        expected_weight = (weights[edge] if edge < 4 else 1) / (n - 1)
        assert abs(x - i / (n - 1)) < 1e-14
        assert abs(weight - expected_weight) < 1e-14
        expected_initial = 1 + amplitude * x * (1 - x)
        assert abs(initial - expected_initial) < 1e-13
        assert (
            abs(value - (expected_initial - 2 * diffusion * amplitude * final_time))
            < 1e-9
        )
        measured_mass += weight * value
        values.append(value)
    initial_mass = 1 + amplitude / 6
    assert (
        abs(measured_mass - (initial_mass - 2 * diffusion * amplitude * final_time))
        < 5e-12
    )
    for step, row in enumerate(budget):
        assert int(row["step"]) == step
        time, mass, outward, residual = (
            float(row[key])
            for key in ("time", "mass", "outward_integral", "balance_error")
        )
        assert all(math.isfinite(v) for v in (time, mass, outward, residual))
        assert abs(time - step * p["dt"]) < 1e-14
        assert abs(mass - (initial_mass - 2 * diffusion * amplitude * time)) < 5e-12
        assert abs(outward - 2 * diffusion * amplitude * time) < 5e-12
        assert abs(mass + outward - initial_mass) < 5e-12
        assert abs(residual - (mass + outward - initial_mass)) < 5e-12
    assert abs(measured_mass - float(budget[-1]["mass"])) < 5e-12
    return p, fields, budget, values


def execute(binary, parent):
    """Invoke the actual model with equivalent flux/derivative data and both signs."""
    parent.mkdir(parents=True, exist_ok=True)
    for n, amplitude, diffusion, final_time in (
        (17, 0.5, 0.2, 0.1),
        (33, -0.4, 0.3, 0.05),
        (65, 0.7, 0.05, 0.02),
    ):
        previous = None
        for boundary in ("flux", "derivative"):
            with tempfile.TemporaryDirectory(prefix="sbp-", dir=parent) as temp:
                folder = Path(temp)
                parameters = dict(
                    n=n,
                    amplitude=amplitude,
                    diffusivity=diffusion,
                    final_time=final_time,
                    boundary=boundary,
                )
                (folder / "input.json").write_text(json.dumps(parameters))
                subprocess.run(
                    [str(binary.resolve()), "input.json"],
                    cwd=folder,
                    check=True,
                    timeout=30,
                )
                _, _, _, values = check(folder)
                if previous is not None:
                    assert max(abs(a - b) for a, b in zip(values, previous)) < 1e-12
                previous = values
    print(
        "Six native SBP cases pass: complete fields, mass budgets and boundary parity."
    )


def plot(folder, output):
    import matplotlib.pyplot as plt
    from render_utilities import save_svg

    p, fields, budget, _ = check(folder)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    x = [float(row["x"]) for row in fields]
    axes[0].plot(
        x, [float(row["initial"]) for row in fields], color="0.6", label="Initial"
    )
    axes[0].plot(
        x,
        [float(row["u"]) for row in fields],
        "o-",
        markersize=3,
        color="#286b83",
        label=f"Native at t = {p['final_time']:g}",
    )
    axes[0].set(xlabel="x", ylabel="u", title="Equal outward flux at both ends")
    time = [float(row["time"]) for row in budget]
    mass = [float(row["mass"]) for row in budget]
    outward = [float(row["outward_integral"]) for row in budget]
    axes[1].plot(time, mass, color="#286b83", label="Weighted mass")
    axes[1].plot(
        time,
        [m + j for m, j in zip(mass, outward)],
        "--",
        color="#ab5137",
        label="Mass + outward integral",
    )
    axes[1].set(
        xlabel="Time", ylabel="Inventory", title="Track what leaves the interval"
    )
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    save_svg(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "boundaries/outward-flux/figures/budget.svg",
    )
    args = parser.parse_args()
    if args.execute:
        execute(args.execute, args.run.resolve())
    elif args.check:
        check(args.run)
        print("Native SBP fields and mass budget pass.")
    else:
        plot(args.run, args.output)


if __name__ == "__main__":
    main()
