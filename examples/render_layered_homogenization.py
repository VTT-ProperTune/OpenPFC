#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check native layered elasticity fields, or plot them with Matplotlib 3.11.2."""

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


def layered_tensor(n, amplitude):
    """Independent layered solution in engineering Voigt order 11,22,33,23,13,12."""
    arithmetic = (
        sum(1 / (1 + amplitude * math.cos(2 * math.pi * i / n)) for i in range(n)) / n
    )
    exact = [[0.0] * 6 for _ in range(6)]
    exact[0][:3] = [3, 1, 1]
    exact[1][:3] = [1, (1 + 8 * arithmetic) / 3, (1 + 2 * arithmetic) / 3]
    exact[2][:3] = [1, (1 + 2 * arithmetic) / 3, (1 + 8 * arithmetic) / 3]
    exact[3][3] = arithmetic
    exact[4][4] = exact[5][5] = 1
    return exact


def check(folder, ranks=None):
    """Check all six load columns and every owned shear node against layered equilibrium."""
    run = folder / "results"
    p = json.loads((run / "run.json").read_text())
    n, amplitude = p["n"], p["amplitude"]
    if ranks is not None:
        assert p["ranks"] == ranks
    assert p["voigt_order"] == ["11", "22", "33", "23", "13", "12"]
    tensor, seen_loads = table(run / "stiffness.csv"), set()
    exact = layered_tensor(n, amplitude)
    for row in tensor:
        component, load = int(row["component"]), int(row["load"])
        assert 0 <= component < 6 and 0 <= load < 6
        assert (component, load) not in seen_loads
        seen_loads.add((component, load))
        for key in ("stiffness", "mean_stress"):
            value = float(row[key])
            assert math.isfinite(value) and abs(value - exact[component][load]) < 1e-8
        macro_strain = (1.0 if load < 3 else 0.5) if component == load else 0.0
        assert abs(float(row["mean_strain"]) - macro_strain) < 1e-10
        assert int(row["iterations"]) <= 200
        assert 0 <= float(row["residual"]) < p["tolerance"] == 1e-11
    assert len(seen_loads) == 36
    fields, seen_nodes = [], set()
    files = sorted(run.glob("field_rank_*.csv"))
    assert len(files) == p["ranks"]
    for path in files:
        for row in table(path):
            x, y, z, weight, strain, stress = (
                float(row[key])
                for key in ("x", "y", "z", "weight", "epsilon_xy", "sigma_xy")
            )
            assert all(math.isfinite(v) for v in (x, y, z, weight, strain, stress))
            node = tuple(round(value * n / (2 * math.pi)) for value in (x, y, z))
            assert all(0 <= i < n for i in node) and node not in seen_nodes
            assert all(
                abs(value - 2 * math.pi * i / n) < 1e-13
                for value, i in zip((x, y, z), node)
            )
            seen_nodes.add(node)
            mu = 1 / (1 + amplitude * math.cos(x))
            soft, hard = 1 / (1 + amplitude), 1 / (1 - amplitude)
            expected_weight = 0 if amplitude == 0 else (mu - soft) / (hard - soft)
            assert abs(weight - expected_weight) < 1e-12
            assert abs(strain - 0.5 * (1 + amplitude * math.cos(x))) < 1e-8
            assert abs(stress - 1) < 1e-8
            fields.append(row)
    assert len(seen_nodes) == n**3
    return p, fields


def execute(binary, parent, launcher, rank_flag, ranks):
    parent.mkdir(parents=True, exist_ok=True)
    for n, amplitude in ((8, 0.3), (16, 0.3), (8, 0.0)):
        with tempfile.TemporaryDirectory(prefix="layers-", dir=parent) as temp:
            folder = Path(temp)
            (folder / "input.json").write_text(
                json.dumps(dict(n=n, amplitude=amplitude))
            )
            command = [launcher, rank_flag, str(ranks)] if launcher else []
            subprocess.run(
                [*command, str(binary.resolve()), "input.json"],
                cwd=folder,
                check=True,
                timeout=60,
            )
            check(folder, ranks)
    print(
        f"Layered elasticity passes on {ranks} ranks: every load, shear node and owner."
    )


def plot(folder, output):
    import matplotlib.pyplot as plt
    from render_utilities import save_svg

    p, fields = check(folder)
    line = sorted(
        (row for row in fields if float(row["y"]) == float(row["z"]) == 0),
        key=lambda row: float(row["x"]),
    )
    assert len(line) == p["n"]
    x = [float(row["x"]) for row in line]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot(
        x,
        [0.5 * (1 + p["amplitude"] * math.cos(v)) for v in x],
        "--",
        color="0.3",
        label="Layered solution",
    )
    axes[0].plot(
        x,
        [float(row["epsilon_xy"]) for row in line],
        "o",
        color="#286b83",
        label="Native tensor strain",
    )
    axes[0].set(ylabel="ε₁₂", title="Soft layers carry more strain")
    axes[1].axhline(1, color="0.3", linestyle="--", label="Layered solution")
    axes[1].plot(
        x,
        [float(row["sigma_xy"]) for row in line],
        "o",
        color="#ab5137",
        label="Native shear stress",
    )
    axes[1].set(
        ylabel="σ₁₂", title="Equilibrium keeps shear stress constant", ylim=(0.8, 1.2)
    )
    for axis in axes:
        axis.set(
            xlabel="x",
            xticks=(0, math.pi, 2 * math.pi),
            xticklabels=("0", "π", "2π"),
            xlim=(0, 2 * math.pi),
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    save_svg(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "materials/layered-homogenization/figures/shear.svg",
    )
    args = parser.parse_args()
    if args.ranks < 1 or (args.ranks != 1 and not args.launcher):
        parser.error("Multiple ranks require a launcher; rank count must be positive")
    if args.execute:
        execute(
            args.execute, args.run.resolve(), args.launcher, args.rank_flag, args.ranks
        )
    elif args.check:
        check(args.run)
        print("Native layered tensor and complete shear fields pass.")
    else:
        plot(args.run, args.output)


if __name__ == "__main__":
    main()
