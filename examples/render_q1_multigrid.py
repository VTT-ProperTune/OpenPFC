#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check native Q1 nodal fields, or plot them with Matplotlib 3.11.2."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile


def discrete_amplitude(n, diffusion, reaction):
    """Tensor-product sine eigenvalue with the analytic consistent weak load."""
    h = 1 / (n + 1)
    theta = math.pi * h
    stiffness = 4 * math.sin(theta / 2) ** 2 / h
    mass = h * (2 + math.cos(theta)) / 3
    load_integral = h * (math.sin(theta / 2) / (theta / 2)) ** 2
    load = (3 * diffusion * math.pi**2 + reaction) * load_integral**3
    return load / (3 * diffusion * stiffness * mass**2 + reaction * mass**3)


def check(folder, ranks=None):
    """Check every free node against an independently derived discrete solution."""
    run = folder / "results"
    p = json.loads((run / "run.json").read_text())
    n, h = p["n"], p["h"]
    assert n in (15, 31) and abs(h - 1 / (n + 1)) < 1e-15
    assert p["levels"] == (2 if n == 15 else 3)
    assert p["coarse_unknowns"] == 343
    assert 0 < p["iterations"] <= 120
    assert math.isfinite(p["residual"]) and p["residual"] >= 0
    if ranks is not None:
        assert p["ranks"] == ranks
    amplitude = discrete_amplitude(n, p["diffusion"], p["reaction"])
    files = sorted(run.glob("field_rank_*.csv"))
    assert len(files) == p["ranks"]
    fields, seen = [], set()
    error2 = norm2 = 0.0
    for path in files:
        with path.open() as file:
            for row in csv.DictReader(file):
                x, y, z, value, continuum = (
                    float(row[key]) for key in ("x", "y", "z", "u", "continuum")
                )
                assert all(math.isfinite(v) for v in (x, y, z, value, continuum))
                node = tuple(round(v / h) for v in (x, y, z))
                assert all(1 <= i <= n for i in node) and node not in seen
                assert all(abs(v - i * h) < 1e-14 for v, i in zip((x, y, z), node))
                seen.add(node)
                sine = math.prod(math.sin(math.pi * v) for v in (x, y, z))
                assert abs(continuum - sine) < 1e-13
                assert abs(value - amplitude * sine) < 1e-9
                error2 += (value - sine) ** 2
                norm2 += sine**2
                fields.append(row)
    assert len(seen) == n**3
    assert abs(math.sqrt(error2 / norm2) - abs(amplitude - 1)) < 1e-9
    return p, fields


def execute(binary, parent, launcher, rank_flag, ranks):
    parent.mkdir(parents=True, exist_ok=True)
    for n, reaction, diffusion in ((15, 0.0, 1.0), (31, 0.0, 1.0), (15, 1.0, 0.5)):
        with tempfile.TemporaryDirectory(prefix="q1-", dir=parent) as temp:
            folder = Path(temp)
            (folder / "input.json").write_text(
                json.dumps(dict(n=n, reaction=reaction, diffusion=diffusion))
            )
            command = [launcher, rank_flag, str(ranks)] if launcher else []
            subprocess.run(
                [*command, str(binary.resolve()), "input.json"],
                cwd=folder,
                check=True,
                timeout=60,
            )
            check(folder, ranks)
    print(f"Q1 fields pass on {ranks} ranks: independent discrete modes and ownership.")


def plot(folder, output):
    import matplotlib.pyplot as plt
    from render_utilities import save_svg

    p, fields = check(folder)
    line = sorted(
        (row for row in fields if float(row["y"]) == float(row["z"]) == 0.5),
        key=lambda row: float(row["x"]),
    )
    assert len(line) == p["n"]
    x = [float(row["x"]) for row in line]
    values = [float(row["u"]) for row in line]
    continuum = [float(row["continuum"]) for row in line]
    amplitude = discrete_amplitude(p["n"], p["diffusion"], p["reaction"])
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot(x, continuum, color="0.5", label="Continuum at nodes")
    axes[0].plot(
        x, values, "o", color="#286b83", markersize=4, label="Native Q1 coefficients"
    )
    axes[0].set(ylabel="u", title="Midline through the unit cube")
    axes[1].plot(
        x,
        [(amplitude - 1) * v for v in continuum],
        "--",
        color="0.4",
        label="Discrete-mode prediction",
    )
    axes[1].plot(
        x,
        [u - v for u, v in zip(values, continuum)],
        "o",
        color="#ab5137",
        markersize=4,
        label="Native minus continuum",
    )
    axes[1].set(
        ylabel="Nodal difference",
        title="Nodal discretization error",
    )
    for axis in axes:
        axis.set(xlabel="x at y = z = ½", xlim=(0, 1))
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
        / "finite-elements/q1-multigrid/figures/nodes.svg",
    )
    args = parser.parse_args()
    if args.ranks not in (1, 2) or (args.ranks != 1 and not args.launcher):
        parser.error("Use one rank or two with a launcher")
    if args.execute:
        execute(
            args.execute, args.run.resolve(), args.launcher, args.rank_flag, args.ranks
        )
    elif args.check:
        check(args.run)
        print("Native Q1 discrete fields and owned-node layout pass.")
    else:
        plot(args.run, args.output)


if __name__ == "__main__":
    main()
