#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw the lesson figures from CSVs written by the C++ examples.

Run with Matplotlib 3.11.2. Numerical execution is deliberately separate
from Quarto rendering. Input CSVs and bulk run output stay under results/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read(path: Path) -> list[dict[str, float]]:
    with path.open() as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--differentiation", type=Path, required=True)
    parser.add_argument("--fd", type=Path, required=True)
    parser.add_argument("--fourier", type=Path, required=True)
    parser.add_argument("--exponential", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#343b40", "text.color": "#343b40",
                         "svg.fonttype": "none", "svg.hashsalt": "openpfc-lessons"})

    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    rows = read(args.differentiation)
    for key, label, colour, marker in (("fd2", "Second-order differences", "#526a82", "o"),
                                      ("fd4", "Fourth-order differences", "#358075", "s"),
                                      ("fourier", "Fourier differentiation", "#a44d2d", "^")):
        ax.loglog([row["n"] for row in rows], [row[key] for row in rows],
                  marker=marker, color=colour, label=label, linewidth=1.6)
    ax.set(xlabel="Grid points in the periodic direction, N",
           ylabel="Maximum grid-point derivative error",
           title="A smooth periodic function: u(x) = exp(sin x)")
    ax.set_xticks([int(row["n"]) for row in rows], [str(int(row["n"])) for row in rows])
    ax.grid(which="major", alpha=.18)
    ax.legend(frameon=False, loc="lower left")
    target = args.output / "differentiation/periodic/figures/convergence.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    for path, label, colour, style in ((args.fd, "Finite differences + Euler", "#526a82", "-"),
                                       (args.fourier, "Fourier + Euler", "#a44d2d", "--"),
                                       (args.exponential, "Fourier + exponential", "#358075", "-.")):
        rows = read(path)[1:]  # Initial zero error has no logarithm.
        ax.semilogy([row["time"] for row in rows], [row["rms_error"] for row in rows],
                    color=colour, linestyle=style, linewidth=1.8, label=label)
    ax.set(xlabel="Time, t", ylabel="RMS error against the continuous solution",
           title="One heat equation · N = 16 · D = 1 · Δt = 0.001")
    ax.grid(which="major", alpha=.18)
    ax.legend(frameon=False, loc="center right")
    target = args.output / "heat/figures/methods.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
