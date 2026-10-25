#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot actual owned Cartesian fields with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data = []
    for path in sorted((args.run / "results").glob("polar_rank_*.csv")):
        with path.open() as file:
            data.extend(csv.DictReader(file))
    assert data
    n = max(int(row["i"]) for row in data) + 1
    assert len(data) == n * n and len({(row["i"], row["j"]) for row in data}) == n * n
    x = sorted({float(row["x"]) for row in data})
    y = sorted({float(row["y"]) for row in data})
    u = np.zeros((n, n))
    for row in data:
        u[int(row["j"]), int(row["i"])] = float(row["u"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openpfc-coordinate-pattern",
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.6), layout="constrained")
    bound = max(float(np.max(abs(u))), 1e-16)
    image = ax.contourf(x, y, u, levels=np.linspace(-bound, bound, 33), cmap="RdBu_r")
    if np.min(u) < 0 < np.max(u):
        ax.contour(x, y, u, levels=[0], colors="#292b2c", linewidths=0.7)
    ax.set(
        title=f"Polar-defined field · N = {n}", xlabel="x", ylabel="y", aspect="equal"
    )
    fig.colorbar(image, ax=ax, label="u", shrink=0.85)
    target = args.output / "geometry/polar-patterns/figures/pattern.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
