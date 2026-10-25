#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native disk fields with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def polar_field(rows, key):
    nr = max(int(row["ir"]) for row in rows) + 1
    nt = max(int(row["it"]) for row in rows) + 1
    arrays = [
        np.array([float(row[name]) for row in rows]).reshape(nr, nt)
        for name in ("x", "y", key)
    ]
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openpfc-disk",
        }
    )
    circle = np.linspace(0, 2 * np.pi, 201)

    def frame(ax):
        ax.plot(np.cos(circle), np.sin(circle), color="#292b2c", lw=0.8)
        ax.set(
            xlabel="x",
            ylabel="y",
            xlim=(-1.03, 1.03),
            ylim=(-1.03, 1.03),
            aspect="equal",
            xticks=(-1, 0, 1),
            yticks=(-1, 0, 1),
        )

    def save(fig, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, metadata={"Date": None})
        plt.close(fig)

    data = [
        row
        for row in read(args.runs / "modes/results/fields.csv")
        if row["kind"] == "curve"
    ]
    selected = read(args.runs / "modes/results/selected.csv")
    limit = np.ceil(max(abs(float(row["u"])) for row in data) * 100) / 100
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), layout="constrained")
    for ax, summary in zip(axes.flat, selected):
        samples = [row for row in data if row["mode"] == summary["mode"]]
        x, y, u = polar_field(samples, "u")
        image = ax.contourf(
            x, y, u, levels=np.linspace(-limit, limit, 31), cmap="RdBu_r"
        )
        ax.contour(x, y, u, levels=[0], colors="#292b2c", linewidths=0.9)
        frame(ax)
        ax.set_title(
            f'Mode {summary["mode"]} · ω / ω₁ = {float(summary["frequency_ratio"]):.5f}'
        )
    fig.colorbar(
        image,
        ax=list(axes.flat),
        shrink=0.8,
        label="u · normalized on the collocation grid",
        ticks=(-1, -0.5, 0, 0.5, 1),
    )
    save(fig, "polynomials/disk-modes/figures/modes.svg")

    data = [
        row
        for row in read(args.runs / "poisson/results/solution.csv")
        if row["kind"] == "curve"
    ]
    diagnostics = read(args.runs / "poisson/results/diagnostics.csv")[0]
    x, y, u = polar_field(data, "u")
    _, _, exact = polar_field(data, "exact")
    error = u - exact
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.3), layout="constrained")
    image = axes[0].contourf(x, y, u, levels=25, cmap="viridis")
    fig.colorbar(image, ax=axes[0], label="u", shrink=0.8)
    axes[0].set_title(
        f'Computed field · N = {diagnostics["n"]}, M = {diagnostics["m"]}'
    )
    bound = max(float(np.max(abs(error))), 1e-16)
    image = axes[1].contourf(
        x, y, error, levels=np.linspace(-bound, bound, 25), cmap="RdBu_r"
    )
    fig.colorbar(image, ax=axes[1], label="Interpolated u − exact", shrink=0.8)
    axes[1].set_title("Difference from the exact solution")
    for ax in axes:
        frame(ax)
    save(fig, "polynomials/poisson-disk/figures/poisson.svg")


if __name__ == "__main__":
    main()
