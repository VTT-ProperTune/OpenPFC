#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native initialization, implicit heat and phase-field CSVs; Matplotlib 3.11.2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def values(data, key):
    return np.array([float(row[key]) for row in data])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openpfc-everyday",
        }
    )
    blue, red, green = "#526a82", "#a44d2d", "#358075"

    def read(case, name):
        return rows(args.runs / case / "results" / name)

    def save(fig, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, metadata={"Date": None})
        plt.close(fig)

    config = json.loads((source / "initialization/patterns/input.json").read_text())
    n = config["n"]
    data = [
        row
        for path in sorted((args.runs / "patterns/results").glob("fields_rank_*.csv"))
        for row in rows(path)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), layout="constrained")
    for ax, name, title, cmap in zip(
        axes,
        ("gaussian", "vortex_speed", "checkerboard"),
        (
            "Gaussian · scalar amplitude",
            "Vortex · tangential speed",
            "Unit-cell checkerboard",
        ),
        ("viridis", "magma", "RdBu_r"),
    ):
        subset = sorted(
            (row for row in data if row["pattern"] == name and float(row["z"]) == 0),
            key=lambda r: (float(r["y"]), float(r["x"])),
        )
        x = values(subset[:n], "x")
        y = values(subset[::n], "y")
        image = ax.pcolormesh(
            x,
            y,
            values(subset, "u").reshape(n, n),
            shading="nearest",
            cmap=cmap,
            rasterized=True,
        )
        fig.colorbar(image, ax=ax, shrink=0.72)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    save(fig, "initialization/patterns/figures/patterns.svg")

    data = read("implicit", "diagnostics.csv")
    config = json.loads((source / "heat/implicit-euler/input.json").read_text())
    t = values(data, "time")
    amplitude = np.sqrt(8) * values(data, "rms_fluctuation")
    exact = 0.2 * np.exp(-3 * config["diffusivity"] * t)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    axes[0].plot(t, exact, color=red, label="Exact linear evolution")
    axes[0].plot(t, amplitude, "o-", ms=3, color=blue, label="Backward Euler")
    axes[0].set(
        title="Same mode, different decay",
        xlabel="Time",
        ylabel="Cosine-product amplitude",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(t, np.abs(amplitude - exact), "o-", ms=3, color=green)
    axes[1].set(
        title=f'Stable step · Δt = {config["dt"]}',
        xlabel="Time",
        ylabel="Absolute amplitude error",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, "heat/implicit-euler/figures/implicit.svg")

    config = json.loads((source / "phase-fields/cahn-hilliard/input.json").read_text())
    n = config["n"]
    data = [
        row
        for path in sorted((args.runs / "phase/results").glob("fields_rank_*.csv"))
        for row in rows(path)
    ]
    selected = [0, config["steps"] // 4, config["steps"] // 2, config["steps"]]
    fig = plt.figure(figsize=(9.2, 8.3), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])
    axes = [fig.add_subplot(grid[j, i]) for j in range(2) for i in range(2)]
    scale = max(1.0, np.max(np.abs(values(data, "u"))))
    for ax, step in zip(axes, selected):
        subset = sorted(
            (row for row in data if int(row["step"]) == step),
            key=lambda r: (float(r["y"]), float(r["x"])),
        )
        image = ax.pcolormesh(
            values(subset[:n], "x"),
            values(subset[::n], "y"),
            values(subset, "u").reshape(n, n),
            shading="nearest",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            rasterized=True,
        )
        ax.set(
            title=f't = {float(subset[0]["time"]):.2f}',
            xlabel="x",
            ylabel="y",
            aspect="equal",
        )
    fig.colorbar(image, ax=axes, label="Composition c", shrink=0.8)
    data = read("phase", "diagnostics.csv")
    energy = fig.add_subplot(grid[2, 0])
    mass = fig.add_subplot(grid[2, 1])
    energy.plot(values(data, "time"), values(data, "energy"), "o-", ms=3, color=green)
    energy.set(title="Discrete free-energy density", xlabel="Time", ylabel="Fh / area")
    mean = values(data, "mean")
    mass.plot(values(data, "time"), mean - mean[0], "o-", ms=3, color=blue)
    mass.set(
        title="Change in conserved mean", xlabel="Time", ylabel="Mean(c) − initial mean"
    )
    mass.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    for ax in (energy, mass):
        ax.grid(alpha=0.15)
    save(fig, "phase-fields/cahn-hilliard/figures/phase-field.svg")


if __name__ == "__main__":
    main()
