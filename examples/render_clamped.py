#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native clamped-boundary CSVs with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam", type=Path, required=True)
    parser.add_argument(
        "--plate", type=Path, help="Native plate run directory (containing results/)"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with args.beam.open() as file:
        data = list(csv.DictReader(file))
    curve = [row for row in data if row["kind"] == "curve"]
    nodes = [row for row in data if row["kind"] == "node"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openpfc-clamped",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    for ax, key, exact, title in zip(
        axes,
        ("u", "du"),
        ("exact", "exact_du"),
        ("Deflection · both ends fixed", "Slope · both ends clamped"),
    ):
        x = [float(row["x"]) for row in curve]
        ax.plot(
            x,
            [float(row[key]) for row in curve],
            color="#526a82",
            label="Constrained interpolant",
        )
        ax.plot(
            x,
            [float(row[exact]) for row in curve],
            "--",
            color="#a44d2d",
            label="Analytic solution",
        )
        ax.plot(
            [float(row["x"]) for row in nodes],
            [float(row[key]) for row in nodes],
            "o",
            ms=3,
            color="#358075",
            label=f"N = {len(nodes) - 1} nodes",
        )
        ax.set(title=title, xlabel="x", ylabel="u(x)" if key == "u" else "u′(x)")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    target = args.output / "polynomials/clamped-beam/figures/beam.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None})
    plt.close(fig)

    if args.plate:
        with (args.plate / "results/shapes.csv").open() as file:
            shapes = [row for row in csv.DictReader(file) if row["kind"] == "curve"]
        with (args.plate / "results/spectrum.csv").open() as file:
            spectrum = list(csv.DictReader(file))
        fig, axes = plt.subplots(5, 5, figsize=(8.8, 8.5), layout="constrained")
        for ax, row in zip(axes.flat, spectrum):
            samples = [sample for sample in shapes if sample["mode"] == row["mode"]]
            side = int(np.sqrt(len(samples)))
            x = np.array([float(sample["x"]) for sample in samples[:side]])
            u = np.array([float(sample["u"]) for sample in samples]).reshape(side, side)
            ax.imshow(
                u,
                extent=(-1, 1, -1, 1),
                origin="lower",
                interpolation="bilinear",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
            )
            ax.contour(x, x, u, levels=[0], colors="#292b2c", linewidths=0.8)
            ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#292b2c", lw=0.8)
            ax.set(
                title=f'{row["mode"]} · {float(row["frequency_ratio"]):.3f}',
                aspect="equal",
                xlim=(-1.02, 1.02),
                ylim=(-1.02, 1.02),
            )
            ax.set_axis_off()
        for ax in list(axes.flat)[len(spectrum) :]:
            ax.set_axis_off()
        target = args.output / "polynomials/clamped-plate/figures/plate.svg"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
