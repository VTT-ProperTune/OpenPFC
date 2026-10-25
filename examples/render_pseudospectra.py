#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native program-24 singular values with Matplotlib 3.11.2."""

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
    with (args.run / "results/pseudospectrum.csv").open() as file:
        samples = list(csv.DictReader(file))
    with (args.run / "results/spectrum.csv").open() as file:
        eigenvalues = list(csv.DictReader(file))
    x = sorted({float(row["real"]) for row in samples})
    y = sorted({float(row["imag"]) for row in samples})
    sigma = np.array([float(row["sigma_min"]) for row in samples]).reshape(
        len(y), len(x)
    )
    assert len(samples) == len(x) * len(y)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openpfc-pseudospectra",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6.3), layout="constrained")
    contours = ax.contour(
        x,
        y,
        sigma,
        levels=10.0 ** np.arange(-4, 0, 0.5),
        colors="#526a82",
        linewidths=1,
    )
    ax.clabel(contours, fmt=lambda value: rf"$10^{{{np.log10(value):g}}}$", fontsize=8)
    ax.scatter(
        [float(row["real"]) for row in eigenvalues],
        [float(row["imag"]) for row in eigenvalues],
        color="#a44d2d",
        s=22,
        label="Finite-matrix eigenvalues",
        zorder=3,
    )
    ax.set(
        title=f"Davies oscillator · N = {len(eigenvalues) + 1}",
        xlabel="Re z",
        ylabel="Im z",
        xlim=(0, 50),
        ylim=(0, 40),
        aspect="equal",
    )
    ax.grid(alpha=0.12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    target = args.output / "polynomials/pseudospectra/figures/pseudospectrum.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
