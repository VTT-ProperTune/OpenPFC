#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native program-40 spectra with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with (args.run / "results/spectrum.csv").open() as file:
        spectrum = list(csv.DictReader(file))
    with (args.run / "results/growth.csv").open() as file:
        summaries = list(csv.DictReader(file))
    assert 1 <= len(summaries) <= 8
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openpfc-orr-sommerfeld",
        }
    )
    columns = 2 if len(summaries) > 1 else 1
    rows = (len(summaries) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.9 * rows),
        squeeze=False,
        layout="constrained",
    )
    for ax, summary in zip(axes.flat, summaries):
        samples = [row for row in spectrum if row["n"] == summary["n"]]
        ax.scatter(
            [float(row["real"]) for row in samples],
            [float(row["imag"]) for row in samples],
            s=14,
            color="#526a82",
            label="Computed eigenvalues",
        )
        ax.scatter(
            float(summary["real"]),
            float(summary["imag"]),
            s=42,
            marker="D",
            color="#a44d2d",
            label="Largest real part",
        )
        ax.axvline(0, color="#6f6a62", lw=1, ls="--", label="Neutral growth")
        ax.set(
            title=f'N = {summary["n"]} · R = {float(summary["reynolds"]):g}\nmax Re λ = {float(summary["real"]):.6e}',
            xlabel="Re λ · growth rate",
            ylabel="Im λ",
            xlim=(-0.8, 0.2),
            ylim=(-1, 0),
            aspect="equal",
        )
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(summaries) :]:
        ax.set_axis_off()
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    target = args.output / "polynomials/orr-sommerfeld/figures/spectrum.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
