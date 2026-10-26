#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native stencil responses with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from render_utilities import save_svg


def central_line(run):
    with (run / "results/responses.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows
    y = min({float(row["y"]) for row in rows}, key=abs)
    z = min({float(row["z"]) for row in rows}, key=abs)
    return sorted(
        (row for row in rows if float(row["y"]) == y and float(row["z"]) == z),
        key=lambda row: float(row["x"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=Path, required=True)
    parser.add_argument("--polynomial", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    edge, polynomial = central_line(args.edge), central_line(args.polynomial)
    figure, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), layout="constrained")
    x = [float(row["x"]) for row in edge]
    axes[0].plot(x, [float(row["u"]) for row in edge], "o-", markersize=4, color="0.4")
    axes[0].set(title="Sample an edge with transverse curvature", ylabel="u")
    for key, color, label, marker in (
        ("central", "#286b83", "Central difference", "o"),
        ("separable", "#ab5137", "Normalized Sobel", "x"),
    ):
        axes[1].plot(
            x,
            [float(row[key]) for row in edge],
            marker=marker,
            markersize=5,
            color=color,
            label=label,
        )
        h = float(polynomial[1]["x"]) - float(polynomial[0]["x"])
        error = [
            (
                float(row[key])
                - (
                    3 * float(row["x"]) ** 2
                    + float(row["y"]) ** 2
                    + 0.5 * float(row["z"]) ** 2
                )
            )
            / h**2
            for row in polynomial
        ]
        axes[2].plot(
            [float(row["x"]) for row in polynomial],
            error,
            marker=marker,
            markersize=5,
            color=color,
            label=label,
        )
    axes[1].set(
        title="The normalized responses detect the same edge",
        ylabel="Response [1 / length]",
    )
    axes[2].set(
        title="A polynomial reveals the error constants",
        ylabel="Derivative error / h²",
        ylim=(0.8, 1.95),
    )
    for plot in axes:
        plot.set_xlabel("x on the central grid line")
        plot.grid(alpha=0.15)
    axes[1].legend(fontsize=9)
    axes[2].legend(fontsize=9)
    save_svg(figure, args.output / "differentiation/stencils/figures/stencils.svg")


if __name__ == "__main__":
    main()
