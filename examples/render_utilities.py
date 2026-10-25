#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot actual utility lesson CSVs with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "openpfc-utilities",
    }
)


def save_svg(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Date": None})
    destination.write_text(
        "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
    )
    plt.close(figure)


def wave_numbers(run, output):
    with (run / "results/axes.csv").open() as file:
        rows = list(csv.DictReader(file))
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for axis, plot, color in zip(range(3), axes, ("#286b83", "#ab5137", "#677a35")):
        data = [row for row in rows if int(row["axis"]) == axis]
        index = [int(row["index"]) for row in data]
        wave = [float(row["k"]) for row in data]
        plot.plot(index, wave, "o-", color=color, label="Wave number")
        special = [row for row in data if float(row["k"]) != float(row["k_odd"])]
        if special:
            plot.scatter(
                [int(row["index"]) for row in special],
                [float(row["k_odd"]) for row in special],
                marker="x",
                s=65,
                color="#202020",
                label="First derivative",
            )
        plot.axhline(0, color="0.7", linewidth=0.7)
        plot.set(
            title=f"{'xyz'[axis]} axis · N = {len(data)}",
            xlabel="Stored DFT index",
            ylabel="Wave number [rad / length]",
        )
        plot.set_xticks(index)
        plot.grid(alpha=0.15)
        plot.legend(fontsize=8)
    destination = output / "fourier/symbols/figures/waves.svg"
    save_svg(figure, destination)


def gaussian(run, output):
    import json

    rows = []
    for path in sorted((run / "results").glob("field_rank_*.csv")):
        with path.open() as file:
            rows.extend(csv.DictReader(file))
    assert rows
    y = min({float(row["y"]) for row in rows}, key=abs)
    z = min({float(row["z"]) for row in rows}, key=abs)
    line = sorted(
        (row for row in rows if float(row["y"]) == y and float(row["z"]) == z),
        key=lambda row: float(row["x"]),
    )
    x = [float(row["x"]) for row in line]
    initial = [float(row["initial"]) for row in line]
    values = [float(row["u"]) for row in line]
    exact = [float(row["exact"]) for row in line]
    summary = json.loads((run / "results/summary.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot(x, initial, color="0.6", label="Initial samples")
    axes[0].plot(x, exact, "--", color="#202020", label="Periodic continuum")
    axes[0].plot(x, values, "o-", markersize=3, color="#286b83", label="Backward Euler")
    axes[0].set(title=f"Pulse at t = {summary['time']:g}", ylabel="u")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        x,
        [value - reference for value, reference in zip(values, exact)],
        "o-",
        markersize=3,
        color="#ab5137",
    )
    axes[1].axhline(0, color="0.6", linewidth=0.7)
    axes[1].set(title="Error on the same grid line", ylabel="Numerical − continuum")
    for plot in axes:
        plot.set_xlabel(f"x   (y = {y:g}, z = {z:g})")
        plot.grid(alpha=0.15)
    destination = output / "heat/gaussian-pulse/figures/pulse.svg"
    save_svg(figure, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=Path)
    parser.add_argument("--gaussian", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.symbols and not args.gaussian:
        parser.error("Provide --symbols or --gaussian")
    if args.symbols:
        wave_numbers(args.symbols, args.output)
    if args.gaussian:
        gaussian(args.gaussian, args.output)


if __name__ == "__main__":
    main()
