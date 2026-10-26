#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw coupled-rate lessons from checked native CSVs with Matplotlib 3.11.2."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from render_utilities import save_svg


def rows(run):
    with (run / "results/rates.csv").open() as file:
        return list(csv.DictReader(file))


def line(data):
    return sorted(
        (row for row in data if int(row["j"]) == 0 and int(row["k"]) == 0),
        key=lambda row: int(row["i"]),
    )


def coupled(runs, output):
    data = line(rows(runs / "fd-24"))
    figure, axes = plt.subplots(2, 1, figsize=(6.5, 6.5), layout="constrained")
    x = [float(row["x"]) for row in data]
    for key, color, label, marker in (
        ("du", "#286b83", "u rate", "o"),
        ("dv", "#ab5137", "v rate", "x"),
    ):
        axes[0].plot(
            x,
            [float(row["exact_" + key]) for row in data],
            color=color,
            linewidth=1.2,
            alpha=0.55,
        )
        axes[0].plot(
            x,
            [float(row[key]) for row in data],
            marker=marker,
            linestyle="none",
            markersize=4,
            color=color,
            label=label + " (FD6)",
        )
    axes[0].set(
        title="One coupled mode, two computed rates",
        xlabel="x at y = z = 0",
        ylabel="Rate at t = 0.75",
    )
    axes[0].legend(fontsize=9)
    ns, errors = [], []
    for n in (12, 16, 24, 32, 48):
        data = rows(runs / f"fd-{n}")
        ns.append(n)
        errors.append(
            max(
                abs(float(row[key]) - float(row["exact_" + key]))
                for row in data
                for key in ("du", "dv")
            )
        )
    axes[1].loglog(ns, errors, "o-", color="#286b83", label="FD6: complete fields")
    axes[1].loglog(
        ns,
        [errors[-1] * (ns[-1] / n) ** 6 for n in ns],
        "--",
        color=".5",
        label="N⁻⁶, anchored at N = 48",
    )
    spectral = rows(runs / "spectral")
    spectral_error = max(
        abs(float(row[key]) - float(row["exact_" + key]))
        for row in spectral
        for key in ("du", "dv")
    )
    axes[1].plot(
        [24],
        [spectral_error],
        "x",
        color="#ab5137",
        markersize=7,
        label="Spectral at N = 24",
    )
    axes[1].set(
        title="Spatial refinement reduces the rate error",
        xlabel="Points per axis, N",
        ylabel="Maximum RHS error",
    )
    axes[1].set_xticks(ns, labels=[str(n) for n in ns])
    axes[1].minorticks_off()
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
    save_svg(figure, output / "coupling/fixed-tuples/figures/coupled.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupled", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    coupled(args.coupled, args.output)


if __name__ == "__main__":
    main()
