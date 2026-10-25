#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw the foundation/book figures from the C++ programs' CSV output.

Use Matplotlib 3.11.2. This script never differentiates or solves a PDE.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def save(fig, root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "openpfc-foundations"})
    red, blue, green = "#a44d2d", "#526a82", "#358075"

    rows = [row for path in (args.runs / "collocation/results").glob("derivative_rank_*.csv")
            for row in table(path)]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), layout="constrained")
    for row_index, name in enumerate(("hat", "analytic")):
        values = sorted((row for row in rows if row["profile"] == name), key=lambda row: float(row["x"]))
        x = [float(row["x"]) for row in values]
        axes[row_index, 0].plot(x, [float(row["value"]) for row in values], "o-", color=blue, markersize=3)
        axes[row_index, 0].set(title="Hat samples" if name == "hat" else "exp(sin x) samples", ylabel="u(x)")
        ax = axes[row_index, 1]
        ax.plot(x, [float(row["matrix"]) for row in values], "-", color=red, label="Collocation matrix")
        ax.plot(x, [float(row["fourier"]) for row in values], "o", color=blue, markersize=3, label="OpenPFC FFT")
        if name == "analytic":
            ax.plot(x, [float(row["exact"]) for row in values], "--", color=green, label="Analytic derivative")
        ax.set(title="Same interpolating derivative", ylabel="Derivative")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.grid(alpha=.15)
    save(fig, args.output, "differentiation/collocation/figures/collocation.svg")

    rows = table(args.runs / "smoothness/results/smoothness.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), layout="constrained")
    names = ("finite-smoothness", "smooth-bump", "analytic-strip", "band-limited")
    titles = ("Finite smoothness: |sin x|³", "Smooth nonanalytic bump", "Analytic in a strip", "One Fourier mode: sin(10x)")
    for ax, name, title in zip(axes.flat, names, titles):
        values = [row for row in rows if row["profile"] == name]
        n = [int(row["n"]) for row in values]
        ax.semilogy(n, [float(row["matrix_error"]) for row in values], "-", color=red, label="Matrix")
        ax.semilogy(n, [float(row["fourier_error"]) for row in values], "o", color=blue, markersize=3, label="FFT")
        ax.set(title=title, xlabel="Grid points, N", ylabel="Maximum derivative error")
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=8)
    save(fig, args.output, "differentiation/smoothness/figures/smoothness.svg")

    rows = table(args.runs / "cardinal/results/interpolation.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), layout="constrained", sharex=True)
    for kind, (ax, title) in enumerate(zip(axes, ("A discrete spike", "A square pulse", "A hat"))):
        values = [row for row in rows if int(row["profile"]) == kind]
        x = [float(row["x"]) for row in values]
        ax.plot(x, [float(row["value"]) for row in values], color=blue, label="Cardinal reconstruction")
        samples = [row for row in values if float(row["x"]).is_integer()]
        ax.plot([float(row["x"]) for row in samples], [float(row["value"]) for row in samples],
                "o", color=red, markersize=4, label="Nodal values")
        ax.set(title=title, ylabel="Value", ylim=(-.45, 1.45))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("x (sample spacing h = 1)")
    save(fig, args.output, "interpolation/cardinal/figures/interpolation.svg")

    rows = [row for path in (args.runs / "transform/results").glob("spectrum_rank_*.csv")
            for row in table(path) if row["j"] == row["k"] == "0"]
    rows.sort(key=lambda row: int(row["i"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
    for ax, component in zip(axes, ("real", "imag")):
        x = [int(row["i"]) for row in rows]
        ax.bar(x, [float(row["reference_" + component]) for row in rows], color=blue, alpha=.45, label="Global direct DFT")
        ax.plot(x, [float(row[component]) for row in rows], "o", color=red, label="OpenPFC FFT")
        ax.set(xlabel="Stored x-mode index", ylabel="Unnormalized coefficient", title=component.capitalize() + " part")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False, fontsize=8)
    save(fig, args.output, "fourier/round-trip/figures/coefficients.svg")

    transport = args.runs / "transport/results"
    if transport.is_dir():
        rows = [row for path in transport.glob("transport_rank_*.csv") for row in table(path)]
        rows.sort(key=lambda row: (int(row["step"]), float(row["x"])))
        x = sorted({float(row["x"]) for row in rows})
        t = sorted({float(row["time"]) for row in rows})
        values = np.array([float(row["u"]) for row in rows]).reshape(len(t), len(x))
        exact = np.array([float(row["exact"]) for row in rows]).reshape(len(t), len(x))
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), layout="constrained")
        for ax, data, title, cmap in ((axes[0], values, "Computed transported packet", "viridis"),
                                       (axes[1], np.abs(values - exact), "Absolute characteristic-reference error", "magma")):
            image = ax.pcolormesh(x, t, data, shading="auto", cmap=cmap, rasterized=True)
            fig.colorbar(image, ax=ax, shrink=.85)
            ax.set(title=title, xlabel="x", ylabel="Time, t")
        save(fig, args.output, "waves/variable-transport/figures/transport.svg")

    stability = args.runs / "stability/results"
    if stability.is_dir():
        rows = table(stability / "multistep.csv")
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), layout="constrained")
        colours = (blue, red, green, "#8a7198", "#9a853c", "#54645d")
        styles = ("-", "--", "-.", ":", "-", "--")
        for ax, family, title, limits in zip(axes.flat, ("AB", "AM", "BDF"),
                                             ("Adams–Bashforth", "Adams–Moulton", "Backward differentiation"),
                                             ((-2.5, .5, -1.5, 1.5), (-7, 1, -4, 4), (-15, 35, -25, 25))):
            orders = sorted({int(row["order"]) for row in rows if row["family"] == family})
            for index, order in enumerate(orders):
                points = [row for row in rows if row["family"] == family and int(row["order"]) == order]
                ax.plot([float(row["real"]) for row in points], [float(row["imag"]) for row in points],
                        color=colours[index], linestyle=styles[index], linewidth=1.3, label=str(order))
            ax.set(title=title, xlim=limits[:2], ylim=limits[2:])
            ax.legend(title="Order", frameon=False, fontsize=8, ncol=2)
        grid = table(stability / "rk.csv")
        x = sorted({float(row["real"]) for row in grid})
        y = sorted({float(row["imag"]) for row in grid})
        for order in range(1, 5):
            values = np.array([float(row[f"rk{order}"]) for row in grid]).reshape(len(y), len(x))
            axes[1, 1].contour(x, y, values, levels=[1], colors=[colours[order - 1]], linestyles=[styles[order - 1]])
        axes[1, 1].set(title="Explicit RK: |R(z)| = 1")
        axes[1, 1].legend([Line2D([], [], color=colours[i], linestyle=styles[i]) for i in range(4)],
                           ["RK1", "RK2", "RK3", "RK4"], frameon=False, fontsize=8)
        for ax in axes.flat:
            ax.axhline(0, color="black", alpha=.25, linewidth=.7)
            ax.axvline(0, color="black", alpha=.25, linewidth=.7)
            ax.set(xlabel="Re(z)", ylabel="Im(z)", aspect="equal")
            ax.grid(alpha=.12)
        save(fig, args.output, "integration/stability/figures/stability.svg")


if __name__ == "__main__":
    main()
