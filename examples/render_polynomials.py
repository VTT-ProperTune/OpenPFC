#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot the polynomial lessons from native model CSVs with Matplotlib 3.11.2."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "polynomials")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "openpfc-polynomials"})
    blue, red, green = "#526a82", "#a44d2d", "#358075"

    def read(lesson, name):
        return table(args.runs / lesson / "results" / name)

    def save(fig, lesson, name):
        for ax in fig.axes:
            ax.grid(alpha=.15)
        path = args.output / lesson / "figures" / (name + ".svg")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)

    data = read("interpolation", "interpolation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    for ax, grid in zip(axes, ("equispaced", "chebyshev")):
        line = [r for r in data if r["grid"] == grid and r["kind"] == "curve"]
        nodes = [r for r in data if r["grid"] == grid and r["kind"] == "node"]
        ax.plot(values(line, "x"), values(line, "exact"), color=green, label="Exact function")
        ax.plot(values(line, "x"), values(line, "value"), color=red, label="Interpolant")
        ax.plot(values(nodes, "x"), values(nodes, "value"), "o", color=blue, ms=3, label="Samples")
        ax.set(title=grid.capitalize() + " · degree 16", xlabel="x", ylabel="Value")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "interpolation", "interpolation")

    line, plane = read("nodal-polynomials", "nodal.csv"), read("nodal-polynomials", "potential.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), layout="constrained")
    for row, grid in enumerate(("equispaced", "chebyshev")):
        curve = [r for r in line if r["grid"] == grid]
        axes[row, 0].plot(values(curve, "x"), values(curve, "value"), color=blue)
        axes[row, 0].set(title=grid.capitalize() + " · signed product", xlabel="x", ylabel=r"$\omega_{17}(x)$")
        samples = [r for r in plane if r["grid"] == grid]
        contours = axes[row, 1].contour(values(samples, "real").reshape(113, 141),
                values(samples, "imag").reshape(113, 141), values(samples, "magnitude").reshape(113, 141),
                levels=[1e-4, 1e-3, 1e-2, 1e-1, 1], colors=[blue], linewidths=.9)
        axes[row, 1].clabel(contours, fontsize=7, fmt="%.0e")
        axes[row, 1].set(title=r"Magnitude $|\omega_{17}(z)|$", xlabel="Re z", ylabel="Im z", aspect="equal")
    save(fig, "nodal-polynomials", "nodal")

    data = read("differentiation", "derivatives.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), layout="constrained")
    for row, n in enumerate((10, 20)):
        subset = sorted((r for r in data if int(r["n"]) == n), key=lambda r: float(r["x"]))
        x = values(subset, "x")
        axes[row, 0].plot(x, values(subset, "value"), "o-", ms=3, color=blue)
        axes[row, 0].set(title=f"Samples · degree {n}", ylabel="exp(x) sin(5x)")
        for method, color, symbol in (("matrix", red, "-"), ("fourier", blue, "o")):
            axes[row, 1].plot(x, values(subset, method)-values(subset, "exact"), symbol,
                              ms=3, color=color, label="Matrix" if method == "matrix" else "FFT coefficients")
        axes[row, 1].set(title="Derivative error", ylabel="Computed − exact")
        axes[row, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[row, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("x")
    save(fig, "differentiation", "differentiation")

    titles = [r"$|x|^3$ · finite smoothness", r"$e^{-1/x^2}$ · smooth bump",
              r"$1/(1+x^2)$ · analytic", r"$x^{10}$ · polynomial"]
    for lesson, filename in (("regularity", "regularity.csv"), ("integration", "integrals.csv")):
        data = read(lesson, filename)
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6), layout="constrained", sharex=True, sharey=True)
        for kind, (ax, title) in enumerate(zip(axes.flat, titles)):
            subset = [r for r in data if int(r["profile"]) == kind]
            ax.semilogy(values(subset, "n"), np.maximum(values(subset, "error"), 1e-17),
                        "o-", ms=2, lw=1, color=blue)
            ax.set(title=title, xlabel="Degree N", ylabel="Absolute error", ylim=(1e-17, 100))
        save(fig, lesson, lesson)

    data = read("boundary-values", "solutions.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), layout="constrained")
    for col, boundary in enumerate(("zero", "lifted", "mixed")):
        line = [r for r in data if r["boundary"] == boundary and r["kind"] == "curve"]
        nodes = [r for r in data if r["boundary"] == boundary and r["kind"] == "node"]
        axes[0, col].plot(values(line, "x"), values(line, "exact"), color=green, label="Exact")
        axes[0, col].plot(values(nodes, "x"), values(nodes, "u"), "o", ms=3, color=blue, label="Computed")
        axes[0, col].set(title={"zero": "Zero endpoint values", "lifted": "Lifted value at x = 1", "mixed": "Slope at x = −1"}[boundary], ylabel="u(x)")
        axes[1, col].plot(values(line, "x"), values(line, "u")-values(line, "exact"), color=red)
        axes[1, col].set(xlabel="x", ylabel="Computed − exact")
        axes[1, col].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "boundary-values", "boundaries")

    data = read("nonlinear-boundary", "nonlinear.csv")
    line = [r for r in data if r["kind"] == "curve"]
    nodes = [r for r in data if r["kind"] == "node"]
    history = read("nonlinear-boundary", "newton.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    axes[0].plot(values(line, "x"), values(line, "exact"), color=green, label="Analytic")
    axes[0].plot(values(nodes, "x"), values(nodes, "u"), "o", color=blue, ms=3, label="Newton–GMRES")
    axes[0].set(title="Nonlinear boundary problem", xlabel="x", ylabel="u(x)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].semilogy(values(history, "iteration"), values(history, "residual"), "o-", color=red)
    axes[1].set(title="Accepted Newton iterations", xlabel="Iteration", ylabel="Residual norm")
    save(fig, "nonlinear-boundary", "nonlinear")


if __name__ == "__main__":
    main()
