#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw native wave, KdV and Allen–Cahn CSVs with Matplotlib 3.11.2."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def array(data, key):
    return np.array([float(row[key]) for row in data])


def frames(data):
    steps = sorted({int(row["step"]) for row in data})
    return [[row for row in data if int(row["step"]) == step] for step in steps]


def chebyshev_display(nodes, points):
    """Evaluate the saved nodal interpolant; this does not evolve a PDE."""
    weights = (-1.0) ** np.arange(len(nodes))
    weights[[0, -1]] *= 0.5
    difference = points[:, None] - nodes[None, :]
    matches = np.abs(difference) < 1e-14
    ordinary = ~np.any(matches, axis=1)
    matrix = np.zeros_like(difference)
    terms = weights / difference[ordinary]
    matrix[ordinary] = terms / np.sum(terms, axis=1)[:, None]
    for row in np.flatnonzero(~ordinary):
        matrix[row, np.flatnonzero(matches[row])[0]] = 1
    return matrix


def fourier_display(nodes, points, half_length):
    n = len(nodes)
    angle = np.pi * (points[:, None] - nodes[None, :]) / half_length
    kernel = 1 + np.cos(n / 2 * angle)
    for k in range(1, n // 2):
        kernel += 2 * np.cos(k * angle)
    return kernel / n


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
            "svg.hashsalt": "openpfc-evolution",
        }
    )
    blue, red, green = "#526a82", "#a44d2d", "#358075"
    colours = [blue, red, green, "#876b95", "#aa8550"]

    def read(case, name):
        return rows(args.runs / case / "results" / name)

    def save(fig, name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, metadata={"Date": None})
        plt.close(fig)

    saved = frames(read("line", "wave.csv"))
    points = np.linspace(-1, 1, 401)
    interpolate = chebyshev_display(array(saved[0], "x"), points)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), layout="constrained")
    for frame in saved[::2]:
        t = float(frame[0]["time"])
        axes[0].plot(
            points, t + 0.25 * (interpolate @ array(frame, "u")), color=blue, lw=0.7
        )
    for t, colour in zip((0, 1, 2, 3, 3.975), colours):
        frame = min(saved, key=lambda f: abs(float(f[0]["time"]) - t))
        axes[1].plot(
            points,
            interpolate @ array(frame, "u"),
            color=colour,
            label=f't = {float(frame[0]["time"]):.3f}',
        )
    axes[0].set(
        title="Pulse and reflections · degree 80", xlabel="x", ylabel="Time + 0.25 u"
    )
    axes[1].set(title="Selected saved profiles", xlabel="x", ylabel="u(x,t)")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, "waves/chebyshev-line/figures/wave.svg")

    saved = frames(read("square", "wave.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), layout="constrained")
    points = np.linspace(-1, 1, 201)
    interpolate = chebyshev_display(array(saved[0][:25], "x"), points)
    fields = [
        interpolate @ array(frame, "u").reshape(25, 25) @ interpolate.T
        for frame in saved
    ]
    maximum = max(np.max(np.abs(field)) for field in fields)
    for ax, frame, field in zip(axes.flat, saved, fields):
        image = ax.pcolormesh(
            points,
            points,
            field,
            shading="nearest",
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
            rasterized=True,
        )
        ax.set(
            title=f't = {float(frame[0]["time"]):.3f}',
            xlabel="x",
            ylabel="y",
            aspect="equal",
        )
    fig.colorbar(image, ax=list(axes.flat), label="u(x,y,t)", shrink=0.8)
    save(fig, "waves/chebyshev-square/figures/wave.svg")

    saved = frames(read("tank", "wave.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.8), layout="constrained")
    points_x = np.linspace(-3, 3, 301)
    points_y = np.linspace(-1, 1, 101)
    interpolate_x = fourier_display(array(saved[0][:50], "x"), points_x, 3)
    interpolate_y = chebyshev_display(array(saved[0][::50], "y"), points_y)
    fields = [
        interpolate_y @ array(frame, "u").reshape(16, 50) @ interpolate_x.T
        for frame in saved
    ]
    maximum = max(np.max(np.abs(field)) for field in fields)
    for ax, frame, field in zip(axes, saved, fields):
        image = ax.pcolormesh(
            points_x,
            points_y,
            field,
            shading="nearest",
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
            rasterized=True,
        )
        ax.set(
            title=f't = {float(frame[0]["time"]):.0f}',
            xlabel="x · periodic",
            ylabel="y · reflecting",
        )
    fig.colorbar(image, ax=list(axes), label="u(x,y,t)", shrink=0.8)
    save(fig, "waves/mixed-tank/figures/wave.svg")

    saved = frames(read("kdv", "kdv.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), layout="constrained")
    t = np.array([float(frame[0]["time"]) for frame in saved])
    x = array(saved[0], "x")
    u = np.array([array(frame, "u") for frame in saved])
    image = axes[0].pcolormesh(
        x, t, u, shading="nearest", cmap="viridis", rasterized=True
    )
    fig.colorbar(image, ax=axes[0], label="u(x,t)", shrink=0.8)
    axes[0].set(title="Two periodic pulses · N = 256", xlabel="x", ylabel="Time")
    diagnostic = read("kdv", "diagnostics.csv")
    for key, colour, label in (
        ("mean", blue, "Mean"),
        ("mean_square", red, "Mean square"),
    ):
        value = array(diagnostic, key)
        axes[1].semilogy(
            array(diagnostic, "time"),
            np.maximum(np.abs((value - value[0]) / value[0]), 1e-16),
            "o-",
            color=colour,
            ms=3,
            label=label,
        )
    axes[1].set(
        title="Changes in integral diagnostics",
        xlabel="Time",
        ylabel="Absolute relative change · floor 10⁻¹⁶",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.15)
    save(fig, "waves/kdv/figures/kdv.svg")

    for case, slug in (("fixed", "allen-cahn-fixed"), ("driven", "allen-cahn-driven")):
        data = read(case, "allen_cahn.csv")
        curves = frames([row for row in data if row["kind"] == "curve"])
        nodes = frames([row for row in data if row["kind"] == "node"])
        fig = plt.figure(figsize=(9.2, 6), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        field = fig.add_subplot(grid[:, 0])
        profiles = fig.add_subplot(grid[0, 1])
        diagnostic = fig.add_subplot(grid[1, 1])
        t = np.array([float(frame[0]["time"]) for frame in curves])
        x = array(curves[0], "x")
        u = np.array([array(frame, "u") for frame in curves])
        image = field.pcolormesh(
            x,
            t,
            u,
            shading="nearest",
            cmap="RdBu_r",
            vmin=-2 if case == "driven" else -1.01,
            vmax=2 if case == "driven" else 1.01,
            rasterized=True,
        )
        fig.colorbar(image, ax=field, label="u(x,t)", shrink=0.6)
        field.set(
            title="Driven boundary" if case == "driven" else "Fixed boundaries",
            xlabel="x",
            ylabel="Time",
        )
        for position, colour in zip((0, 5, 10, 25, 50), colours):
            frame = curves[position]
            profiles.plot(
                array(frame, "x"),
                array(frame, "u"),
                color=colour,
                label=f't = {float(frame[0]["time"]):.0f}',
            )
        profiles.set(title="Interpolated saved profiles", xlabel="x", ylabel="u")
        profiles.legend(frameon=False, fontsize=8, ncol=2)
        if case == "fixed":
            defect = [
                np.max(np.abs(array(frame, "u") + array(frame, "u")[::-1]))
                for frame in nodes
            ]
            diagnostic.semilogy(t, defect, color=green)
            diagnostic.set(
                title="Roundoff grows in a sensitive direction",
                xlabel="Time",
                ylabel="Nodal oddness defect",
            )
        else:
            diagnostic.plot(t, [float(frame[0]["u"]) for frame in nodes], color=red)
            diagnostic.set(
                title="Prescribed value at x = 1", xlabel="Time", ylabel="u(1,t)"
            )
        for ax in (profiles, diagnostic):
            ax.grid(alpha=0.15)
        save(fig, f"polynomials/{slug}/figures/allen-cahn.svg")


if __name__ == "__main__":
    main()
