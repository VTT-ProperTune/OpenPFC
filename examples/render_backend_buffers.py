#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw host/device round trips from native output with Matplotlib 3.11.2."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from render_utilities import save_svg


def load(run):
    folder = run / "results"
    summary = json.loads((folder / "summary.json").read_text())
    values = {}
    for path in sorted(folder.glob("field_rank_*.csv")):
        with path.open() as file:
            for row in csv.DictReader(file):
                index = tuple(int(row[k]) for k in ("i", "j", "k"))
                assert index not in values
                values[index] = row
    assert len(values) == int(np.prod(summary["shape"]))
    return summary, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    host, cpu = load(args.cpu)
    device, gpu = load(args.device)
    assert host["backend"] == "cpu" and device["backend"] in ("cuda", "hip")
    assert host["shape"] == device["shape"] and host["length"] == device["length"]
    assert cpu.keys() == gpu.keys()
    nx, ny, nz = host["shape"]
    y = [float(cpu[0, j, 0]["y"]) for j in range(ny)]
    z = [float(cpu[0, 0, k]["z"]) for k in range(nz)]
    initial = np.array(
        [[float(cpu[0, j, k]["initial"]) for j in range(ny)] for k in range(nz)]
    )
    recovered = np.array(
        [[float(gpu[0, j, k]["u"]) for j in range(ny)] for k in range(nz)]
    )
    figure, axes = plt.subplots(3, 1, figsize=(6.5, 8), layout="constrained")
    for ax, data, title in (
        (axes[0], initial, "Prescribed nodal field at x = 0"),
        (
            axes[1],
            recovered,
            f"{device['backend'].upper()} after {device['cycles']} round trips",
        ),
    ):
        image = ax.pcolormesh(
            y,
            z,
            data,
            shading="nearest",
            cmap="cividis",
            vmin=initial.min(),
            vmax=initial.max(),
            rasterized=False,
        )
        ax.set(
            xlabel="y", ylabel="z", title=title, xlim=(y[0], y[-1]), ylim=(z[0], z[-1])
        )
        ax.set_aspect("equal", adjustable="box")
        figure.colorbar(image, ax=ax, label="u", pad=0.02)
    # Compare an actual grid line; neither curve is a Python FFT computation.
    j, k = ny // 8, 0
    x = [float(cpu[i, j, k]["x"]) for i in range(nx)]
    for data, color, marker, name in (
        (cpu, "#286b83", "o", "CPU"),
        (gpu, "#ab5137", "x", device["backend"].upper()),
    ):
        error = [
            float(data[i, j, k]["u"]) - float(cpu[i, j, k]["initial"])
            for i in range(nx)
        ]
        axes[2].plot(x, error, color=color, marker=marker, markersize=4, label=name)
    axes[2].set(
        xlabel=f"x at j = {j}, k = {k}",
        ylabel="Recovered − prescribed",
        title="Round-trip differences are floating-point errors",
    )
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[2].grid(alpha=0.15)
    axes[2].legend(fontsize=9)
    save_svg(figure, args.output / "fourier/backend-buffers/figures/buffers.svg")


if __name__ == "__main__":
    main()
