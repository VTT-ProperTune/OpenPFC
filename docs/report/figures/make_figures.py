#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate the scalability figures for the applications report.

The report itself has no compute engine: it renders with `quarto` alone and
reads the SVGs this script writes. Run it after editing anything under
`docs/report/data/`.

    python3 docs/report/figures/make_figures.py

Requires matplotlib. On LUMI, `module load cray-python` does not provide it;
use a virtual environment.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend choice)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Brand-neutral, colour-blind-safe, and distinguishable in greyscale print.
COLORS = {"tungsten": "#1f6fb4", "fd": "#c1662f", "spectral": "#3f8a5c"}
GRID = {"color": "#d5d8dc", "linewidth": 0.6}


def read(name):
    with (DATA / name).open() as stream:
        rows = [r for r in csv.DictReader(line for line in stream
                                          if not line.startswith("#"))]
    return rows


def curve(rows):
    p = [int(r["gcds"]) for r in rows]
    t = [float(r["wall_step_ms"]) for r in rows]
    speedup = [t[0] / v for v in t]
    return p, t, speedup


def style(ax):
    ax.grid(True, which="both", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_speedup():
    series = [
        ("tungsten_hip 768³ (PFC, spectral)", "tungsten_hip_768.csv", "tungsten", "o"),
        ("heat3d_spectral_hip 768³", "heat3d_spectral_hip_768.csv", "spectral", "s"),
        ("heat3d_fd_hip 512³", "heat3d_fd_hip_512.csv", "fd", "^"),
    ]
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for label, name, key, marker in series:
        p, _, speedup = curve(read(name))
        ax_s.plot(p, speedup, marker=marker, color=COLORS[key], label=label, lw=1.8, ms=5)
        ax_e.plot(p, [100.0 * s / n for s, n in zip(speedup, p)],
                  marker=marker, color=COLORS[key], label=label, lw=1.8, ms=5)

    ideal = [1, 32]
    ax_s.plot(ideal, ideal, ls="--", color="#8a8f98", lw=1.2, label="ideal", zorder=0)
    ax_s.set_xscale("log", base=2)
    ax_s.set_yscale("log", base=2)
    ax_s.set_xticks([1, 2, 4, 8, 16, 24, 32])
    ax_s.set_xticklabels(["1", "2", "4", "8", "16", "24", "32"])
    ax_s.set_yticks([1, 2, 4, 8, 16, 32])
    ax_s.set_yticklabels(["1", "2", "4", "8", "16", "32"])
    ax_s.set_xlabel("GCDs (1 MPI rank per GCD)")
    ax_s.set_ylabel("speedup vs 1 GCD")
    ax_s.set_title("Strong-scaling speedup", loc="left", fontsize=11)
    style(ax_s)
    ax_s.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax_e.axhline(100.0, ls="--", color="#8a8f98", lw=1.2, zorder=0)
    ax_e.set_xscale("log", base=2)
    ax_e.set_xticks([1, 2, 4, 8, 16, 24, 32])
    ax_e.set_xticklabels(["1", "2", "4", "8", "16", "24", "32"])
    ax_e.set_ylim(0, 115)
    ax_e.set_xlabel("GCDs (1 MPI rank per GCD)")
    ax_e.set_ylabel("parallel efficiency (%)")
    ax_e.set_title("Parallel efficiency", loc="left", fontsize=11)
    style(ax_e)

    # Everything right of this line is multi-node and crosses the interconnect.
    for ax in (ax_s, ax_e):
        ax.axvline(8, color="#6b7076", lw=1.0, ls=(0, (4, 3)), zorder=1, alpha=0.75)
        lo, hi = ax.get_ylim()
        ax.annotate("1 node │ multi-node", xy=(8, hi), xytext=(0, -11),
                    textcoords="offset points", fontsize=7.5, color="#6b7076",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85))

    fig.tight_layout()
    out = HERE / "scaling_strong.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def figure_sizing():
    rows = read("tungsten_hip_sizing.csv")
    nx = [int(r["nx"]) for r in rows]
    t = [float(r["wall_step_ms"]) for r in rows]
    cells = [n ** 3 / 1e6 for n in nx]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(cells, t, marker="o", color=COLORS["tungsten"], lw=1.8, ms=5)
    for c, v, n in zip(cells, t, nx):
        ax.annotate(f"{n}³", xy=(c, v), xytext=(5, -3), textcoords="offset points",
                    fontsize=8, color="#4a4f55", ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Explicit ticks at the measured sizes; the default log minor labels
    # collide at this aspect ratio.
    ax.set_xticks(cells, minor=False)
    ax.set_xticklabels([f"{c:.0f}" for c in cells])
    ax.set_xticks([], minor=True)
    ax.set_yticks([20, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["20", "50", "100", "200", "500", "1000"])
    ax.set_yticks([], minor=True)
    ax.set_xlim(min(cells) * 0.75, max(cells) * 1.45)
    ax.set_xlabel("grid cells (millions)")
    ax.set_ylabel("median wall_step (ms)")
    ax.set_title("Single-GCD sizing, tungsten_hip", loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    out = HERE / "scaling_sizing.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (figure_speedup(), figure_sizing()):
        print("wrote", path.relative_to(HERE.parent.parent.parent))
