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


# Okabe-Ito qualitative palette: colour-blind-safe, distinguishable in
# greyscale print (paired with a distinct marker per order below).
ORDER_COLORS = {
    2: "#E69F00",
    4: "#0072B2",
    6: "#009E73",
    8: "#D55E00",
    10: "#CC79A7",
    12: "#000000",
}
ORDER_MARKERS = {2: "o", 4: "s", 6: "^", 8: "D", 10: "v", 12: "P"}


def figure_heat3d_fd_order_convergence():
    """Log-log L2 error vs dx, one line per FD order, with reference slopes.

    Reads `heat3d_fd_order_convergence.csv` (written by
    `heat3d_fd_convergence_study`, see `apps/heat3d/README.md`): a single
    Fourier mode, swept over fd_order x N. No compute engine here --
    plotting only.
    """
    rows = read("heat3d_fd_order_convergence.csv")
    by_order = {}
    for r in rows:
        by_order.setdefault(int(r["fd_order"]), []).append(r)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    err_all = [float(r["l2_error"]) for r in rows if float(r["l2_error"]) > 0]
    curves = {}
    for order in sorted(by_order):
        pts = sorted(by_order[order], key=lambda r: -float(r["dx"]))
        dx = [float(r["dx"]) for r in pts]
        err = [float(r["l2_error"]) for r in pts]
        curves[order] = (dx, err)
        color = ORDER_COLORS.get(order, "#4a4f55")
        marker = ORDER_MARKERS.get(order, "o")
        ax.plot(dx, err, marker=marker, color=color, lw=1.6, ms=5.5,
                label=f"order {order}")

    # Reference slope triangles anchored to each curve's own coarsest
    # (dx, error) point, so the dotted line touches real data and the eye
    # can read the *slope* off the figure directly -- not an offset guess.
    for order in (2, 8, 12):
        if order not in curves:
            continue
        dx, err = curves[order]
        x0, y0 = dx[0], err[0]
        x1 = dx[-2]  # stop one point short of the round-off floor, if any
        y1 = y0 * (x1 / x0) ** order
        ax.plot([x0, x1], [y0, y1], ls=":", color="#8a8f98", lw=1.1, zorder=0)
        ax.annotate(f"slope {order}", xy=(x1, y1), xytext=(4, -2),
                    textcoords="offset points", fontsize=7.5, color="#6b7076")

    # The order-12 curve's finest point sits at the double-precision
    # round-off floor (see apps/heat3d/README.md): mark it rather than let
    # readers mistake the flattening for a stencil defect.
    floor = min(err_all) if err_all else None
    if floor is not None:
        ax.axhline(floor, ls=(0, (1, 2)), color="#8a8f98", lw=1.0, zorder=0)
        ax.text(0.02, 0.025, "round-off floor (order 12, N=64)", transform=ax.transAxes,
                fontsize=7.5, color="#6b7076", ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx (grid spacing)")
    ax.set_ylabel("L2 error vs single-mode analytic solution")
    ax.set_title("heat3d_fd: FD order-of-accuracy sweep", loc="left", fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    if err_all:
        ax.set_ylim(min(err_all) * 0.3, max(err_all) * 3.0)

    fig.tight_layout()
    out = HERE / "heat3d_fd_order_convergence.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def figure_tungsten_dealias_resolution():
    """How much the 2/3 dealias mask changes tungsten, against resolution.

    Reads `tungsten_dealias_resolution.csv` (written by
    `tungsten_dealias_study`, see `apps/tungsten/README.md`): the same seeded
    solidification run twice at each spacing, mask off and mask on. The point
    of the plot is that the gap between the two collapses once the grid clears
    six points per lattice period -- below that the grid cannot represent the
    crystal's own harmonics, so neither answer is right.
    """
    rows = sorted(read("tungsten_dealias_resolution.csv"),
                  key=lambda r: float(r["points_per_lattice"]))
    ppl = [float(r["points_per_lattice"]) for r in rows]
    dpower = [100.0 * float(r["rel_dpower"]) for r in rows]
    dmax = [100.0 * float(r["rel_dmax"]) for r in rows]
    dk1 = [100.0 * float(r["rel_dk1"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.semilogy(ppl, dpower, "o-", color=COLORS["tungsten"], linewidth=1.8,
                label="spectral power")
    ax.semilogy(ppl, dmax, "s-", color=COLORS["fd"], linewidth=1.6,
                label=r"peak density $\max\psi$")
    ax.semilogy(ppl, dk1, "^-", color=COLORS["spectral"], linewidth=1.6,
                label=r"selected wavenumber $k_1$")

    ax.axvline(6.0, color="#7a7a7a", linestyle="--", linewidth=1.0)
    ax.annotate(r"$\Delta x=\pi/3$: third harmonic fits under Nyquist" "\n"
                r"and the 2/3 cut clears $2k_0$",
                xy=(6.0, max(dpower)), xytext=(6.12, max(dpower) * 0.9),
                fontsize=7.5, color="#4a4a4a", va="top")
    ax.axvline(8.0, color="#7a7a7a", linestyle=":", linewidth=1.0)
    ax.annotate(r"$\Delta x=\pi/4$: 1/2 rule, cubic term exactly dealiased",
                xy=(8.0, max(dpower)), xytext=(7.9, max(dpower) * 0.9),
                fontsize=7.5, color="#4a4a4a", va="top", ha="right")

    ax.set_xlabel("grid points per lattice period")
    ax.set_ylabel("difference, mask on vs mask off  [%]")
    ax.set_title("Tungsten PFC: what dealiasing changes, by resolution")
    ax.grid(True, which="both", **GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    out = HERE / "tungsten_dealias_resolution.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def figure_tungsten_weak_16n():
    """Weak scaling to 16 nodes: constant work per GCD, growing problem.

    Reads `tungsten_hip_weak_16n.csv`. The local block is exactly
    2048 x 2048 x 16 at every point -- OpenPFC decomposes into z-slabs, so
    growing only Lz keeps it constant rather than approximately constant,
    which is what a cubic ladder would give. Perfect weak scaling is a flat
    line at the 1-node time.
    """
    rows = sorted(read("tungsten_hip_weak_16n.csv"), key=lambda r: int(r["gcds"]))
    nodes = [int(r["nodes"]) for r in rows]
    t = [float(r["wall_step_ms"]) for r in rows]
    eff = [100.0 * t[0] / v for v in t]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax.plot(nodes, t, "o-", color=COLORS["tungsten"], linewidth=1.8)
    ax.axhline(t[0], color="#7a7a7a", linestyle="--", linewidth=1.0)
    ax.annotate("ideal (flat)", xy=(nodes[0], t[0]), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color="#4a4a4a")
    ax.set_xscale("log", base=2); ax.set_xticks(nodes); ax.set_xticklabels(nodes)
    ax.set_xlabel("nodes (8 GCDs each)"); ax.set_ylabel("wall time per step [ms]")
    ax.set_title("Constant 67.1M cells per GCD")
    ax.grid(True, which="both", **GRID)

    ax2.plot(nodes, eff, "o-", color=COLORS["tungsten"], linewidth=1.8)
    for x, y, r in zip(nodes, eff, rows):
        ax2.annotate(f"{y:.0f}%", xy=(x, y), xytext=(0, -14),
                     textcoords="offset points", fontsize=7.5, ha="center")
    ax2.set_xscale("log", base=2); ax2.set_xticks(nodes); ax2.set_xticklabels(nodes)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("nodes"); ax2.set_ylabel("weak-scaling efficiency [%]")
    ax2.set_title(r"$8.59\times10^{9}$ cells at 16 nodes")
    ax2.grid(True, which="both", **GRID)
    fig.suptitle("Tungsten PFC weak scaling, LUMI-G", y=1.02)

    out = HERE / "tungsten_weak_16n.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def figure_tungsten_strong_1280():
    """Strong scaling at 1280^3, and what happens when the grid does not divide.

    Reads `tungsten_hip_strong_1280.csv`. Points where the z-planes divide
    evenly among the ranks are filled; the one that does not (96 GCDs,
    1280/96 = 13.33) is hollow and annotated, because it is slower than both
    its neighbours and that is the point of the figure.
    """
    rows = sorted(read("tungsten_hip_strong_1280.csv"), key=lambda r: int(r["gcds"]))
    g = [int(r["gcds"]) for r in rows]
    t = [float(r["wall_step_ms"]) for r in rows]
    ok = [r["divides"] == "1" for r in rows]
    base_t, base_g = t[0], g[0]
    speedup = [base_t / v for v in t]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ideal = [x / base_g for x in g]
    ax.plot(g, ideal, "--", color="#7a7a7a", linewidth=1.0, label="ideal")
    good = [(x, y) for x, y, k in zip(g, speedup, ok) if k]
    bad = [(x, y) for x, y, k in zip(g, speedup, ok) if not k]
    ax.plot([p[0] for p in good], [p[1] for p in good], "o-",
            color=COLORS["tungsten"], linewidth=1.8, label="z-planes divide evenly")
    if bad:
        ax.plot([p[0] for p in bad], [p[1] for p in bad], "o", markersize=9,
                markerfacecolor="white", markeredgecolor=COLORS["fd"],
                markeredgewidth=1.8, label="does not divide")
        bx, by = bad[0]
        ax.annotate("1280/96 = 13.33:\nranks get 13 or 14 planes,\nevery step waits for the slowest",
                    xy=(bx, by), xytext=(bx * 0.62, by * 1.9), fontsize=7.5,
                    color="#4a4a4a",
                    arrowprops=dict(arrowstyle="->", color="#7a7a7a", linewidth=0.8))
    for x, y, k in zip(g, speedup, ok):
        if k:
            ax.annotate(f"{100 * y / (x / base_g):.0f}%", xy=(x, y), xytext=(4, -10),
                        textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log", base=2); ax.set_xticks(g)
    ax.set_xticklabels([f"{x}\n({x // 8}n)" for x in g])
    ax.set_xlabel("GCDs (nodes)"); ax.set_ylabel(f"speedup vs {base_g} GCDs")
    ax.set_title(r"Tungsten PFC strong scaling, $1280^3$, LUMI-G")
    ax.grid(True, which="both", **GRID)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    out = HERE / "tungsten_strong_1280.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


def figure_heat3d_method_comparison():
    """Spectral against finite difference: cost at equal grid, and scaling.

    Reads `heat3d_method_cost.csv` and `heat3d_method_strong_1536.csv`. Left:
    what one step costs on the same grid and hardware, the only difference
    being the spatial operator. Right: how each method strong-scales to 16
    nodes. The two panels answer different questions and the chapter is
    careful not to conflate them -- cheap per step is not the same as cheap
    per unit accuracy.
    """
    cost = read("heat3d_method_cost.csv")
    spec_cost = next(float(r["wall_step_ms"]) for r in cost if r["method"] == "spectral")
    fd = sorted((r for r in cost if r["method"] == "fd"), key=lambda r: int(r["fd_order"]))
    orders = [int(r["fd_order"]) for r in fd]
    times = [float(r["wall_step_ms"]) for r in fd]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax.plot(orders, times, "o-", color=COLORS["fd"], linewidth=1.8,
            label="finite difference")
    ax.axhline(spec_cost, color=COLORS["spectral"], linestyle="--", linewidth=1.6,
               label="spectral")
    ax.annotate(f"spectral {spec_cost:.0f} ms\n= {spec_cost / times[0]:.0f}x FD-2",
                xy=(orders[-1], spec_cost), xytext=(orders[-1], spec_cost * 0.42),
                fontsize=8, color="#4a4a4a", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xlabel("finite-difference order"); ax.set_ylabel("wall time per step [ms]")
    ax.set_title(r"Cost at equal grid ($1024^3$, 8 GCDs)")
    ax.grid(True, which="both", **GRID)
    ax.legend(frameon=False, fontsize=8, loc="center right")

    strong = read("heat3d_method_strong_1536.csv")
    series = {}
    for r in strong:
        key = "spectral" if r["method"] == "spectral" else f"FD-{r['fd_order']}"
        series.setdefault(key, []).append((int(r["gcds"]), float(r["wall_step_ms"])))
    style = {"spectral": (COLORS["spectral"], "o-"), "FD-2": (COLORS["fd"], "s-"),
             "FD-8": (COLORS["tungsten"], "^-")}
    for key in ("spectral", "FD-2", "FD-8"):
        pts = sorted(series[key])
        g = [p[0] for p in pts]; t = [p[1] for p in pts]
        eff = [100.0 * t[0] / v / (x / g[0]) for x, v in zip(g, t)]
        c, m = style[key]
        ax2.plot([x // 8 for x in g], eff, m, color=c, linewidth=1.7, label=key)
    ax2.axhline(100, color="#7a7a7a", linestyle="--", linewidth=1.0)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([4, 8, 12, 16]); ax2.set_xticklabels([4, 8, 12, 16])
    ax2.set_ylim(0, 115)
    ax2.set_xlabel("nodes"); ax2.set_ylabel("strong-scaling efficiency [%]")
    ax2.set_title(r"Strong scaling, $1536^3$, 4$\to$16 nodes")
    ax2.grid(True, which="both", **GRID)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")

    out = HERE / "heat3d_method_comparison.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    figures = (figure_speedup(), figure_sizing(),
               figure_heat3d_fd_order_convergence(),
               figure_tungsten_dealias_resolution(),
               figure_tungsten_weak_16n(),
               figure_tungsten_strong_1280(),
               figure_heat3d_method_comparison())
    for path in figures:
        print("wrote", path.relative_to(HERE.parent.parent.parent))
