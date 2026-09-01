#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grain-identity and microsegregation maps for the two-grain FTA runs.

    ./apps/alloy_pf_directional/scripts/plot_bicrystal.py results/alloy_pf_directional_bi_w10 \
        --label "W0 = 10 nm" --out results/alloy_pf_directional_bi_w10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm

mpl.use("Agg")

# grain 1 / grain 2 / liquid
C_G1 = "#c94f3d"
C_G2 = "#2f6fb5"
C_LIQ = "#eceff4"


def load_meta(path: Path) -> dict[str, float]:
    meta: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                meta[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return meta


def load_run(root: Path) -> dict:
    meta = load_meta(root / "meta.txt")
    nx, ny = int(meta["Nx"]), int(meta["Ny"])
    dx = float(meta["dx"])

    def raw(name: str) -> np.ndarray | None:
        p = root / name
        if not p.exists():
            return None
        return np.fromfile(p, dtype=np.float64).reshape(ny, nx)

    phi1 = raw("phi_final.raw")
    phi2 = raw("phi2_final.raw")
    c = raw("c_final.raw")
    if phi1 is None or c is None:
        raise SystemExit(f"{root}: missing phi_final.raw / c_final.raw")
    if phi2 is None:
        phi2 = np.full_like(phi1, -1.0)
    return {
        "meta": meta,
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "phi1": phi1,
        "phi2": phi2,
        "c": c,
        "extent": (0.0, nx * dx * 1e6, 0.0, ny * dx * 1e6),
    }


def grain_id(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """0 = liquid, 1 = grain 1, 2 = grain 2."""
    solid = np.clip(phi1 + phi2 + 1.0, -1.0, 1.0) > 0.0
    return np.where(solid, np.where(phi1 >= phi2, 1, 2), 0)


def overlap(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return 0.25 * (1.0 + phi1) * (1.0 + phi2)


def gb_rows(ov: np.ndarray) -> list[int]:
    """Row indices of the grain-boundary bands (local maxima of the y-profile)."""
    prof = ov.max(axis=1)
    if prof.max() <= 0.0:
        return []
    hits = np.where(prof > 0.5 * prof.max())[0]
    # collapse contiguous runs (with periodic wrap) to their centre
    rows, run = [], [hits[0]]
    for j in hits[1:]:
        if j == run[-1] + 1:
            run.append(j)
        else:
            rows.append(int(np.mean(run)))
            run = [j]
    rows.append(int(np.mean(run)))
    return rows


def zoom_window(run: dict, ov: np.ndarray) -> tuple[slice, slice]:
    """A box around the interior grain boundary, in the fully solid region."""
    ny, nx, dx = run["ny"], run["nx"], run["dx"]
    rows = [j for j in gb_rows(ov) if 0.15 * ny < j < 0.85 * ny]
    jc = rows[0] if rows else ny // 2
    half_y = max(8, int(round(0.45e-6 / dx)))
    solid = np.clip(run["phi1"] + run["phi2"] + 1.0, -1.0, 1.0) > 0.0
    cols = np.where(solid.any(axis=0))[0]
    xhi = cols.max() if cols.size else nx - 1
    width = max(20, int(round(2.2e-6 / dx)))
    x1 = int(max(0, xhi - width))
    return slice(max(0, jc - half_y), min(ny, jc + half_y + 1)), slice(x1, xhi + 1)


def sub_extent(run: dict, sy: slice, sx: slice) -> tuple[float, float, float, float]:
    d = run["dx"] * 1e6
    return (sx.start * d, sx.stop * d, sy.start * d, sy.stop * d)


def plot_grains(run: dict, label: str, out: Path) -> Path:
    gid = grain_id(run["phi1"], run["phi2"])
    ov = overlap(run["phi1"], run["phi2"])
    sy, sx = zoom_window(run, ov)
    cmap = ListedColormap([C_LIQ, C_G1, C_G2])

    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 5.6), gridspec_kw={"height_ratios": [1.0, 1.25]}
    )
    ax = axes[0]
    ax.imshow(gid, origin="lower", extent=run["extent"], cmap=cmap, vmin=-0.5, vmax=2.5,
              interpolation="nearest", aspect="auto")
    ax.contour(ov, levels=[0.02], origin="lower", extent=run["extent"],
               colors="k", linewidths=0.8)
    ax.set_title(f"{label}: grain 1 (+30°) / grain 2 (−30°) / liquid, "
                 f"black = grain boundary ($\\hat\\phi_1\\hat\\phi_2 = 0.02$)")
    ax.set_ylabel("y (µm)")
    rect = plt.Rectangle(
        (sx.start * run["dx"] * 1e6, sy.start * run["dx"] * 1e6),
        (sx.stop - sx.start) * run["dx"] * 1e6,
        (sy.stop - sy.start) * run["dx"] * 1e6,
        fill=False, ec="k", lw=1.2, ls="--",
    )
    ax.add_patch(rect)

    ax = axes[1]
    ax.imshow(gid[sy, sx], origin="lower", extent=sub_extent(run, sy, sx), cmap=cmap,
              vmin=-0.5, vmax=2.5, interpolation="nearest", aspect="auto")
    ax.contour(ov[sy, sx], levels=[0.02], origin="lower",
               extent=sub_extent(run, sy, sx), colors="k", linewidths=1.0)
    ax.set_title("zoom on the grain boundary where the two grains met")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=10, mfc=C_G1, mec="0.3", label="grain 1  (+30°)"),
        plt.Line2D([], [], marker="s", ls="", ms=10, mfc=C_G2, mec="0.3", label="grain 2  (−30°)"),
        plt.Line2D([], [], marker="s", ls="", ms=10, mfc=C_LIQ, mec="0.3", label="liquid"),
    ]
    axes[0].legend(handles=handles, loc="upper left", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    path = out / "bicrystal_grains.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_microsegregation(run: dict, label: str, out: Path) -> Path:
    c, meta = run["c"], run["meta"]
    c_inf = float(meta.get("c_inf", np.nan))
    gid = grain_id(run["phi1"], run["phi2"])
    ov = overlap(run["phi1"], run["phi2"])
    sy, sx = zoom_window(run, ov)
    norm = LogNorm(vmin=max(c.min(), 1e-3), vmax=c.max())

    fig = plt.figure(figsize=(11.0, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 0.95], hspace=0.42)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(c, origin="lower", extent=run["extent"], cmap="magma", norm=norm,
                   interpolation="nearest", aspect="auto")
    ax.contour(gid == 0, levels=[0.5], origin="lower", extent=run["extent"],
               colors="w", linewidths=0.6)
    ax.set_title(f"{label}: Cu concentration (wt%), white = solid/liquid interface")
    ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax, pad=0.012).set_label("c (wt% Cu)")

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(c[sy, sx], origin="lower", extent=sub_extent(run, sy, sx),
                   cmap="magma", norm=norm, interpolation="nearest", aspect="auto")
    ax.contour(ov[sy, sx], levels=[0.02], origin="lower",
               extent=sub_extent(run, sy, sx), colors="c", linewidths=1.0)
    ax.set_title("microsegregation at the grain boundary (cyan = GB)")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax, pad=0.012).set_label("c (wt% Cu)")

    # transverse profile through the GB, averaged over the zoom window in x
    ax = fig.add_subplot(gs[2])
    yv = (np.arange(sy.start, sy.stop) + 0.5) * run["dx"] * 1e6
    ax.semilogy(yv, c[sy, sx].mean(axis=1), color="k", lw=1.6, label="mean over zoom window")
    ax.semilogy(yv, c[sy, sx].max(axis=1), color="0.55", lw=1.0, ls="--", label="max")
    if np.isfinite(c_inf):
        ax.axhline(c_inf, color=C_G2, lw=1.0, ls=":", label=f"$c_\\infty$ = {c_inf:.2f} wt%")
    jgb = [j for j in gb_rows(ov) if sy.start <= j < sy.stop]
    for j in jgb:
        ax.axvline((j + 0.5) * run["dx"] * 1e6, color=C_G1, lw=1.0, ls="-.",
                   label="grain boundary" if j == jgb[0] else None)
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("c (wt% Cu)")
    ax.set_title("transverse solute profile across the grain boundary")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, ncol=4, loc="upper center")

    path = out / "bicrystal_microsegregation.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = load_run(args.root)
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or args.root.name

    gid = grain_id(run["phi1"], run["phi2"])
    ov = overlap(run["phi1"], run["phi2"])
    n = gid.size
    print(f"{label}: solid {(gid > 0).sum() / n:.4f}  "
          f"grain1 {(gid == 1).sum() / n:.4f}  grain2 {(gid == 2).sum() / n:.4f}  "
          f"both-solid {int(((run['phi1'] > 0) & (run['phi2'] > 0)).sum())}")
    print(f"  max overlap {ov.max():.4f}   c range [{run['c'].min():.4g}, {run['c'].max():.4g}]")

    for p in (plot_grains(run, label, out), plot_microsegregation(run, label, out)):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
