#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strip + front-zoom maps for the advertised 2D directional benchmark.

    ./apps/alloy_pf_directional/scripts/plot_benchmark.py \\
        results/alloy_pf_directional/benchmark/ly3.2_w10nm_bicrystal/reference \\
        --out results/alloy_pf_directional/benchmark/ly3.2_w10nm_bicrystal/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

mpl.use("Agg")

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

    def raw(name: str) -> np.ndarray:
        p = root / name
        if not p.exists():
            raise SystemExit(f"{root}: missing {name}")
        return np.fromfile(p, dtype=np.float64).reshape(ny, nx)

    phi1 = raw("phi_final.raw")
    phi2_path = root / "phi2_final.raw"
    phi2 = (
        np.fromfile(phi2_path, dtype=np.float64).reshape(ny, nx)
        if phi2_path.exists()
        else np.full_like(phi1, -1.0)
    )
    c = raw("c_final.raw")
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
    solid = np.clip(phi1 + phi2 + 1.0, -1.0, 1.0) > 0.0
    return np.where(solid, np.where(phi1 >= phi2, 1, 2), 0)


def front_extent(run: dict, width_um: float = 4.0) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = run["extent"]
    x_left = max(x0, x1 - width_um)
    return (x_left, x1, y0, y1)


def front_slice(run: dict, width_um: float = 4.0) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    dx_um = run["dx"] * 1e6
    n = max(1, int(round(width_um / dx_um)))
    sx = slice(max(0, run["nx"] - n), run["nx"])
    gid = grain_id(run["phi1"], run["phi2"])
    return gid[:, sx], run["c"][:, sx], front_extent(run, width_um)


def plot_strip(run: dict, out: Path, stem: str = "ly3.2_w10nm_strip") -> Path:
    gid = grain_id(run["phi1"], run["phi2"])
    cmap = ListedColormap([C_LIQ, C_G1, C_G2])
    c = run["c"]
    fig, axes = plt.subplots(
        2, 1, figsize=(11.4, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.05]}
    )
    ax = axes[0]
    ax.imshow(
        gid, origin="lower", extent=run["extent"], cmap=cmap, vmin=-0.5, vmax=2.5,
        interpolation="nearest", aspect="auto",
    )
    ax.contour(gid == 0, levels=[0.5], origin="lower", extent=run["extent"], colors="k", linewidths=0.35)
    ax.set_ylabel("y (µm)")
    ax.set_title("grains  (+30° red, −30° blue)  /  liquid")
    ax = axes[1]
    im = ax.imshow(
        c, origin="lower", extent=run["extent"], cmap="magma",
        norm=Normalize(vmin=float(np.percentile(c, 2)), vmax=float(np.percentile(c, 99.5))),
        interpolation="nearest", aspect="auto",
    )
    ax.contour(gid == 0, levels=[0.5], origin="lower", extent=run["extent"], colors="w", linewidths=0.35)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Cu (at%)")
    fig.colorbar(im, ax=ax, pad=0.012, fraction=0.025).set_label("c (at% Cu)")
    fig.tight_layout()
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_front(run: dict, out: Path, stem: str = "ly3.2_w10nm_front") -> Path:
    gid_z, c_z, ext = front_slice(run)
    cmap = ListedColormap([C_LIQ, C_G1, C_G2])
    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.08]}
    )
    ax = axes[0]
    ax.imshow(gid_z, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=2.5,
              interpolation="nearest", aspect="auto")
    ax.contour(gid_z == 0, levels=[0.5], origin="lower", extent=ext, colors="k", linewidths=0.6)
    ax.set_ylabel("y (µm)")
    ax.set_title("front (~last 4 µm): cells, GB V, grooves")
    ax = axes[1]
    im = ax.imshow(
        c_z, origin="lower", extent=ext, cmap="magma",
        norm=Normalize(vmin=float(np.percentile(run["c"], 2)), vmax=float(np.percentile(run["c"], 99.5))),
        interpolation="nearest", aspect="auto",
    )
    ax.contour(gid_z == 0, levels=[0.5], origin="lower", extent=ext, colors="w", linewidths=0.6)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Cu in intercellular liquid")
    fig.colorbar(im, ax=ax, pad=0.012, fraction=0.046).set_label("c (at% Cu)")
    fig.tight_layout()
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = load_run(args.root)
    out = args.out or (args.root / "figures")
    out.mkdir(parents=True, exist_ok=True)
    gid = grain_id(run["phi1"], run["phi2"])
    n = gid.size
    print(
        f"{args.root}: solid {(gid > 0).sum() / n:.4f}  "
        f"g1 {(gid == 1).sum() / n:.4f}  g2 {(gid == 2).sum() / n:.4f}"
    )
    for p in (plot_strip(run, out), plot_front(run, out)):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
