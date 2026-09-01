#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare single-grain Al-Cu DS runs in a fixed physical box."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTROOT = Path("/Users/tptatu/Data/OpenPFC/alloy_pf_directional_ds/until_right")


def load_meta(path: Path) -> dict[str, float]:
    meta: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            meta[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return meta


def load_phi(root: Path) -> tuple[dict[str, float], np.ndarray]:
    meta = load_meta(root / "meta.txt")
    nx = int(meta["Nx"])
    ny = int(meta["Ny"])
    phi = np.fromfile(root / "phi_final.raw", dtype=np.float64).reshape(ny, nx)
    return meta, phi


def plot_fields(runs: list[tuple[str, Path]], out: Path, title: str) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 1.9 * n), squeeze=False)
    for ax, (label, root) in zip(axes[:, 0], runs):
        meta, phi = load_phi(root)
        lx = meta["Nx"] * meta["dx"] * 1e6
        ly = meta["Ny"] * meta["dx"] * 1e6
        ax.imshow(
            phi,
            origin="lower",
            cmap="gray",
            vmin=-1.0,
            vmax=1.0,
            extent=(0.0, lx, 0.0, ly),
            aspect="equal",
        )
        ax.set_ylabel("y (μm)")
        ax.set_title(label, loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("x (μm)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_tips(runs: list[tuple[str, Path]], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, root in runs:
        hist = np.loadtxt(root / "history.tsv", comments="#")
        ax.plot(hist[:, 0] * 1e6, hist[:, 2] * 1e6, lw=1.5, label=label)
    ax.set_xlabel("t (μs)")
    ax.set_ylabel("leading tip x (μm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def existing(name: str) -> Path | None:
    p = OUTROOT / name
    if (p / "phi_final.raw").exists() and (p / "history.tsv").exists():
        return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTROOT / "figures")
    args = parser.parse_args()
    figdir = args.out_dir
    figdir.mkdir(parents=True, exist_ok=True)

    dx_runs = []
    for label, name in (
        (r"$\Delta x = W_0$ (5 nm)", "w0_5nm_dx1"),
        (r"$\Delta x = 0.4\,W_0$ (5 nm)", "w0_5nm_dx0.4"),
    ):
        p = existing(name)
        if p is not None:
            dx_runs.append((label, p))
    if len(dx_runs) >= 2:
        plot_tips(dx_runs, figdir / "ds_dx_tip.png", "Δx study, W0 = 5 nm, same physical box")
        if all((r / "phi_final.raw").exists() for _, r in dx_runs):
            plot_fields(dx_runs, figdir / "ds_dx_phi.png", "φ at stop (right wall or t cap), Δx study")

    w0_runs = []
    for label, name in (
        (r"$W_0=2.5$ nm", "w0_2.5nm_dx1"),
        (r"$W_0=5$ nm", "w0_5nm_dx1"),
        (r"$W_0=10$ nm", "w0_10nm_dx1"),
        (r"$W_0=20$ nm", "w0_20nm_dx1"),
    ):
        p = existing(name)
        if p is not None:
            w0_runs.append((label, p))
    if w0_runs:
        plot_tips(w0_runs, figdir / "ds_w0_tip.png", "W0 study, Δx = W0, same physical box")
        plot_fields(w0_runs, figdir / "ds_w0_phi.png", "φ at stop (right wall or t cap), W0 study")
    print(f"wrote figures under {figdir}")


if __name__ == "__main__":
    main()
