#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tungsten polycrystal report figures from HIP job dumps."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from numpy.lib.stride_tricks import sliding_window_view

from field_io import read_vti

HERE = Path(__file__).resolve().parent


def _root() -> Path:
    env = os.environ.get("FIELD_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/tungsten")
    cands = sorted(scratch.glob("poly_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no poly_* dumps; set FIELD_DATA_DIR")
    return cands[-1] / "results" / "tungsten_polycrystal"


def _show(ax, sl, title: str, n0: float = -0.10):
    norm = TwoSlopeNorm(vcenter=n0, vmin=float(sl.min()), vmax=float(sl.max()))
    ax.imshow(sl, origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    root = _root()
    f0 = read_vti(root / "psi_0000.vti", time=0)
    f1 = read_vti(root / "psi_0008.vti", time=400)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7))
    _show(axes[0], f0.data, r"$t=0$  (9 seeds)")
    _show(axes[1], f1.data, r"$t=400$")
    fig.tight_layout()
    fig.savefig(HERE / "tungsten_polycrystal_pair.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close()
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _show(ax, f1.data, "")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    fig.savefig(HERE / "tungsten_polycrystal.png", dpi=140, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close()
    w = 8
    pad = np.pad(f1.data, w // 2, mode="wrap")
    amp = sliding_window_view(pad, (w, w)).std(axis=(-1, -2))[: f1.data.shape[0],
                                                               : f1.data.shape[1]]
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.imshow(amp, origin="lower", cmap="cividis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    fig.savefig(HERE / "tungsten_polycrystal_amp.png", dpi=140, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close()
    print("wrote polycrystal figures from", root)


if __name__ == "__main__":
    main()
