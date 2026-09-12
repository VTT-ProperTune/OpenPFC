#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thin-film FD-through-rupture report figures."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from field_io import read_vti


def _root() -> Path:
    env = os.environ.get("FIELD_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/thin-film")
    cands = sorted(scratch.glob("fd_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no fd_* dumps; set FIELD_DATA_DIR")
    return cands[-1]


def _show(ax, field, title: str, center: float = 1.0):
    vmin, vmax = float(field.data.min()), float(field.data.max())
    norm = TwoSlopeNorm(vcenter=center, vmin=min(vmin, 0.2), vmax=max(vmax, 1.5))
    ax.imshow(field.data, origin="lower", cmap="RdBu_r", norm=norm,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def _vti(run: Path, idx: int, saveat: float):
    p = run / "results" / f"h_{idx:04d}.vti"
    return read_vti(p, time=idx * saveat)


def main() -> None:
    root = _root()
    print("data", root)
    # saveat=10. Pick last existing frame per case plus a mid one.
    def latest(case: str):
        files = sorted((root / case / "results").glob("h_*.vti"))
        return files[-1]

    spont_late = latest("spontaneous")
    defect_late = latest("defect")
    si = int(spont_late.stem.split("_")[1])
    di = int(defect_late.stem.split("_")[1])
    sl = read_vti(spont_late, time=si * 10.0)
    dl = read_vti(defect_late, time=di * 10.0)
    print("spontaneous", spont_late.name, sl.data.min(), sl.data.max())
    print("defect", defect_late.name, dl.data.min(), dl.data.max())

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _show(ax, sl, "")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    out = HERE / "thin_film_fd_rupture.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", out)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    _show(axes[0], sl, rf"FD spontaneous  $t={si*10:g}$")
    _show(axes[1], dl, rf"FD defect  $t={di*10:g}$")
    fig.tight_layout()
    out2 = HERE / "thin_film_fd_pair.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
