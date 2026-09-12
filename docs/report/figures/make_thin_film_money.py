#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""512^2 thin-film money figures for the applications report.

Reads HIP dumps from apps/thin_film/slurm/report_512.sbatch
(FIELD_DATA_DIR or the latest /scratch/.../thin-film/fig512_* tree).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from field_io import read_vti
from field_plots import render_comparison


def _data_root() -> Path:
    env = os.environ.get("FIELD_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/thin-film")
    cands = sorted(scratch.glob("fig512_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no fig512_* dumps; set FIELD_DATA_DIR")
    return cands[-1]


def _save(fig, name: str) -> Path:
    out = HERE / name
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def _imshow(ax, field, center: float, vmin: float, vmax: float):
    norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    ax.imshow(
        field.data,
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        extent=field.extent,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def _latest(run_dir: Path, pattern: str):
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {run_dir}")
    return files[-1]


def figure_dewetting_wallpaper(root: Path) -> Path:
    late_p = _latest(root / "spontaneous/results/dewetting", "h_*.vti")
    # saveat=5; index i is t = i * 5
    idx = int(late_p.stem.split("_")[1])
    late = read_vti(late_p, time=idx * 5.0)
    ny, nx = late.data.shape
    assert (nx, ny) == (512, 512), (nx, ny)
    vmin = min(float(late.data.min()), 0.999)
    vmax = max(float(late.data.max()), 1.001)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _imshow(ax, late, center=1.0, vmin=vmin, vmax=vmax)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    return _save(fig, "thin_film_dewetting_512.png")


def figure_defect_pair(root: Path) -> Path:
    # Defect ruptures at t=85 (index 17); spontaneous has not.
    spont = read_vti(root / "spontaneous/results/dewetting/h_0017.vti", time=85.0)
    defect = read_vti(root / "defect/results/defect/h_0017.vti", time=85.0)
    fig = render_comparison(
        spont,
        defect,
        kind="diverging",
        center=1.0,
        label_a=r"spontaneous  $t=85$",
        label_b=r"defect-triggered  $t=85$",
        cbar_label="thickness $h$",
        axis_units="grid units",
        figsize=(9.2, 4.6),
    )
    return _save(fig, "thin_film_comparison.png")


def main() -> None:
    root = _data_root()
    print("data", root)
    figure_dewetting_wallpaper(root)
    figure_defect_pair(root)


if __name__ == "__main__":
    main()
