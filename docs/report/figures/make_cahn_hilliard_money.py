#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""512^2 Cahn–Hilliard money figures for the applications report.

Reads the HIP dumps from slurm/report_512.sbatch (FIELD_DATA_DIR or the
latest /scratch/.../cahn-hilliard/fig512_* tree). High-resolution PNG so
a 512-cell field fills a report page without the 128^2 pixelation of the
old montage.
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

from field_io import read_vti  # noqa: E402
from field_plots import render_comparison  # noqa: E402


def _data_root() -> Path:
    env = os.environ.get("FIELD_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/cahn-hilliard")
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


def figure_spinodal_wallpaper(root: Path) -> Path:
    """Full 512^2 late science field — the money figure."""
    # t=4000 is index 40 at saveat=100.
    late = read_vti(root / "science/results/fe_cr_coarsening/c_0040.vti", time=4000.0)
    ny, nx = late.data.shape
    assert (nx, ny) == (512, 512), (nx, ny)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _imshow(ax, late, center=0.5, vmin=float(late.data.min()), vmax=float(late.data.max()))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    return _save(fig, "cahn_hilliard_spinodal_512.png")


def figure_coarsening_pair(root: Path) -> Path:
    """Early (wavelength selection) vs late (coarsened) — coarsening as a picture."""
    # t=400 is the first save where the binodals are reached (c in [0.32, 0.67]);
    # t=100 is still linear (amplitude 0.003) and looks blank on a shared scale.
    early = read_vti(root / "science/results/fe_cr_coarsening/c_0004.vti", time=400.0)
    late = read_vti(root / "science/results/fe_cr_coarsening/c_0040.vti", time=4000.0)
    fig = render_comparison(
        early,
        late,
        kind="diverging",
        center=0.5,
        label_a=r"$t=400$  ($L \approx 17.7$ cells)",
        label_b=r"$t=4000$  ($L \approx 37.3$ cells)",
        cbar_label="Cr mole fraction $c$",
        axis_units="grid units",
        figsize=(9.2, 4.6),
    )
    return _save(fig, "cahn_hilliard_coarsening_pair.png")


def figure_ic_comparison(root: Path) -> Path:
    from field_io import Field2D

    single = read_vti(root / "single/results/cahn_hilliard/c_0005.vti", time=10.0)
    broad = read_vti(root / "broadband/results/cahn_hilliard/c_0002.vti", time=10.0)

    def crop(field, n: int = 192) -> Field2D:
        ny, nx = field.data.shape
        i0, j0 = (nx - n) // 2, (ny - n) // 2
        x0, x1, y0, y1 = field.extent
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        return Field2D(
            data=field.data[j0:j0 + n, i0:i0 + n],
            extent=(x0 + i0 * dx, x0 + (i0 + n) * dx, y0 + j0 * dy, y0 + (j0 + n) * dy),
            name=field.name,
            time=field.time,
            units=field.units,
        )

    fig = render_comparison(
        crop(single),
        crop(broad),
        kind="diverging",
        center=0.32,
        label_a="single-mode seed (central $192\\times192$)",
        label_b="broadband noise (same window)",
        cbar_label="Cr mole fraction $c$",
        axis_units="grid units",
        figsize=(9.2, 4.6),
    )
    return _save(fig, "cahn_hilliard_comparison.png")


def main() -> None:
    root = _data_root()
    print("data", root)
    figure_spinodal_wallpaper(root)
    figure_coarsening_pair(root)
    figure_ic_comparison(root)


if __name__ == "__main__":
    main()
