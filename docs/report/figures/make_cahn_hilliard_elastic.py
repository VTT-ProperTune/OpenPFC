#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Elastic Cahn–Hilliard report figures from the 256^2 campaign."""
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

from field_io import read_vti, read_vti_volume


def _root() -> Path:
    env = os.environ.get("FIELD_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/cahn-hilliard")
    cands = sorted(scratch.glob("elcamp_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no elcamp_* dumps; set FIELD_DATA_DIR")
    return cands[-1]


def _latest(run: Path):
    files = sorted((run / "results/fe_cr_elastic").glob("c_*.vti"))
    if not files:
        raise FileNotFoundError(run)
    return files[-1]


def _show(ax, field, title: str):
    vmin, vmax = float(field.data.min()), float(field.data.max())
    norm = TwoSlopeNorm(vcenter=0.5, vmin=min(vmin, 0.2), vmax=max(vmax, 0.8))
    ax.imshow(field.data, origin="lower", cmap="RdBu_r", norm=norm,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    root = _root()
    print("data", root)
    cases = {
        "none": r"no misfit",
        "isotropic": r"isotropic Vegard",
        "cubic": r"cubic Vegard $\langle 100\rangle$",
        "strong": r"cubic, $\varepsilon_0=0.08$ (suppressed)",
    }
    fields = {}
    for key in cases:
        p = _latest(root / key)
        idx = int(p.stem.split("_")[1])
        fields[key] = read_vti(p, time=idx * 20.0)
        print(key, p.name, fields[key].data.min(), fields[key].data.max())

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.8))
    order = ["none", "isotropic", "cubic", "strong"]
    for ax, key in zip(axes.ravel(), order):
        _show(ax, fields[key], cases[key])
    fig.tight_layout()
    out = HERE / "cahn_hilliard_elastic_quartet.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _show(ax, fields["cubic"], "")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    out2 = HERE / "cahn_hilliard_elastic_aligned.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", out2)


def _el64_root() -> Path | None:
    env = os.environ.get("EL64_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/cahn-hilliard")
    cands = sorted(scratch.glob("el64_*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _isometric(vol, thresh: float = 0.5, pix: int = 4):
    """Surface voxels of {c > thresh}, painter's isometric."""
    import numpy as np

    solid = vol > thresh
    nz, ny, nx = solid.shape
    neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    def at(i, j, k):
        return bool(solid[k % nz, j % ny, i % nx])

    w = (nx + nz + 2) * pix
    h = (ny + nz // 2 + 2) * pix
    img = np.full((h, w, 3), 238, dtype=np.uint8)  # cream
    navy = np.array([18, 22, 36], dtype=np.int16)
    teal = np.array([46, 140, 148], dtype=np.int16)
    ink = np.array([16, 16, 20], dtype=np.int16)
    white = np.array([255, 255, 255], dtype=np.int16)
    for k in range(nz):
        zf = k / max(nz - 1, 1)
        col = (navy + zf * (teal - navy)).astype(np.int16)
        top = (col + 0.18 * (white - col)).astype(np.int16)
        south = (col + 0.35 * (ink - col)).astype(np.int16)
        east = (col + 0.18 * (ink - col)).astype(np.int16)
        for j in range(ny):
            for i in range(nx):
                if not at(i, j, k):
                    continue
                surf = False
                for di, dj, dk in neigh:
                    if not at(i + di, j + dj, k + dk):
                        surf = True
                        break
                if not surf:
                    continue
                x = (i + k + 1) * pix
                y = (j + k // 2 + 1) * pix
                face = top if (k + 1 >= nz or not at(i, j, k + 1)) else col
                drop = pix // 2 + 1
                for dy in range(drop):
                    yy = y + pix + dy
                    if 0 <= yy < h:
                        img[yy, x : x + pix] = south
                    for dx in range(drop):
                        xx = x + pix + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            img[yy, xx] = south
                        y2 = y + dx
                        if 0 <= y2 < h and 0 <= xx < w:
                            img[y2, xx] = east
                img[y : y + pix, x : x + pix] = face
    return img


def figure_3d(root: Path) -> None:
    files = sorted((root / "results/fe_cr_elastic").glob("c_*.vti"))
    late = files[-1]
    # Prefer the last frame still inside (0,1); t=40 of job 21979862
    # hit the projection floor.
    vol = None
    for p in reversed(files):
        probe = read_vti_volume(p)
        if float(probe.min()) > 0.02:
            late = p
            vol = probe
            break
    if vol is None:
        vol = read_vti_volume(late)
    nz, ny, nx = vol.shape
    print("3d", late.name, vol.shape, vol.min(), vol.max())
    iso = _isometric(vol, thresh=0.5, pix=4)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    titles = [r"$xy$ mid", r"$xz$ mid", r"$yz$ mid"]
    slices = [vol[nz // 2], vol[:, ny // 2, :], vol[:, :, nx // 2]]
    for ax, sl, title in zip(axes, slices, titles):
        _show(ax, type("F", (), {"data": sl})(), title)
    fig.tight_layout()
    out = HERE / "cahn_hilliard_elastic_3d_slices.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.imshow(iso, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    out2 = HERE / "cahn_hilliard_elastic_3d_iso.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    if os.environ.get("SKIP_QUARTET") != "1":
        try:
            main()
        except SystemExit as e:
            print(e)
    r64 = _el64_root()
    if r64 is not None:
        figure_3d(r64)
    else:
        print("no el64_* dumps yet")

