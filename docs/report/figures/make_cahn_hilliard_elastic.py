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

from field_io import read_vti


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


if __name__ == "__main__":
    main()
