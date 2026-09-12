#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""3-D dendrite figures from a long isothermal 256^3 run."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent


def _root() -> Path:
    env = os.environ.get("FIELD_DATA_DIR")
    if env:
        return Path(env)
    scratch = Path("/scratch/project_462001519/juaho/alloy-dendrite")
    cands = sorted(scratch.glob("long3d_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no long3d_* dumps; set FIELD_DATA_DIR")
    return cands[-1]


def _load_phi(fields: Path, run: str, idx: int, nx: int, ny: int, nz: int) -> np.ndarray:
    p = fields / f"{run}_phi_{idx:04d}.bin"
    raw = np.fromfile(p, dtype="<f8")
    cube = raw.reshape((nx, ny, nz), order="F")
    return np.transpose(cube, (2, 1, 0))  # [nz, ny, nx]


def _show_phi(ax, sl, title: str):
    ax.imshow(sl, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    root = _root()
    man = json.loads((root / "fields" / "long3d_manifest.json").read_text())
    nx, ny, nz = int(man["nx"]), int(man["ny"]), int(man["nz"])
    times = man["times"]
    idx = len(times) - 1
    phi = _load_phi(root / "fields", "long3d", idx, nx, ny, nz)
    t = times[idx]
    print("phi", phi.shape, "t", t, "min", phi.min(), "max", phi.max(),
          "solid_frac", float((phi > 0).mean()))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    _show_phi(axes[0], phi[nz // 2], rf"$xy$ mid, $t={t:g}$")
    _show_phi(axes[1], phi[:, ny // 2, :], rf"$xz$ mid")
    _show_phi(axes[2], phi[:, :, nx // 2], rf"$yz$ mid")
    fig.tight_layout()
    out = HERE / "alloy_dendrite_3d_long_slices.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    # Mid-z is the classic 4-arm cross; save it large.
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _show_phi(ax, phi[nz // 2], "")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    out2 = HERE / "alloy_dendrite_3d_long_mid.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
