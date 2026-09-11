#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figures for the Vlasov-Maxwell chapter (@sec-vlasov).

Inputs are what `vlasov_run --fields-dir=DIR` writes: one raw Fortran-ordered
`double` brick of `f(x, v_x, v_y)` per snapshot, plus a JSON manifest. The
brick axes are (x, v_x, v_y), which the manifest calls (nx, ny, nz) because
that is what the writer's grid fields are named -- the *phase space is the
grid*, and this is the one place that identification has to be undone to
read the file.

    make_vlasov_figures.py --two DIR --weibel DIR --landau DIR \
                           --data ../data --out ..
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import field_plots as fp  # noqa: E402


def load(d: Path):
    m = json.loads(sorted(d.glob("*_manifest.json"))[0].read_text())
    return m


def brick(d: Path, m: dict, idx: int) -> np.ndarray:
    """`f(x, v_x, v_y)` for snapshot `idx`."""
    name = m["pattern"].replace("{field}", "f").replace("{index:04d}", f"{idx:04d}")
    n = (m["nx"], m["ny"], m["nz"])
    return np.fromfile(d / name).reshape(n, order="F")


def xv_plane(d: Path, m: dict, idx: int) -> np.ndarray:
    """`f(x, v_x)`, integrated over `v_y`. The classic phase-space view."""
    return brick(d, m, idx).sum(axis=2)


def crop_v(a: np.ndarray, vmax: float, keep: float):
    """Trim the velocity axis to +-keep*vmax; most of a Maxwellian box is
    empty and plotting it wastes the panel on white space."""
    nv = a.shape[1]
    half = int(nv * keep / 2)
    c = nv // 2
    return a[:, max(0, c - half):min(nv, c + half)]


def fig_phase_montage(d: Path, out: Path, title: str, vlabel: str,
                      keep: float = 1.0, n: int = 4) -> None:
    m = load(d)
    times = m["times"]
    idxs = np.linspace(0, len(times) - 1, n).round().astype(int)
    planes = [crop_v(xv_plane(d, m, int(i)), 1.0, keep) for i in idxs]
    vmax = max(float(p.max()) for p in planes)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    im = None
    for j, (i, p) in enumerate(zip(idxs, planes)):
        ax = axes[0][j]
        im = ax.imshow(p.T, origin="lower", aspect="auto", cmap=fp.SEQUENTIAL_CMAP,
                       vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"$t={times[int(i)]:g}$", fontsize=fp.TITLE_FS)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if j == 0:
            ax.set_ylabel(vlabel, fontsize=fp.LABEL_FS)
            ax.set_xlabel("$x$", fontsize=fp.LABEL_FS)
    fig.suptitle(title, x=0.01, ha="left", fontsize=fp.TITLE_FS + 0.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    cax = fig.add_axes((0.94, 0.15, 0.015, 0.65))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\int f\,\mathrm{d}v_y$", fontsize=fp.LABEL_FS)
    cb.ax.tick_params(labelsize=fp.TICK_FS)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_velocity_plane(d: Path, out: Path, title: str) -> None:
    """`f(v_x, v_y)` summed over `x`: the anisotropy the Weibel mode eats."""
    m = load(d)
    idxs = [0, len(m["times"]) // 2, len(m["times"]) - 1]
    planes = [brick(d, m, i).sum(axis=0) for i in idxs]
    vmax = max(float(p.max()) for p in planes)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), squeeze=False)
    im = None
    for j, (i, p) in enumerate(zip(idxs, planes)):
        ax = axes[0][j]
        im = ax.imshow(p.T, origin="lower", cmap=fp.SEQUENTIAL_CMAP,
                       vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"$t={m['times'][i]:g}$", fontsize=fp.TITLE_FS)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        if j == 0:
            ax.set_xlabel("$v_x$", fontsize=fp.LABEL_FS)
            ax.set_ylabel("$v_y$", fontsize=fp.LABEL_FS)
    fig.suptitle(title, x=0.01, ha="left", fontsize=fp.TITLE_FS + 0.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.92))
    cax = fig.add_axes((0.94, 0.15, 0.015, 0.65))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\int f\,\mathrm{d}x$", fontsize=fp.LABEL_FS)
    cb.ax.tick_params(labelsize=fp.TICK_FS)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _series(path: Path, col: str):
    import csv
    t, v = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                a, b = float(r["t"]), float(r[col])
            except (ValueError, KeyError):
                continue
            if np.isfinite(b):
                t.append(a); v.append(b)
    return np.array(t), np.array(v)


def fig_rates(data: Path, out: Path) -> None:
    """The three linear benchmarks on one log axis, with their oracles."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    panels = [
        ("vlasov_ts_landau.csv", "mode_ex", -0.1533594669,
         r"Landau damping, $k\lambda_D=0.5$", r"$|\hat E_x(1)|$"),
        ("vlasov_ts_twostream.csv", "mode_ex", +0.3199435094,
         r"two-stream, electrostatic", r"$|\hat E_x(1)|$"),
        ("vlasov_ts_weibel.csv", "mode_bz", +0.05459243599,
         r"Weibel, electromagnetic", r"$|\hat B_z(1)|$"),
    ]
    for ax, (fn, col, rate, title, ylab) in zip(axes, panels):
        p = data / fn
        if not p.exists():
            print(f"  (skipping {fn})")
            continue
        t, y = _series(p, col)
        ok = y > 0
        ax.semilogy(t[ok], y[ok], lw=1.3, label="measured")
        # the oracle, anchored where the exponential phase begins
        # Anchor in the exponential phase and stop at saturation: an
        # oracle line drawn through the plateau invites the reader to see
        # a disagreement that is the physics, not an error.
        if rate < 0:
            i0 = len(t) // 8
            span = t[ok][-1]
        else:
            ymax = y.max()
            i0 = int(np.argmax(y > 5.0 * y[ok][0]))
            span = float(t[int(np.argmax(y > 0.3 * ymax))])
        t0, y0 = t[i0], y[i0]
        tt = np.linspace(t0, max(span, t0 + 1.0), 50)
        ax.semilogy(tt, y0 * np.exp(rate * (tt - t0)), "k--", lw=1.1,
                    label=f"oracle $\\gamma={rate:.4f}$")
        ax.set_title(title, fontsize=fp.TITLE_FS)
        fp._style_line_axes(ax, r"$t\ (\omega_{pe}^{-1})$", ylab)
        ax.legend(fontsize=fp.TICK_FS - 1, frameon=False)
    fig.suptitle("the three linear benchmarks against dispersion relations "
                 "solved numerically, not remembered",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--two", type=Path)
    ap.add_argument("--weibel", type=Path)
    ap.add_argument("--landau", type=Path)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.two and a.two.exists():
        fig_phase_montage(a.two, a.out / "vlasov_twostream_phase.svg",
                          r"electrostatic two-stream: $f(x,v_x)$ rolling up "
                          r"into phase-space vortices", "$v_x$", keep=0.55)
    if a.landau and a.landau.exists():
        fig_phase_montage(a.landau, a.out / "vlasov_landau_phase.svg",
                          r"Landau damping: the field decays while $f$ "
                          r"filaments in velocity", "$v_x$", keep=0.30)
    if a.weibel and a.weibel.exists():
        fig_phase_montage(a.weibel, a.out / "vlasov_weibel_phase.svg",
                          r"Weibel: $f(x,v_x)$ through magnetic saturation",
                          "$v_x$", keep=0.5)
        fig_velocity_plane(a.weibel, a.out / "vlasov_weibel_velocity.svg",
                           r"Weibel: the velocity anisotropy $f(v_x,v_y)$ "
                           r"relaxing as the magnetic field grows")
    fig_rates(a.data, a.out / "vlasov_rates.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
