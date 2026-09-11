#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figures for the thermo-solutal-elastic dendrite chapter (@sec-alloy-dendrite).

Inputs are what `alloy_dendrite_growth --fields-dir=DIR` writes: one raw
Fortran-ordered `double` brick per field per snapshot, plus a JSON manifest
giving the grid, the field list and the snapshot times.  Everything else --
the CSV time series and summaries -- comes from `../data/`.

    make_alloy_dendrite_figures.py --fields-on  DIR_WITH_ELASTICITY \
                                   --fields-off DIR_WITHOUT \
                                   --data ../data --out ..

Figures produced
----------------
`alloy_dendrite_fields.svg`      the coupled state at steady tip: phi, U,
                                 theta, f_el, dfel/dphi, hydrostatic stress
`alloy_dendrite_growth.svg`      phi montage through the run
`alloy_dendrite_elastic_effect.svg`  elastic off vs on, tip region, and the
                                 difference field that makes the effect visible
`alloy_dendrite_tip_history.svg` tip velocity and radius against time for the
                                 off/on pair, with the fit window marked
`alloy_dendrite_selection.svg`   sigma* against grid spacing and stencil order
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

from field_io import Field2D, GridSpec, read_bin  # noqa: E402
import field_plots as fp  # noqa: E402


def load_manifest(d: Path) -> dict:
    hits = sorted(d.glob("*_manifest.json"))
    if not hits:
        raise SystemExit(f"{d}: no manifest; was the run given --fields-dir?")
    return json.loads(hits[0].read_text())


def snap(d: Path, man: dict, field: str, idx: int) -> Field2D:
    """One snapshot as a Field2D, mid-z plane, physical extent in W0."""
    g = GridSpec(nx=man["nx"], ny=man["ny"], nz=man["nz"],
                 dx=man["dx"], dy=man["dx"], dz=man["dx"])
    name = man["pattern"].replace("{field}", field).replace("{index:04d}", f"{idx:04d}")
    cube = read_bin(d / name, g)
    plane = cube[g.nz // 2, :, :]
    ex = g.axis_extent(g.nx, g.dx) + g.axis_extent(g.ny, g.dy)
    t = man["times"][idx] if idx < len(man["times"]) else None
    return Field2D(data=plane.astype(np.float64), extent=ex, name=field, time=t)


def crop(f: Field2D, frac: float) -> Field2D:
    """Central `frac` of each axis, so a tip is not four pixels of a big box."""
    ny, nx = f.data.shape
    hx, hy = int(nx * frac / 2), int(ny * frac / 2)
    cx, cy = nx // 2, ny // 2
    sub = f.data[cy - hy:cy + hy, cx - hx:cx + hx]
    x0, x1, y0, y1 = f.extent
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    ex = (x0 + (cx - hx) * dx, x0 + (cx + hx) * dx,
          y0 + (cy - hy) * dy, y0 + (cy + hy) * dy)
    return Field2D(data=sub, extent=ex, name=f.name, time=f.time)


def fig_state(d: Path, man: dict, idx: int, out: Path) -> None:
    """The whole coupled state in one figure. This is the chapter's anchor."""
    have_el = "f_el" in man["fields"]
    panels = [("phi", "diverging", 0.0, r"$\phi$"),
              ("U", "diverging", 0.0, r"$U$"),
              ("theta", "sequential", None, r"$\theta$")]
    if have_el:
        panels += [("f_el", "sequential", None, r"$f_{\rm el}$"),
                   ("dfel_dphi", "diverging", 0.0, r"$\partial f_{\rm el}/\partial\phi$"),
                   ("p_hydro", "diverging", 0.0, r"$\mathrm{tr}\,\sigma/3$")]
    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, (name, kind, centre, label) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        f = crop(snap(d, man, name, idx), 0.55)
        norm = fp._norm_for(f.data, kind, centre, None, None)
        im = ax.imshow(f.data, extent=f.extent, origin="lower",
                       cmap=fp._cmap_for(kind), norm=norm, interpolation="nearest")
        ax.set_title(label, fontsize=fp.TITLE_FS)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=fp.TICK_FS - 1)
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    t = man["times"][idx] if idx < len(man["times"]) else float("nan")
    fig.suptitle(f"coupled state at $t = {t:g}\\,\\tau_0$ "
                 f"(central 55% of a {man['nx']}$\\times${man['ny']} box)",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS + 0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_growth(d: Path, man: dict, out: Path, n: int = 5) -> None:
    idxs = np.linspace(0, len(man["times"]) - 1, n).round().astype(int)
    fields = [crop(snap(d, man, "phi", int(i)), 0.8) for i in idxs]
    fig = fp.render_montage(
        fields, kind="diverging", center=0.0,
        panel_titles=[f"$t={man['times'][int(i)]:g}$" for i in idxs],
        suptitle=r"$\phi$: seed to steady four-armed dendrite",
        cbar_label=r"$\phi$", axis_units=r"$W_0$")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_elastic_effect(don: Path, doff: Path, out: Path) -> None:
    mon, moff = load_manifest(don), load_manifest(doff)
    i = min(len(mon["times"]), len(moff["times"])) - 1
    a = crop(snap(doff, moff, "phi", i), 0.45)
    b = crop(snap(don, mon, "phi", i), 0.45)
    diff = Field2D(data=b.data - a.data, extent=a.extent, name="d phi", time=a.time)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, f, ttl, kind, ctr in (
            (axes[0], a, r"elasticity off", "diverging", 0.0),
            (axes[1], b, r"elasticity on", "diverging", 0.0),
            (axes[2], diff, r"$\phi_{\rm on}-\phi_{\rm off}$", "diverging", 0.0)):
        norm = fp._norm_for(f.data, kind, ctr, None, None)
        im = ax.imshow(f.data, extent=f.extent, origin="lower",
                       cmap=fp._cmap_for(kind), norm=norm, interpolation="nearest")
        ax.set_title(ttl, fontsize=fp.TITLE_FS)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=fp.TICK_FS - 1)
    fig.suptitle(f"$t = {a.time:g}\\,\\tau_0$. The difference panel is where the "
                 "effect lives: a shifted interface, not a changed bulk.",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _series(path: Path, col: str):
    import csv
    t, v = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                tv, vv = float(r["t"]), float(r[col])
            except (ValueError, KeyError):
                continue
            if np.isfinite(vv):
                t.append(tv); v.append(vv)
    return np.array(t), np.array(v)


def fig_tip_history(data: Path, out: Path, pairs, fit_fraction=0.3) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for label, fname in pairs:
        p = data / fname
        if not p.exists():
            print(f"  (skipping {fname}: not present)")
            continue
        for ax, col, ylab in ((axes[0], "v_tip", r"$V$  ($W_0/\tau_0$)"),
                              (axes[1], "rho_tip", r"$\rho$  ($W_0$)")):
            t, y = _series(p, col)
            ax.plot(t, y, lw=1.3, label=label)
            fp._style_line_axes(ax, r"$t$  ($\tau_0$)", ylab)
    tmax = max((ax.get_xlim()[1] for ax in axes), default=1.0)
    for ax in axes:
        ax.axvspan((1 - fit_fraction) * tmax, tmax, color="0.85", zorder=0)
        ax.legend(fontsize=fp.TICK_FS, frameon=False)
    fig.suptitle("tip velocity and radius; the shaded band is the fit window",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields-on", type=Path)
    ap.add_argument("--fields-off", type=Path)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.fields_on and args.fields_on.exists():
        man = load_manifest(args.fields_on)
        last = len(man["times"]) - 1
        fig_state(args.fields_on, man, last, args.out / "alloy_dendrite_fields.svg")
        fig_growth(args.fields_on, man, args.out / "alloy_dendrite_growth.svg")
    if (args.fields_on and args.fields_off
            and args.fields_on.exists() and args.fields_off.exists()):
        fig_elastic_effect(args.fields_on, args.fields_off,
                           args.out / "alloy_dendrite_elastic_effect.svg")
    fig_tip_history(args.data, args.out / "alloy_dendrite_tip_history.svg",
                    [("elasticity off", "alloy_dendrite_ts_off.csv"),
                     ("elasticity on", "alloy_dendrite_ts_on.csv")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
