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
    make_alloy_dendrite_figures.py --fta-dir /scratch/.../fta_<jobid> --out ..

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
`alloy_dendrite_fta.svg`         FTA directional campaign: downstream-cropped
                                 phi of aligned / misori / bicrystal
`alloy_dendrite_fta_tips.svg`    FTA tip histories against the isotherm
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


def crop_downstream(f: Field2D, pad_frac: float = 0.12) -> Field2D:
    """Crop around the most-downstream solid, not the box centre.

    FTA grows from the cold (small-`x`) side of a long cell. A central crop
    of a 640×256 box either misses the tip or fills the panel with melt.
    """
    ny, nx = f.data.shape
    solid = f.data > 0.0
    if not solid.any():
        return f
    ys, xs = np.nonzero(solid)
    x_lo = max(0, int(xs.min()) - int(pad_frac * nx))
    x_hi = min(nx, int(xs.max()) + max(4, int(pad_frac * nx)) + 1)
    y_lo = max(0, int(ys.min()) - int(pad_frac * ny))
    y_hi = min(ny, int(ys.max()) + int(pad_frac * ny) + 1)
    sub = f.data[y_lo:y_hi, x_lo:x_hi]
    x0, x1, y0, y1 = f.extent
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    ex = (x0 + x_lo * dx, x0 + x_hi * dx, y0 + y_lo * dy, y0 + y_hi * dy)
    return Field2D(data=sub, extent=ex, name=f.name, time=f.time)


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


def solid_frac(phi: Field2D, pad: float = 1.25) -> float:
    """Crop fraction that contains the solid with `pad` margin.

    A fixed crop is wrong for a growing dendrite: at t = 400 a 55% window
    wastes most of its area on melt, and by t = 900 the arms reach 91% of
    the box and the *tip* -- the thing being measured -- falls outside the
    window entirely. That is how the first version of this figure came to
    show only side branches. Size the window from the solid instead.
    """
    ny, nx = phi.data.shape
    solid = phi.data > 0.0
    if not solid.any():
        return 0.6
    ys, xs = np.nonzero(solid)
    cx, cy = nx / 2.0, ny / 2.0
    reach = max(np.abs(xs - cx).max() / (nx / 2.0),
                np.abs(ys - cy).max() / (ny / 2.0))
    return float(min(1.0, pad * reach))


def fig_state(d: Path, man: dict, idx: int, out: Path) -> None:
    """The whole coupled state in one figure. This is the chapter's anchor."""
    have_el = "f_el" in man["fields"]
    wanted = [("phi", "diverging", 0.0, r"$\phi$"),
              ("U", "diverging", 0.0, r"$U$"),
              ("theta", "sequential", None, r"$\theta$")]
    if have_el:
        wanted += [("f_el", "sequential", None, r"$f_{\rm el}$"),
                   ("dfel_dphi", "diverging", 0.0, r"$\partial f_{\rm el}/\partial\phi$"),
                   ("p_hydro", "diverging", 0.0, r"$\mathrm{tr}\,\sigma/3$"),
                   ("sig_vm", "sequential", None, r"$\sigma_{\rm vM}$")]
    # A panel of a field that is identically zero says nothing and costs a
    # sixth of the figure: the isothermal Stage-4 case keeps theta at zero by
    # construction, so its theta panel is a flat colour. Drop such panels
    # and say so in the caption rather than printing them.
    frac = solid_frac(snap(d, man, "phi", idx))
    panels, dropped = [], []
    for spec in wanted:
        f = crop(snap(d, man, spec[0], idx), frac)
        if float(np.ptp(f.data)) <= 0.0:
            dropped.append(spec[0])
        else:
            panels.append((spec, f))
    panels = panels[:6]
    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, ((name, kind, centre, label), f) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
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
    note = f" ($\\theta \\equiv 0$)" if "theta" in dropped else ""
    fig.suptitle(f"coupled state at $t = {t:g}\\,\\tau_0$, central "
                 f"{100*frac:.0f}% of a {man['nx']}$\\times${man['ny']} box{note}",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS + 0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_growth(d: Path, man: dict, out: Path, n: int = 5) -> None:
    idxs = np.linspace(0, len(man["times"]) - 1, n).round().astype(int)
    # One crop for every panel, sized on the *last* one, so the montage
    # shows growth rather than a sequence of same-sized blobs.
    frac = solid_frac(snap(d, man, "phi", int(idxs[-1])))
    fields = [crop(snap(d, man, "phi", int(i)), frac) for i in idxs]
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
    # The elastic-off run grows faster, so it is the one that sets the window.
    frac = solid_frac(snap(doff, moff, "phi", i))
    a = crop(snap(doff, moff, "phi", i), frac)
    b = crop(snap(don, mon, "phi", i), frac)
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


def fig_3d(slice_dir: Path, out: Path) -> None:
    """Mid-plane slices of the 512^3 coupled run.

    The 3-D bricks are 1.07 GB each and there are seven fields per snapshot,
    so they are not kept. `slices_*/` holds the mid-`z` and mid-`y` planes as
    `.npy`, extracted once from the run and small enough to live beside the
    report.
    """
    man = json.loads((slice_dir / "manifest.json").read_text())
    idx = man["slice_indices"][-1]
    t = man["times"][idx]
    dx = man["dx"]
    n = man["nx"]
    ex = (0.0, n * dx, 0.0, n * dx)
    panels = [("phi", "z", "diverging", 0.0, r"$\phi$, mid-$z$"),
              ("phi", "y", "diverging", 0.0, r"$\phi$, mid-$y$"),
              ("U", "z", "diverging", 0.0, r"$U$, mid-$z$"),
              ("f_el", "z", "sequential", None, r"$f_{\rm el}$, mid-$z$"),
              ("dfel_dphi", "z", "diverging", 0.0,
               r"$\partial f_{\rm el}/\partial\phi$, mid-$z$"),
              ("p_hydro", "z", "diverging", 0.0, r"$\mathrm{tr}\,\sigma/3$, mid-$z$")]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), squeeze=False)
    frac = None
    for i, (name, plane, kind, centre, label) in enumerate(panels):
        ax = axes[i // 3][i % 3]
        arr = np.load(slice_dir / f"{name}_{idx:04d}_{plane}.npy").T
        f = Field2D(data=arr, extent=ex, name=name, time=t)
        if frac is None:
            frac = solid_frac(Field2D(
                data=np.load(slice_dir / f"phi_{idx:04d}_z.npy").T,
                extent=ex, name="phi", time=t))
        f = crop(f, frac)
        norm = fp._norm_for(f.data, kind, centre, None, None)
        im = ax.imshow(f.data, extent=f.extent, origin="lower",
                       cmap=fp._cmap_for(kind), norm=norm, interpolation="nearest")
        ax.set_title(label, fontsize=fp.TITLE_FS)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=fp.TICK_FS - 1)
    fig.suptitle(f"$512^3$ coupled run on 2048 ranks, $t = {t:g}\\,\\tau_0$, "
                 f"central {100*frac:.0f}% of a {n}$^3$ box",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS + 0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    tstop = float("inf")
    for label, fname in pairs:
        p = data / fname
        if not p.exists():
            print(f"  (skipping {fname}: not present)")
            continue
        for ax, col, ylab in ((axes[0], "v_tip", r"$V$  ($W_0/\tau_0$)"),
                              (axes[1], "rho_tip", r"$\rho$  ($W_0$)")):
            t, y = _series(p, col)
            tstop = min(tstop, float(t[-1]))
            ax.plot(t, y, lw=1.3, label=label)
            fp._style_line_axes(ax, r"$t$  ($\tau_0$)", ylab)
    # Shade the fit window of the *shortest* run. The two runs do not end at
    # the same time -- the elastic-off dendrite grows faster and trips the
    # "tip has reached 84% of the half-box" guard sooner -- so shading to the
    # axis limit would paint a band over time neither run reached.
    for ax in axes:
        ax.axvspan((1 - fit_fraction) * tstop, tstop, color="0.88", zorder=0)
        ax.set_xlim(0.0, tstop * 1.02)
        ax.legend(fontsize=fp.TICK_FS, frameon=False)
    fig.suptitle(f"tip velocity and radius; the shaded band is the fit window "
                 f"(trailing {100*fit_fraction:.0f}% to $t={tstop:g}$)",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_fta(job: Path, out: Path) -> None:
    """Three FTA cases, downstream-cropped phi at the last snapshot."""
    cases = (("fta-aligned", r"aligned $\langle 100\rangle$"),
             ("fta-misori", r"misori $0.2\,\mathrm{rad}$"),
             ("fta-bicrystal", r"bicrystal $\pm 0.2\,\mathrm{rad}$"))
    panels = []
    for rid, label in cases:
        d = job / f"fields_{rid}"
        if not d.exists():
            print(f"  (skipping {rid}: no fields dir)")
            continue
        man = load_manifest(d)
        idx = len(man["times"]) - 1
        panels.append((label, crop_downstream(snap(d, man, "phi", idx)), man, idx))
    if not panels:
        print(f"  (no FTA field snapshots in {job})")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4),
                             squeeze=False)
    for ax, (label, f, man, idx) in zip(axes[0], panels):
        norm = fp._norm_for(f.data, "diverging", 0.0, None, None)
        im = ax.imshow(f.data, extent=f.extent, origin="lower",
                       cmap=fp._cmap_for("diverging"), norm=norm,
                       interpolation="nearest")
        ax.set_title(label, fontsize=fp.TITLE_FS)
        ax.set_xlabel(r"$x$ ($W_0$)", fontsize=fp.LABEL_FS)
        ax.set_ylabel(r"$y$ ($W_0$)", fontsize=fp.LABEL_FS)
        ax.tick_params(labelsize=fp.TICK_FS)
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=fp.TICK_FS - 1)
        t = man["times"][idx]
        ax.text(0.02, 0.98, rf"$t={t:g}\,\tau_0$", transform=ax.transAxes,
                va="top", ha="left", fontsize=fp.TICK_FS)
    fig.suptitle(r"FTA directional campaign: downstream crop of $\phi$ "
                 r"(not a central crop of the $640\times 256$ cell)",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_fta_tips(job: Path, out: Path, pulling: float = 0.05) -> None:
    """Tip histories against the isotherm for the FTA campaign."""
    pairs = (("fta-aligned", r"aligned $\langle 100\rangle$"),
             ("fta-misori", r"misori $0.2\,\mathrm{rad}$"),
             ("fta-bicrystal", r"bicrystal grain 1"),
             ("fta-misori-22.5", r"misori $22.5^\circ$"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    drawn = 0
    for rid, label in pairs:
        p = job / f"ts_{rid}.csv"
        if not p.exists():
            print(f"  (skipping {rid}: no time series)")
            continue
        t, xt = _series(p, "x_tip")
        _, vt = _series(p, "v_tip")
        tr, vr = _series(p, "v_rel")
        ti, xi = _series(p, "x_iso")
        if t.size:
            axes[0].plot(t, xt, lw=1.3, label=label)
            drawn += 1
        if vt.size:
            axes[1].plot(t, vt, lw=1.3, label=label)
        if vr.size:
            axes[2].plot(tr, vr, lw=1.3, label=label)
        if rid == "fta-aligned" and ti.size:
            axes[0].plot(ti, xi, lw=1.0, ls="--", color="0.4",
                         label=r"isotherm $x_0+V_p t$")
    if drawn == 0:
        print(f"  (no FTA CSVs in {job})")
        plt.close(fig)
        return
    axes[1].axhline(pulling, color="0.4", ls="--", lw=1.0, label=r"$V_p$")
    axes[2].axhline(0.0, color="0.4", ls="--", lw=1.0)
    fp._style_line_axes(axes[0], r"$t$  ($\tau_0$)", r"$x_{\mathrm{tip}}$  ($W_0$)")
    fp._style_line_axes(axes[1], r"$t$  ($\tau_0$)", r"$V_{\mathrm{tip}}$  ($W_0/\tau_0$)")
    fp._style_line_axes(axes[2], r"$t$  ($\tau_0$)", r"$V_{\mathrm{tip}}-V_p$")
    for ax in axes:
        ax.legend(fontsize=fp.TICK_FS - 1, frameon=False)
    fig.suptitle(r"FTA tip vs isotherm. $V_{\mathrm{rel}}<0$ lags the Bridgman field.",
                 x=0.01, ha="left", fontsize=fp.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields-on", type=Path)
    ap.add_argument("--fields-off", type=Path)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--slices-3d", type=Path,
                    help="slices_*/ directory from a 3-D coupled run")
    ap.add_argument("--fta-dir", type=Path,
                    help="job OUT from slurm/alloy_dendrite_fta.sbatch "
                         "(fields_*/ and ts_*.csv). Downstream crop, not central.")
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
    if args.slices_3d and args.slices_3d.exists():
        fig_3d(args.slices_3d, args.out / "alloy_dendrite_3d.svg")
    fig_tip_history(args.data, args.out / "alloy_dendrite_tip_history.svg",
                    [("elasticity off", "alloy_dendrite_ts_off.csv"),
                     ("elasticity on", "alloy_dendrite_ts_on.csv")])
    if args.fta_dir:
        if args.fta_dir.exists():
            fig_fta(args.fta_dir, args.out / "alloy_dendrite_fta.svg")
            fig_fta_tips(args.fta_dir, args.out / "alloy_dendrite_fta_tips.svg")
        else:
            print(f"{args.fta_dir}: not present; FTA figures wait on the "
                  "LUMI job. Re-run with --fta-dir pointing at "
                  "/scratch/project_462001519/juaho/alloy-dendrite/fta_<jobid>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
