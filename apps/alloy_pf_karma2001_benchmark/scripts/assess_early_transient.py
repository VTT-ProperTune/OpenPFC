#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Early-time (t* ≤ 1500) diagnosis vs digitized Karma 2001 Fig. 1."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import compare_karma2001 as ck  # noqa: E402
import plot_figures as pf  # noqa: E402


def _v_from_r(hc: dict[str, np.ndarray], meta: dict[str, float | str]) -> np.ndarray:
    """Short-window dr/dt so the seed spike is not washed out by the t*≥50 LS."""
    d0 = float(meta["d0"])
    D = float(meta["D"])
    t = hc["t"]
    r = hc["r"]
    v = np.full_like(t, np.nan, dtype=float)
    for i in range(1, len(t)):
        i0 = max(0, i - 3)
        dt = t[i] - t[i0]
        if dt > 0.0:
            v[i] = (r[i] - r[i0]) / dt
    v[0] = v[1] if len(v) > 1 else 0.0
    return v * d0 / D


def _analytic_ic(meta: dict[str, float | str], r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w0 = float(meta["W0"])
    d0 = float(meta["d0"])
    k = float(meta["k"])
    u_inf = float(meta["u_inf"])
    r_seed = float(meta.get("r_seed", 22.0 * d0))
    cl0 = float(meta.get("cl0", 1.0))
    eta = (r - r_seed) / (math.sqrt(2.0) * w0)
    phi = -np.tanh(eta)
    den = 1.0 + k - (1.0 - k) * phi
    c_over = 0.5 * math.exp(u_inf) * den / cl0 * cl0
    return phi, c_over


def plot_digitization(out: Path) -> None:
    paper = pf.load_paper_xy(pf.FIG1_PAPER)
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax.plot(paper[:, 0], paper[:, 1], "k.-", ms=2.5, lw=1.0, label="digitized TSV")
    ax.set_xlim(0.0, 10000.0)
    ax.set_ylim(0.0, 0.08)
    ax.set_xlabel(r"$t D/d_0^2$")
    ax.set_ylabel(r"$V d_0/D$")
    ax.set_title("Full digitized present-model curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    axz.plot(paper[:, 0], paper[:, 1], "k.-", ms=3, lw=1.0)
    axz.set_xlim(0.0, 1500.0)
    axz.set_ylim(0.0, 0.08)
    axz.set_xlabel(r"$t D/d_0^2$")
    axz.set_title(r"Early window (published $Y$ clips at $0.08$)")
    axz.grid(True, alpha=0.25)
    t0, v0 = paper[0]
    axz.annotate(
        rf"first sample $t^*={t0:.0f}$, $V^*={v0:.3f}$",
        xy=(t0, v0),
        xytext=(400, 0.070),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_ic(runs: list[Path], out: Path) -> None:
    fig, (axp, axc) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)
    drawn = False
    for i, root in enumerate(runs):
        ic = root / "ic_profile.tsv"
        if not ic.exists():
            continue
        meta = pf.load_meta(pf.meta_path(root))
        pc = pf.profile_cols(pf.load_axis(ic))
        r = pc["r"]
        phi_a, c_a = _analytic_ic(meta, r)
        if not drawn:
            axp.plot(pc["r_over_d0"], phi_a, color="0.35", lw=3.0, alpha=0.45, label=r"analytic $\tanh$")
            axc.plot(pc["r_over_d0"], c_a, color="0.35", lw=3.0, alpha=0.45, label=r"analytic $c(u_\infty)$")
            drawn = True
        kw = pf.run_line_kw(i)
        axp.plot(pc["r_over_d0"], pc["phi"], label=ck.run_label(meta), **kw)
        axc.plot(pc["r_over_d0"], pc["c_over"], label=ck.run_label(meta), **kw)
    axp.set_ylabel(r"$\phi$")
    axp.set_title(r"Initial $\phi$ along the growth ray vs Karma tanh profile")
    axp.set_xlim(0.0, 50.0)
    axp.set_ylim(-1.05, 1.05)
    axp.legend(frameon=False, fontsize=7)
    axp.grid(True, alpha=0.25)
    axc.set_xlabel(r"$r/d_0$")
    axc.set_ylabel(r"$c/c_l^0$")
    axc.set_title(r"Initial $c$ (uniform $u=u_\infty$) vs eq. (18)")
    axc.set_ylim(0.0, 0.70)
    axc.legend(frameon=False, fontsize=7)
    axc.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_early_v(runs: list[Path], out: Path) -> None:
    paper = pf.load_paper_xy(pf.FIG1_PAPER) if pf.FIG1_PAPER.exists() else None
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.8, 7.6), gridspec_kw={"height_ratios": [1.3, 1.0]})
    if paper is not None:
        ax.plot(paper[:, 0], paper[:, 1], label="Karma 2001 digitized", **pf.PAPER_LINE)
    for i, root in enumerate(runs):
        meta = pf.load_meta(pf.meta_path(root))
        hc = pf.hist_tip_velocity(
            pf.hist_cols(pf.load_history(pf.history_path(root))), meta
        )
        kw = pf.run_line_kw(i)
        ax.plot(hc["t_star"], hc["V_star"], label=ck.run_label(meta), **kw)
        axr.plot(hc["t_star"], hc["r"] / float(meta["d0"]), **kw)
    ax.set_xlim(0.0, 1500.0)
    ax.set_ylim(0.0, 0.08)
    ax.set_ylabel(r"$V d_0/D$")
    ax.set_title(r"Early tip speed ($0 \leq t^* \leq 1500$)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.grid(True, alpha=0.25)
    axr.axhline(22.0, color="0.5", ls=":", lw=1.0, label=r"seed $R=22\,d_0$")
    axr.set_xlim(0.0, 1500.0)
    axr.set_xlabel(r"$t D/d_0^2$")
    axr.set_ylabel(r"$r_{\mathrm{tip}}/d_0$")
    axr.set_title("Tip position (same window)")
    axr.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    runs = [Path(r).resolve() for r in args.roots if pf.history_path(Path(r)).exists()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_digitization(out / "fig_digitization_fig1.png")
    print(f"wrote {out / 'fig_digitization_fig1.png'}")
    ic_runs = [r for r in runs if (r / "ic_profile.tsv").exists()]
    if ic_runs:
        plot_ic(ic_runs, out / "fig_ic_phi_c.png")
        print(f"wrote {out / 'fig_ic_phi_c.png'}")
    if runs:
        plot_early_v(runs, out / "fig_early_vstar.png")
        print(f"wrote {out / 'fig_early_vstar.png'}")
        print(f"{'run':<42} {'θ':>5} {'iso':>3} {'fd':>3} {'h':>3} {'V*(150)':>9} {'V*(500)':>9} {'V*(1500)':>9}")
        paper = pf.load_paper_xy(pf.FIG1_PAPER)
        for root in runs:
            meta = pf.load_meta(pf.meta_path(root))
            hc = pf.hist_tip_velocity(
                pf.hist_cols(pf.load_history(pf.history_path(root))), meta
            )
            v = hc["V_star"]
            t = hc["t_star"]

            def at(ts: float) -> float:
                if len(t) < 2:
                    return float("nan")
                return float(np.interp(ts, t, v))

            print(
                f"{root.name:<42} {ck._phi1_deg(meta):5.1f} "
                f"{int(float(meta.get('use_isotropic', 1))):3d} "
                f"{int(float(meta.get('fd_order', 2))):3d} "
                f"{int(float(meta.get('n_halves', 1))):3d} "
                f"{at(150):9.4f} {at(500):9.4f} {at(1500):9.4f}"
            )
        print(
            f"{'paper digitized':<42} {'—':>5} {'':>3} {'':>3} {'':>3} "
            f"{float(np.interp(150, paper[:,0], paper[:,1])):9.4f} "
            f"{float(np.interp(500, paper[:,0], paper[:,1])):9.4f} "
            f"{float(np.interp(1500, paper[:,0], paper[:,1])):9.4f}"
        )


if __name__ == "__main__":
    main()
