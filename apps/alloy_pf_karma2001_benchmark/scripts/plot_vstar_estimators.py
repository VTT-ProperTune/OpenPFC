#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How V*(t*) ringing changes with the tip-speed estimator.

Two families, both centered on each sample:

* Interpolating (classical high-order FD): polynomial degree = n−1 on n
  neighbour points. More points raise the formal order, but a staircased
  r_tip (grid pinning) is a high-frequency signal, so the amplitude often
  grows.
* Overdetermined linear LS: degree 1 on the same n points. Extra neighbours
  average pinning cycles and the oscillation drops; late mean V* stays put.

The paper Fig. 1 default is centered linear LS on a fixed Δt*=80 window
(~11 history samples on the fast Glasner [100] run).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import plot_figures as pf  # noqa: E402

LATE_LO, LATE_HI = 4000.0, 10000.0
NPTS_FD = (3, 5, 7, 9, 11)
NPTS_LS = (5, 11, 21, 41, 81)


def stencil_derivative(
    t: np.ndarray, r: np.ndarray, n_points: int, degree: int
) -> np.ndarray:
    """Centered derivative from an n-point stencil (n odd)."""
    if n_points % 2 != 1:
        raise ValueError("n_points must be odd")
    if degree >= n_points:
        raise ValueError("degree must be < n_points")
    n = len(t)
    v = np.full(n, np.nan)
    half = n_points // 2
    for i in range(half, n - half):
        sl = slice(i - half, i + half + 1)
        tau = t[sl] - t[i]
        scale = float(np.max(np.abs(tau)))
        if scale < 1.0e-30:
            continue
        coef = np.polynomial.polynomial.polyfit(tau / scale, r[sl], degree)
        # d r / d t = (d r / d u) / scale, u=tau/scale; linear term is coef[1]
        v[i] = coef[1] / scale
    return v


def late_stats(t_star: np.ndarray, v_star: np.ndarray) -> tuple[float, float, float]:
    m = np.isfinite(t_star) & np.isfinite(v_star) & (t_star >= LATE_LO) & (t_star <= LATE_HI)
    if np.count_nonzero(m) < 8:
        return float("nan"), float("nan"), float("nan")
    v = v_star[m]
    return float(np.mean(v)), float(np.std(v)), float(np.ptp(v))


def dtstar_of_n(t_star: np.ndarray, n_points: int) -> float:
    dt = np.median(np.diff(t_star))
    return float((n_points - 1) * dt)


def short_label(meta: dict[str, float | str]) -> str:
    d0w = float(meta.get("d0_over_W", 0.0))
    dxw = float(meta.get("dx_over_W", 0.0))
    return rf"$d_0/W={d0w:g}$, $\Delta x={dxw:g}\,W_0$"


def plot_estimators(runs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = pf.load_paper_xy(pf.FIG1_PAPER) if pf.FIG1_PAPER.exists() else None
    rows: list[dict[str, float | str]] = []

    ringing = runs[0]
    meta0 = pf.load_meta(pf.meta_path(ringing))
    hc0 = pf.hist_cols(pf.load_history(pf.history_path(ringing)))
    d0, D = float(meta0["d0"]), float(meta0["D"])
    t, r, t_star = hc0["t"], hc0["r"], hc0["t_star"]
    v80 = pf.hist_tip_velocity(hc0, meta0)["V_star"]

    fig, axes = plt.subplots(
        3, 1, figsize=(6.8, 9.4), gridspec_kw={"height_ratios": [1.15, 1.15, 1.05]}
    )
    ax_fd, ax_ls, ax_m = axes

    def draw_paper(ax: plt.Axes) -> None:
        if paper is None:
            return
        ax.plot(paper[:, 0], paper[:, 1], label="Karma 2001 present (digitized)", **pf.PAPER_LINE)

    draw_paper(ax_fd)
    ax_fd.plot(t_star, v80, color="0.35", lw=1.8, label=r"paper default: LS $\Delta t^*=80$", zorder=4)
    for i, npts in enumerate(NPTS_FD):
        v = stencil_derivative(t, r, npts, degree=npts - 1) * d0 / D
        mean, std, ptp = late_stats(t_star, v)
        order = npts - 1
        ax_fd.plot(
            t_star,
            v,
            label=rf"{npts}-pt interp. ($O(\Delta t^{{{order}}})$)",
            **pf.run_line_kw(i),
        )
        rows.append(
            {
                "run": ringing.name,
                "family": "interpolating",
                "n_points": npts,
                "degree": order,
                "dtstar": dtstar_of_n(t_star, npts),
                "V_star_late": mean,
                "V_star_std": std,
                "V_star_ptp": ptp,
            }
        )

    ax_fd.set_xlim(LATE_LO, LATE_HI)
    ax_fd.set_ylim(0.0, 0.05)
    ax_fd.set_ylabel(r"$V d_0 / D$")
    ax_fd.set_title(
        rf"Interpolating FD (higher order) — {short_label(meta0)}, late time"
    )
    ax_fd.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    ax_fd.grid(True, alpha=0.25)

    draw_paper(ax_ls)
    ax_ls.plot(t_star, v80, color="0.35", lw=1.8, label=r"paper default: LS $\Delta t^*=80$", zorder=4)
    for i, npts in enumerate(NPTS_LS):
        v = stencil_derivative(t, r, npts, degree=1) * d0 / D
        mean, std, ptp = late_stats(t_star, v)
        ax_ls.plot(
            t_star,
            v,
            label=rf"{npts}-pt linear LS ($\Delta t^*\approx{dtstar_of_n(t_star, npts):.0f}$)",
            **pf.run_line_kw(i),
        )
        rows.append(
            {
                "run": ringing.name,
                "family": "linear_ls",
                "n_points": npts,
                "degree": 1,
                "dtstar": dtstar_of_n(t_star, npts),
                "V_star_late": mean,
                "V_star_std": std,
                "V_star_ptp": ptp,
            }
        )
    ax_ls.set_xlim(LATE_LO, LATE_HI)
    ax_ls.set_ylim(0.0, 0.05)
    ax_ls.set_ylabel(r"$V d_0 / D$")
    ax_ls.set_title(r"Linear LS on more neighbours (same [100] Glasner run)")
    ax_ls.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    ax_ls.grid(True, alpha=0.25)

    for root, mk, col in zip(
        runs,
        ("o", "s", "D"),
        ("C0", "C1", "C2"),
        strict=True,
    ):
        meta = pf.load_meta(pf.meta_path(root))
        hc = pf.hist_cols(pf.load_history(pf.history_path(root)))
        tt, rr, ts = hc["t"], hc["r"], hc["t_star"]
        d0i, Di = float(meta["d0"]), float(meta["D"])
        stds, ns = [], []
        for npts in NPTS_LS:
            v = stencil_derivative(tt, rr, npts, degree=1) * d0i / Di
            _, std, _ = late_stats(ts, v)
            ns.append(npts)
            stds.append(std)
            if root != ringing:
                mean, std, ptp = late_stats(ts, v)
                rows.append(
                    {
                        "run": root.name,
                        "family": "linear_ls",
                        "n_points": npts,
                        "degree": 1,
                        "dtstar": dtstar_of_n(ts, npts),
                        "V_star_late": mean,
                        "V_star_std": std,
                        "V_star_ptp": ptp,
                    }
                )
        ax_m.plot(ns, stds, mk + "-", color=col, ms=6, lw=1.6, label=short_label(meta))
    std_fd = []
    for npts in NPTS_FD:
        v = stencil_derivative(t, r, npts, degree=npts - 1) * d0 / D
        std_fd.append(late_stats(t_star, v)[1])
    ax_m.plot(
        list(NPTS_FD),
        std_fd,
        "x--",
        color="0.25",
        ms=7,
        lw=1.2,
        label=rf"interpolating FD ({short_label(meta0)})",
    )
    ax_m.set_xlabel(r"neighbour points in the centered stencil")
    ax_m.set_ylabel(r"late std$(V d_0/D)$  ($4000\leq t^*\leq 10^4$)")
    ax_m.set_title(r"Ringing amplitude vs stencil size")
    ax_m.legend(frameon=False, fontsize=7)
    ax_m.grid(True, alpha=0.25)
    ax_m.set_xlim(0, 90)

    fig.tight_layout()
    out = out_dir / "fig_vstar_estimators.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")

    # Early-time cost of a wide LS window on the ringing run.
    fig2, ax = plt.subplots(figsize=(6.8, 3.6))
    draw_paper(ax)
    ax.plot(t_star, v80, color="0.35", lw=1.8, label=r"LS $\Delta t^*=80$", zorder=4)
    for i, npts in enumerate((11, 41, 81)):
        v = stencil_derivative(t, r, npts, degree=1) * d0 / D
        ax.plot(
            t_star,
            v,
            label=rf"{npts}-pt linear LS",
            **pf.run_line_kw(i),
        )
    ax.set_xlim(0.0, 2500.0)
    ax.set_ylim(0.0, 0.08)
    ax.set_xlabel(r"$t D / d_0^2$")
    ax.set_ylabel(r"$V d_0 / D$")
    ax.set_title(rf"Wide LS windows smear the early transient — {short_label(meta0)}")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig2.tight_layout()
    out2 = out_dir / "fig_vstar_estimators_early.png"
    fig2.savefig(out2, dpi=170)
    plt.close(fig2)
    print(f"wrote {out2}")

    tsv = out_dir / "vstar_estimators.tsv"
    fields = [
        "run",
        "family",
        "n_points",
        "degree",
        "dtstar",
        "V_star_late",
        "V_star_std",
        "V_star_ptp",
    ]
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {tsv}")
    print(
        f"{'run':<36} {'family':<14} {'n':>4} {'deg':>4} {'Δt*':>7} "
        f"{'V*late':>9} {'std':>9} {'ptp':>9}"
    )
    for row in rows:
        print(
            f"{str(row['run']):<36} {str(row['family']):<14} "
            f"{int(row['n_points']):4d} {int(row['degree']):4d} "
            f"{float(row['dtstar']):7.1f} {float(row['V_star_late']):9.5f} "
            f"{float(row['V_star_std']):9.5f} {float(row['V_star_ptp']):9.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="result directories with tip_history.tsv")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    runs = [Path(r).resolve() for r in args.roots]
    out_dir = Path(args.out_dir) if args.out_dir else runs[0].parent / "figures"
    plot_estimators(runs, out_dir)


if __name__ == "__main__":
    main()
