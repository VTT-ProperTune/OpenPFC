#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare isothermal Karma 2001 runs to digitized PRL 87, 115701 (2001) Figs. 1–2.

Tip speed is V d0/D vs t D/d0^2 (Fig. 1). Solid concentration is sampled along the
fast <100> growth ray (ray_profile.tsv), not the box x-axis, so a 30°/45°
grain is compared on the same dendrite axis as [100].
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

TSTAR_STEADY = 5000.0  # late V* and cs plateau along [100] need at least this t*


def _phi1_deg(meta: dict[str, float | str]) -> float:
    return float(meta.get("phi1", meta.get("theta0", 0.0))) * 180.0 / np.pi


def _dx_over_w(meta: dict[str, float | str]) -> float:
    if "dx_over_W" in meta:
        return float(meta["dx_over_W"])
    w0 = float(meta.get("W0", 0.0))
    dx = float(meta.get("dx", 0.0))
    return dx / w0 if w0 > 0.0 else float("nan")


def run_label(meta: dict[str, float | str], *, with_dt: bool | None = None) -> str:
    d0w = float(meta.get("d0_over_W", 0.0))
    th = _phi1_deg(meta)
    dxw = _dx_over_w(meta)
    dtt = float(meta.get("dt_over_tau", 0.02))
    if abs(th) < 0.5:
        orient = r"$[100]$"
    else:
        orient = rf"$\theta={th:.0f}^\circ$"
    extra = ""
    at = float(meta.get("A_trap", 0.0))
    b0 = float(meta.get("beta0", 0.0))
    if at > 1.0e-12 or b0 > 1.0e-12:
        extra += rf", $A={at:.2f}$, $\beta_0={b0:g}$"
    show_dt = abs(dtt - 0.02) > 1.0e-6 if with_dt is None else with_dt
    if show_dt:
        extra += rf", $\Delta t={dtt:g}\,\tau_0$"
    tau_eu = float(meta.get("tau_eu_local", 1.0))
    if tau_eu < 0.5:
        extra += r", no $e^u$ on $\tau$"
    if float(meta.get("use_isotropic", 1.0)) < 0.5:
        extra += r", 5-pt"
    if float(meta.get("fd_order", 2.0)) >= 3.5:
        extra += r", 4th-order $\nabla^2$"
    if float(meta.get("use_glasner", 1.0)) < 0.5:
        extra += r", no Glasner"
    nh = int(round(float(meta.get("n_halves", 1.0))))
    if nh == 2:
        extra += r", 2-quad"
    elif nh == 4:
        extra += r", full plane"
    return rf"$d_0/W={d0w:g}$, {orient}, $\Delta x={dxw:.2g}\,W_0${extra}"


def paper_suite_label(meta: dict[str, float | str]) -> str:
    """Short legend for the advertised 3-case Fig. 1 (no Δt clutter)."""
    d0w = float(meta.get("d0_over_W", 0.0))
    dxw = _dx_over_w(meta)
    th = _phi1_deg(meta)
    orient = r"$[100]$" if abs(th) < 0.5 else rf"$\theta={th:.0f}^\circ$"
    if float(meta.get("use_glasner", 1.0)) < 0.5:
        return rf"$d_0/W={d0w:g}$, {orient}, $\Delta x={dxw:g}\,W_0$ (2001-like)"
    if abs(d0w - 0.544) < 0.02:
        return rf"$d_0/W={d0w:g}$, {orient}, $\Delta x={dxw:g}\,W_0$ (Glasner)"
    if dxw >= 0.95:
        return rf"$d_0/W={d0w:g}$, {orient}, $\Delta x={dxw:g}\,W_0$ (Glasner)"
    return rf"$d_0/W={d0w:g}$, {orient}, $\Delta x={dxw:g}\,W_0$ (Glasner)"


def _is_pinned_dxW0_277(meta: dict[str, float | str]) -> bool:
    return (
        abs(_dx_over_w(meta) - 1.0) < 0.05
        and abs(float(meta.get("d0_over_W", 0.0)) - 0.277) < 0.02
        and abs(_phi1_deg(meta)) < 0.5
    )


def late_vstar(hc: dict[str, np.ndarray], frac: float = 0.1) -> float:
    t = hc["t_star"]
    v = hc["V_star"]
    ok = np.isfinite(t) & np.isfinite(v)
    if not np.any(ok):
        return float("nan")
    t = t[ok]
    v = v[ok]
    tcut = t[-1] - frac * max(t[-1] - t[0], 1.0e-30)
    sel = t >= tcut
    return float(np.mean(v[sel])) if np.any(sel) else float(v[-1])


def late_vstar_stats(
    t_star: np.ndarray, v_star: np.ndarray, tlo: float = TSTAR_STEADY
) -> tuple[float, float, float]:
    """Mean, std and peak-to-peak of smoothed V* for t* ≥ tlo."""
    m = np.isfinite(t_star) & np.isfinite(v_star) & (t_star >= tlo)
    if np.count_nonzero(m) < 8:
        return float("nan"), float("nan"), float("nan")
    v = v_star[m]
    return float(np.mean(v)), float(np.std(v)), float(np.ptp(v))


def _pct_spread(values: list[float]) -> float:
    ok = [x for x in values if np.isfinite(x)]
    if len(ok) < 2:
        return float("nan")
    mid = float(np.mean(ok))
    if abs(mid) < 1.0e-30:
        return float("nan")
    return 100.0 * (max(ok) - min(ok)) / abs(mid)


def velocity_pinning_note(
    series: list[tuple[dict[str, float | str], str, np.ndarray, np.ndarray]],
) -> str:
    """On-figure caption: pinning on Δx=W0, and how well late ⟨V*⟩ agree."""
    rows: list[tuple[float, float, float, float]] = []
    has_pin = False
    for meta, _label, t_star, v_star in series:
        mu = late_vstar({"t_star": t_star, "V_star": v_star})
        _mean5000, sd, _ptp = late_vstar_stats(t_star, v_star)
        rows.append((_dx_over_w(meta), float(meta.get("d0_over_W", 0.0)), mu, sd))
        if _is_pinned_dxW0_277(meta):
            has_pin = True
    means = [r[2] for r in rows]
    spread = _pct_spread(means)
    # The two non-pinned paper cases: thicker W (d0/W=0.544) and Δx=0.4 W0.
    smooth = [
        r[2]
        for r in rows
        if not (abs(r[0] - 1.0) < 0.05 and abs(r[1] - 0.277) < 0.02)
    ]
    smooth_spread = _pct_spread(smooth)
    mean_txt = ", ".join(f"{m:.4f}" for m in means)
    lines = []
    if has_pin:
        lines.append(
            r"$\Delta x=W_0$ $[100]$ oscillations: grid pinning "
            r"(unchanged at $\Delta t/2$ and $\Delta t/4$)."
        )
    lines.append(
        rf"Late smoothed $\langle V^*\rangle$ (last 10% of $t^*$, "
        rf"$\Delta t^*=80$ LS): {mean_txt}."
    )
    if np.isfinite(spread):
        extra = ""
        if np.isfinite(smooth_spread) and len(smooth) >= 2:
            extra = f"; the two smoother cases agree to {smooth_spread:.1f}%"
        lines.append(f"Range is {spread:.0f}% of the mean{extra}.")
    return "\n".join(lines)


def vstar_rmse_vs_paper(t_star: np.ndarray, v_star: np.ndarray, paper: np.ndarray) -> float:
    if paper.size == 0 or len(t_star) < 2:
        return float("nan")
    lo = max(float(t_star[0]), float(paper[0, 0]))
    hi = min(float(t_star[-1]), float(paper[-1, 0]))
    if not (hi > lo):
        return float("nan")
    grid = np.linspace(lo, hi, 200)
    v_run = np.interp(grid, t_star, v_star)
    v_pap = np.interp(grid, paper[:, 0], paper[:, 1])
    return float(np.sqrt(np.mean((v_run - v_pap) ** 2)))


def grown_solid_mask(
    phi: np.ndarray,
    x_over_d0: np.ndarray,
    r_tip_over_d0: float,
    d0: float,
    w0: float,
    r_seed_over_d0: float,
) -> np.ndarray:
    """Solid grown after the seed, excluding the interface pile-up.

    The seed is not at the partition-k plateau; the ~k c_l^0 shelf lives at
    r ≳ 2 R_seed once the tip has run to t* ≳ 5000.
    """
    interior = pf.interior_solid_mask(phi, x_over_d0, r_tip_over_d0, d0, w0)
    return interior & (x_over_d0 > 2.0 * max(r_seed_over_d0, 22.0))


def cs_plateau(root: Path, meta: dict[str, float | str]) -> float:
    """Median c/c_l^0 on grown solid along the [100] ray (target ≈ k = 0.15)."""
    prof_path = pf.profile_path(root)
    if not prof_path.exists():
        return float("nan")
    pc = pf.profile_cols(pf.load_axis(prof_path))
    hist = pf.hist_cols(pf.load_history(pf.history_path(root)))
    d0 = float(meta["d0"])
    w0 = float(meta.get("W0", d0))
    r_tip = float(hist["r"][-1]) / d0
    r_seed = float(meta.get("r_seed_over_d0", 22.0))
    solid = grown_solid_mask(pc["phi"], pc["r_over_d0"], r_tip, d0, w0, r_seed)
    if not np.any(solid):
        return float("nan")
    return float(np.median(pc["c_over"][solid]))


def cs_rmse_vs_paper(root: Path, meta: dict[str, float | str], paper: np.ndarray) -> float:
    """RMSE of interior solid cs/cl0 vs Fig. 2 (excludes the interface pile-up)."""
    prof_path = pf.profile_path(root)
    if not prof_path.exists() or paper.size == 0:
        return float("nan")
    pc = pf.profile_cols(pf.load_axis(prof_path))
    hist = pf.hist_cols(pf.load_history(pf.history_path(root)))
    d0 = float(meta["d0"])
    w0 = float(meta.get("W0", d0))
    r_tip = float(hist["r"][-1]) / d0
    x = pc["r_over_d0"]
    solid = pf.interior_solid_mask(pc["phi"], x, r_tip, d0, w0)
    if not np.any(solid):
        return float("nan")
    xs = x[solid]
    c_run = pc["c_over"][solid]
    c_pap = np.interp(xs, paper[:, 0], paper[:, 1])
    return float(np.sqrt(np.mean((c_run - c_pap) ** 2)))


def _run_finite(root: Path) -> bool:
    hc = pf.hist_cols(pf.load_history(pf.history_path(root)))
    v = hc["V_star"]
    return bool(len(v) > 2 and np.isfinite(v).sum() > 2 and np.nanmax(np.abs(v)) < 1.0)


def _tstar_limits(runs: list[Path]) -> tuple[float, float]:
    tmax = 0.0
    for root in runs:
        hc = pf.hist_cols(pf.load_history(pf.history_path(root)))
        tmax = max(tmax, float(np.nanmax(hc["t_star"])))
    if tmax < 500.0:
        return 0.0, max(tmax * 1.15, 50.0)
    if tmax < 4000.0:
        return 0.0, max(tmax * 1.08, 200.0)
    return 0.0, 10000.0


def _window_ylim(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    paper: np.ndarray | None,
    tlo: float,
    thi: float,
    *,
    tmin_for_max: float = 0.0,
    ymax_floor: float = 0.02,
    ymax_cap: float | None = None,
    ymax_fixed: float | None = None,
    pad: float = 1.12,
) -> tuple[float, float]:
    if ymax_fixed is not None:
        return (0.0, ymax_fixed)
    vmax = ymax_floor
    for _, t, v in series:
        m = np.isfinite(t) & np.isfinite(v) & (t >= tlo) & (t <= thi) & (t >= tmin_for_max)
        if np.any(m):
            vmax = max(vmax, float(np.nanmax(v[m])))
    if paper is not None and paper.size:
        inwin = (paper[:, 0] >= tlo) & (paper[:, 0] <= thi) & (paper[:, 0] >= tmin_for_max)
        if np.any(inwin):
            vmax = max(vmax, float(np.nanmax(paper[inwin, 1])))
    ymax = vmax * pad
    if ymax_cap is not None:
        ymax = min(ymax_cap, ymax)
    return (0.0, ymax)


def plot_velocity(runs: list[Path], out: Path) -> None:
    paper = pf.load_paper_xy(pf.FIG1_PAPER) if pf.FIG1_PAPER.exists() else None
    fig, (ax, axe, axz) = plt.subplots(
        3, 1, figsize=(6.6, 9.6), gridspec_kw={"height_ratios": [1.35, 1.0, 1.15]}
    )
    x0, xmax = _tstar_limits(runs)
    if xmax < TSTAR_STEADY:
        print(
            f"warning: runs end at t*={xmax:.0f} < {TSTAR_STEADY:.0f}; "
            "late V* and the cs≈0.15 plateau are not established (use the paper suite, not QUICK=1)",
            file=sys.stderr,
        )
    series: list[tuple[dict[str, float | str], str, np.ndarray, np.ndarray]] = []
    for root in runs:
        meta = pf.load_meta(pf.meta_path(root))
        hc = pf.hist_tip_velocity(
            pf.hist_cols(pf.load_history(pf.history_path(root))), meta
        )
        series.append((meta, paper_suite_label(meta), hc["t_star"], hc["V_star"]))
    ylim_full = (0.0, 0.08)
    ylim_early = (0.0, 0.08)
    ylim_late = (0.0, 0.05)
    pin_i = next((i for i, (meta, *_rest) in enumerate(series) if _is_pinned_dxW0_277(meta)), None)

    def draw(ax_i: plt.Axes, xlim: tuple[float, float], ylim_i: tuple[float, float],
             legend: bool) -> None:
        for i, (_meta, label, t_star, v_star) in enumerate(series):
            ax_i.plot(t_star, v_star, label=label, **pf.run_line_kw(i))
        if paper is not None:
            pf.plot_paper_xy(ax_i, paper, label="Karma 2001 present (digitized)")
        ax_i.set_xlim(*xlim)
        ax_i.set_ylim(*ylim_i)
        ax_i.set_ylabel(r"$V d_0 / D$")
        ax_i.grid(True, alpha=0.25)
        if legend:
            handles, labels = ax_i.get_legend_handles_labels()
            if paper is not None and handles:
                handles = [handles[-1], *handles[:-1]]
                labels = [labels[-1], *labels[:-1]]
            ax_i.legend(handles, labels, frameon=False, fontsize=7.0, loc="upper right")

    draw(ax, (x0, xmax), ylim_full, True)
    ax.set_title(r"Tip speed vs Karma 2001 Fig. 1 (present model, $[100]$)")
    if xmax >= 2500.0:
        ax.axvspan(0.0, 2500.0, color="C0", alpha=0.05, zorder=0)
    if xmax >= 4000.0:
        ax.axvspan(4000.0, min(10000.0, xmax), color="C1", alpha=0.05, zorder=0)
        draw(axe, (0.0, 2500.0), ylim_early, False)
        axe.set_title(r"Early zoom ($0 \leq t^* \leq 2500$)")
        draw(axz, (4000.0, 10000.0), ylim_late, False)
        axz.set_title(r"Late zoom ($4000 \leq t^* \leq 10^{4}$)")
        if pin_i is not None:
            axz.annotate(
                r"$[100]$ grid pinning at $\Delta x=W_0$",
                xy=(7200.0, 0.026),
                xytext=(4300.0, 0.041),
                fontsize=7.5,
                color="0.15",
                arrowprops=dict(
                    arrowstyle="->",
                    color=pf.run_line_kw(pin_i)["color"],
                    lw=1.05,
                ),
                zorder=16,
            )
        note = velocity_pinning_note(series)
        if note:
            axz.text(
                0.02,
                0.03,
                note,
                transform=axz.transAxes,
                fontsize=7.0,
                va="bottom",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="0.75",
                    alpha=0.92,
                ),
                zorder=17,
            )
    else:
        draw(axe, (x0, xmax), ylim_early, False)
        axe.set_title("Early window (run has not reached $t^*=2500$)")
        draw(axz, (x0, xmax), ylim_full, False)
        axz.set_title("Same window (run has not reached $t^*=4000$)")
    axz.set_xlabel(r"$t D / d_0^2$")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_concentration(runs: list[Path], out: Path) -> None:
    """Top: full ray (shows the interface pile-up). Bottom: interior solid vs Fig. 2."""
    paper = pf.load_paper_xy(pf.FIG2_PAPER) if pf.FIG2_PAPER.exists() else None
    fig, (ax_full, ax_s) = plt.subplots(2, 1, figsize=(6.6, 7.4), sharex=False)
    rmax_data = 40.0
    for i, root in enumerate(runs):
        meta = pf.load_meta(pf.meta_path(root))
        prof_path = pf.profile_path(root)
        if not prof_path.exists():
            continue
        pc = pf.profile_cols(pf.load_axis(prof_path))
        hist = pf.hist_cols(pf.load_history(pf.history_path(root)))
        d0 = float(meta["d0"])
        w0 = float(meta.get("W0", d0))
        r_tip = float(hist["r"][-1]) / d0
        x = pc["r_over_d0"]
        kw = pf.run_line_kw(i)
        ax_full.plot(x, pc["c_over"], label=run_label(meta), **kw)
        ax_full.axvline(r_tip, color=kw["color"], ls=":", lw=1.0, alpha=0.7, zorder=4)
        seed = float(meta.get("r_seed_over_d0", 22.0))
        grown = grown_solid_mask(pc["phi"], x, r_tip, d0, w0, seed)
        solid = grown if np.any(grown) else pf.interior_solid_mask(pc["phi"], x, r_tip, d0, w0)
        if np.any(solid):
            ax_s.plot(x[solid], pc["c_over"][solid], label=run_label(meta), **kw)
            rmax_data = max(rmax_data, float(x[solid][-1]) * 1.2)
        else:
            rmax_data = max(rmax_data, r_tip)
    if paper is not None:
        pf.plot_paper_xy(
            ax_full,
            paper,
            label="Karma 2001 Fig. 2 (digitized $c_s$)",
        )
        pf.plot_paper_xy(
            ax_s,
            paper,
            label="Karma 2001 Fig. 2 (digitized $c_s$)",
        )
    paper_xmax = float(paper[-1, 0]) if paper is not None and paper.size else 50.0
    xmax = max(rmax_data, paper_xmax, 50.0)
    kpart = 0.15
    if runs:
        kpart = float(pf.load_meta(pf.meta_path(runs[0])).get("k", 0.15))
    ax_full.axhline(kpart, color="0.35", ls="--", lw=1.0, zorder=3)
    ax_s.axhline(kpart, color="0.35", ls="--", lw=1.0, label=rf"$k={kpart:g}$", zorder=3)
    ax_full.set_ylabel(r"$c / c_l^0$ (full ray)")
    ax_full.set_title(r"Full growth-ray profile (interface pile-up is not $c_s$)")
    ax_full.set_xlim(0.0, xmax)
    ax_full.set_ylim(0.0, 0.70)
    ax_full.legend(frameon=False, fontsize=7)
    ax_full.grid(True, alpha=0.25)
    ax_s.set_xlabel(r"$r / d_0$ along fast $\langle 100\rangle$")
    ax_s.set_ylabel(r"$c_s / c_l^0$ (grown solid)")
    ax_s.set_title(r"Grown solid vs Fig. 2 (behind seed, cut $6W_0$ behind the tip); shelf $\approx k$")
    ax_s.set_xlim(0.0, xmax)
    ax_s.set_ylim(0.0, 0.40)
    ax_s.legend(frameon=False, fontsize=7)
    ax_s.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def write_metrics(runs: list[Path], out: Path) -> None:
    paper_v = pf.load_paper_xy(pf.FIG1_PAPER) if pf.FIG1_PAPER.exists() else np.empty((0, 2))
    paper_c = pf.load_paper_xy(pf.FIG2_PAPER) if pf.FIG2_PAPER.exists() else np.empty((0, 2))
    rows: list[dict[str, str | float]] = []
    for root in runs:
        meta = pf.load_meta(pf.meta_path(root))
        hc = pf.hist_tip_velocity(
            pf.hist_cols(pf.load_history(pf.history_path(root))), meta
        )
        v_late = late_vstar(hc)
        v_mean5000, v_std5000, v_ptp5000 = late_vstar_stats(hc["t_star"], hc["V_star"])
        t_last = float(hc["t_star"][-1])
        v_paper = (
            float(np.interp(t_last, paper_v[:, 0], paper_v[:, 1]))
            if paper_v.size
            else float("nan")
        )
        r_last = float(hc["r"][-1]) / max(float(meta.get("d0", 1.0)), 1.0e-30)
        c_wall = float(hc["c_wall"][-1]) if "c_wall" in hc else float("nan")
        rows.append(
            {
                "run": root.name,
                "d0_over_W": float(meta.get("d0_over_W", float("nan"))),
                "phi1_deg": _phi1_deg(meta),
                "dx_over_W": _dx_over_w(meta),
                "L_over_d0": float(meta.get("L_over_d0", float("nan"))),
                "r_over_d0": r_last,
                "c_wall": c_wall,
                "A_trap": float(meta.get("A_trap", float("nan"))),
                "beta0": float(meta.get("beta0", float("nan"))),
                "k": float(meta.get("k", float("nan"))),
                "eps_c": float(meta.get("eps_c", float("nan"))),
                "Omega": float(meta.get("Omega", float("nan"))),
                "Tdot": float(meta.get("Tdot", float("nan"))),
                "dt_over_tau": float(meta.get("dt_over_tau", float("nan"))),
                "tau_eu_local": float(meta.get("tau_eu_local", 1.0)),
                "fourier": float(meta.get("fourier", float("nan"))),
                "t_star_last": t_last,
                "V_star_late": v_late,
                "V_star_mean_t5000": v_mean5000,
                "V_star_std_t5000": v_std5000,
                "V_star_ptp_t5000": v_ptp5000,
                "V_star_paper_at_t": v_paper,
                "V_star_rel_err": (v_late - v_paper) / v_paper
                if (v_paper and np.isfinite(v_paper) and abs(v_paper) > 1e-30)
                else float("nan"),
                "V_star_rmse": vstar_rmse_vs_paper(hc["t_star"], hc["V_star"], paper_v),
                "cs_rmse_fig2": cs_rmse_vs_paper(root, meta, paper_c),
                "cs_plateau": cs_plateau(root, meta),
            }
        )
    fields = [
        "run",
        "d0_over_W",
        "phi1_deg",
        "dx_over_W",
        "L_over_d0",
        "r_over_d0",
        "c_wall",
        "A_trap",
        "beta0",
        "k",
        "eps_c",
        "Omega",
        "Tdot",
        "dt_over_tau",
        "tau_eu_local",
        "fourier",
        "t_star_last",
        "V_star_late",
        "V_star_mean_t5000",
        "V_star_std_t5000",
        "V_star_ptp_t5000",
        "V_star_paper_at_t",
        "V_star_rel_err",
        "V_star_rmse",
        "cs_rmse_fig2",
        "cs_plateau",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out}")
    print(
        f"{'run':<36} {'t*':>8} {'V*late':>10} {'⟨V*⟩5k':>10} {'std5k':>10} {'cs plat.':>10}"
    )
    for row in rows:
        tlast = float(row["t_star_last"])
        flag = "" if tlast >= TSTAR_STEADY else "  INCOMPLETE (t*<5000)"
        print(
            f"{str(row['run']):<36} {tlast:8.0f} "
            f"{float(row['V_star_late']):10.5f} "
            f"{float(row['V_star_mean_t5000']):10.5f} "
            f"{float(row['V_star_std_t5000']):10.5f} "
            f"{float(row['cs_plateau']):10.5f}{flag}"
        )


def _unique_paths(runs: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out_runs: list[Path] = []
    for root in runs:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        out_runs.append(root)
    return out_runs


def dx_scan_label(meta: dict[str, float | str]) -> str:
    dxw = _dx_over_w(meta)
    if float(meta.get("use_glasner", 1.0)) < 0.5:
        return rf"$\Delta x={dxw:g}\,W_0$ (2001-like)"
    return rf"$\Delta x={dxw:g}\,W_0$ (Glasner)"


def plot_dx_pinning(runs: list[Path], out: Path) -> None:
    """d0/W=0.277 [100]: where Δx pinning in V* becomes strong."""
    rows: list[dict[str, object]] = []
    for root in _unique_paths(runs):
        meta = pf.load_meta(pf.meta_path(root))
        if abs(float(meta.get("d0_over_W", 0.0)) - 0.277) > 0.02:
            continue
        if abs(_phi1_deg(meta)) > 0.5:
            continue
        hc = pf.hist_tip_velocity(
            pf.hist_cols(pf.load_history(pf.history_path(root))), meta
        )
        mu, sd, ptp = late_vstar_stats(hc["t_star"], hc["V_star"])
        rows.append(
            {
                "dx": _dx_over_w(meta),
                "label": dx_scan_label(meta),
                "t": hc["t_star"],
                "v": hc["V_star"],
                "mean": mu,
                "std": sd,
                "ptp": ptp,
            }
        )
    rows.sort(key=lambda r: float(r["dx"]))
    if not rows:
        raise SystemExit("no d0/W=0.277 [100] runs for the Δx pinning figure")

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(6.6, 7.2), gridspec_kw={"height_ratios": [1.35, 1.0]}
    )
    for i, row in enumerate(rows):
        ax.plot(
            row["t"],
            row["v"],
            label=str(row["label"]),
            **pf.run_line_kw(i),
        )
    ax.set_xlim(4000.0, 10000.0)
    ax.set_ylim(0.008, 0.034)
    ax.set_ylabel(r"$V d_0 / D$")
    ax.set_title(r"$d_0/W=0.277$ $[100]$: $V^*$ pinning vs $\Delta x$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.06,
        r"Wiggle is weak at $0.4$–$0.6\,W_0$, clear at $0.8\,W_0$, strong at $W_0$.",
        transform=ax.transAxes,
        fontsize=8,
        color="0.25",
    )

    xs = [float(r["dx"]) for r in rows]
    stds = [float(r["std"]) for r in rows]
    ptps = [float(r["ptp"]) for r in rows]
    axb.plot(xs, stds, "o-", color="C0", lw=1.6, ms=7, label=r"std of $V^*$")
    axb.plot(xs, ptps, "s--", color="C1", lw=1.4, ms=6, label=r"peak-to-peak of $V^*$")
    axb.set_xlabel(r"$\Delta x / W_0$")
    axb.set_ylabel(r"late $V^*$ scatter ($t^*\geq 5000$)")
    axb.set_xlim(0.3, 1.08)
    axb.grid(True, alpha=0.25)
    axb.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_dt_scan(runs: list[Path], out: Path) -> None:
    """Timestep study: V*(t) and late V* vs Δt/τ₀."""
    paper = pf.load_paper_xy(pf.FIG1_PAPER) if pf.FIG1_PAPER.exists() else None
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(6.6, 7.0), gridspec_kw={"height_ratios": [1.35, 1.0]}
    )
    series = []
    dtt_v = []
    vmax = 0.02
    for root in runs:
        meta = pf.load_meta(pf.meta_path(root))
        hc = pf.hist_tip_velocity(
            pf.hist_cols(pf.load_history(pf.history_path(root))), meta
        )
        dtt = float(meta.get("dt_over_tau", float("nan")))
        fo = float(meta.get("fourier", float("nan")))
        v_late = late_vstar(hc)
        series.append((run_label(meta, with_dt=True), hc["t_star"], hc["V_star"]))
        dtt_v.append((dtt, v_late, fo, float(hc["t_star"][-1])))
        if len(hc["V_star"]):
            vmax = max(vmax, float(np.nanmax(hc["V_star"])))
    if paper is not None:
        vmax = max(vmax, float(np.nanmax(paper[:, 1])))
        v_paper_late = float(paper[-1, 1])
    else:
        v_paper_late = float("nan")
    for i, (label, t_star, v_star) in enumerate(series):
        ax.plot(t_star, v_star, label=label, **pf.run_line_kw(i))
    if paper is not None:
        pf.plot_paper_xy(ax, paper, label="Karma 2001 present (digitized)")
    ax.set_xlim(0.0, 10000.0)
    ax.set_ylim(0.0, 0.08)
    ax.set_ylabel(r"$V d_0 / D$")
    ax.set_xlabel(r"$t D / d_0^2$")
    ax.set_title(r"$d_0/W=0.277$, $\theta=45^\circ$, $\Delta x=W_0$: $\Delta t$ dependence")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    dtt_v.sort(key=lambda r: r[0])
    xs = [r[0] for r in dtt_v]
    ys = [r[1] for r in dtt_v]
    axb.plot(xs, ys, "o-", color="C0", lw=1.6, ms=6, label="late $V^*$")
    if np.isfinite(v_paper_late):
        axb.axhline(v_paper_late, color="0.35", ls="--", lw=1.2, label="paper at $t^*=10^4$")
    # Ji 2D isotropic Laplacian: |λ|_max h² = 16/3, Euler Fo ≤ 3/8.
    axb.axvline(0.1875, color="0.45", ls=":", lw=1.2, label=r"iso. VN $\Delta t/\tau_0=0.1875$")
    axb.axvline(0.20, color="0.65", ls=":", lw=1.0, label=r"requested cap $0.20$")
    axb.set_xlabel(r"$\Delta t / \tau_0$ at $\Delta x=W_0$")
    axb.set_ylabel(r"late $V d_0/D$")
    axb.grid(True, alpha=0.25)
    axb.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Karma 2001 isothermal runs to digitized Figs. 1–2."
    )
    parser.add_argument("roots", nargs="+", help="result directories with tip_history.tsv")
    parser.add_argument(
        "--out-dir",
        default="",
        help="figure output directory (default: <first-root>/../figures)",
    )
    parser.add_argument(
        "--dt-scan",
        nargs="*",
        default=None,
        help="extra run dirs for a Δt-dependence figure (45° dx=W0 family)",
    )
    parser.add_argument(
        "--dx-scan",
        nargs="*",
        default=None,
        help="run dirs for a d0/W=0.277 [100] Δx pinning figure",
    )
    args = parser.parse_args()
    runs = [Path(r).resolve() for r in args.roots]
    for root in runs:
        if not pf.history_path(root).exists():
            raise SystemExit(f"missing tip_history.tsv in {root}")
    runs = [r for r in runs if _run_finite(r)]
    if not runs:
        raise SystemExit("no finite-history runs to plot")
    out_dir = Path(args.out_dir) if args.out_dir else runs[0].parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_velocity(runs, out_dir / "fig1_tip_velocity.png")
    print(f"wrote {out_dir / 'fig1_tip_velocity.png'}")
    plot_concentration(runs, out_dir / "fig2_cs_growth_ray.png")
    print(f"wrote {out_dir / 'fig2_cs_growth_ray.png'}")
    write_metrics(runs, out_dir / "karma2001_metrics.tsv")
    if args.dx_scan:
        dx_runs = [Path(r).resolve() for r in args.dx_scan]
        for root in dx_runs:
            if not pf.history_path(root).exists():
                raise SystemExit(f"missing tip_history.tsv in {root}")
        dx_runs = [r for r in dx_runs if _run_finite(r)]
        if not dx_runs:
            raise SystemExit("no finite-history Δx-scan runs to plot")
        plot_dx_pinning(dx_runs, out_dir / "fig_dx_pinning.png")
        print(f"wrote {out_dir / 'fig_dx_pinning.png'}")
        write_metrics(dx_runs, out_dir / "karma2001_dx_pinning.tsv")
    if args.dt_scan:
        dt_runs = [Path(r).resolve() for r in args.dt_scan]
        for root in dt_runs:
            if not pf.history_path(root).exists():
                raise SystemExit(f"missing tip_history.tsv in {root}")
        dt_runs = [r for r in dt_runs if _run_finite(r)]
        if not dt_runs:
            raise SystemExit("no finite-history Δt-scan runs to plot")
        plot_dt_scan(dt_runs, out_dir / "fig_dt_scan_th45.png")
        print(f"wrote {out_dir / 'fig_dt_scan_th45.png'}")
        write_metrics(dt_runs, out_dir / "karma2001_dt_scan.tsv")


if __name__ == "__main__":
    main()
