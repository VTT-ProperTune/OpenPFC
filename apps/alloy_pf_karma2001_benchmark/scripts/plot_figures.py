#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Karma 2001 isothermal dendrite plots: W0-convergence of trapping kinetics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG1_PAPER = DATA_DIR / "karma2001_fig1_present.tsv"
FIG2_PAPER = DATA_DIR / "karma2001_fig2_present.tsv"


def load_meta(path: Path) -> dict[str, float | str]:
    meta: dict[str, float | str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, val = parts
        try:
            meta[key] = float(val)
        except ValueError:
            meta[key] = val
    return meta


def load_history(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def load_axis(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def load_paper_xy(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def meta_path(root: Path) -> Path:
    for name in ("meta.txt", "meta.txt"):
        p = root / name
        if p.exists():
            return p
    return root / "meta.txt"


def history_path(root: Path) -> Path:
    for name in ("tip_history.tsv", "tip_history.tsv"):
        p = root / name
        if p.exists():
            return p
    return root / "tip_history.tsv"


def is_notrap(meta: dict[str, float | str]) -> bool:
    a = float(meta.get("A_trap", 0.0))
    b = float(meta.get("beta0", 0.0))
    vd = float(meta.get("VD_pf", meta.get("VD", 0.0)))
    return a <= 1.0e-12 and b <= 1.0e-12 and vd <= 1.0e-12


def run_label(root: Path) -> str:
    meta = load_meta(meta_path(root))
    w_nm = float(meta.get("W0_nm", float(meta.get("W0", 0.0)) * 1.0e9))
    tdot = float(meta.get("Tdot", meta.get("Tdot", 0.0)))
    if tdot > 0.0:
        trap = ", no trap" if is_notrap(meta) else ""
        return rf"$W_0={w_nm:.0f}\,\mathrm{{nm}}${trap}"
    trap = "no trap" if is_notrap(meta) else (
        rf"$A={float(meta.get('A_trap', 0.0)):.2f}$, "
        rf"$\beta_0={float(meta.get('beta0', 0.0)):g}\,\mathrm{{s/m}}$"
    )
    extra = ""
    dtt = float(meta.get("dt_over_tau", 0.0))
    if not (dtt > 0.0):
        tau0 = float(meta.get("tau0", 0.0))
        w0 = float(meta.get("W0", 0.0))
        dx = float(meta.get("dx", 0.0))
        dt = float(meta.get("dt", 0.0))
        if tau0 > 0.0 and w0 > 0.0 and dx > 0.0:
            dtt = dt / (tau0 * (dx / w0))
    if dtt > 0.0 and abs(dtt - 0.02) > 1.0e-9:
        extra = rf", $\Delta t={dtt:g}\,\tau_0$"
    return rf"$W_0={w_nm:.1f}\,\mathrm{{nm}}$, {trap}{extra}"


def _sci_tex(x: float) -> str:
    if not (x > 0.0) or not np.isfinite(x):
        return rf"{x:g}"
    exp = int(np.floor(np.log10(x) + 1.0e-12))
    mant = x / (10.0 ** exp)
    if abs(mant - 1.0) < 1.0e-6:
        return rf"10^{{{exp}}}"
    return rf"{mant:.2g}\times 10^{{{exp}}}"


def am_cooling_title(meta: dict[str, float | str]) -> str:
    """One-line protocol: exponential T-dot(t) and saturating Delta T_cool."""
    tdot = float(meta.get("Tdot", 0.0))
    tau = float(meta.get("t_decay", 0.0))
    if tau > 0.0 and tdot > 0.0:
        dTsat = tdot * tau
        return (
            rf"AM: $\dot T(t)=\dot T_0 e^{{-t/\tau}}$, "
            rf"$\dot T_0={_sci_tex(tdot)}\,\mathrm{{K/s}}$, "
            rf"$\tau={tau * 1e6:.0f}\,\mu\mathrm{{s}}$ "
            rf"$(\Delta T_\mathrm{{cool}}=\dot T_0\tau(1-e^{{-t/\tau}})\to {dTsat:.0f}\,\mathrm{{K}})$; "
            rf"Glasner + Ji, Neumann, $\Omega=0$"
        )
    if tdot > 0.0:
        return (
            rf"AM: linear $\dot T={_sci_tex(tdot)}\,\mathrm{{K/s}}$ "
            rf"(Glasner + Ji, Neumann, $\Omega=0$)"
        )
    return r"AM cooling (Glasner + Ji, Neumann, $\Omega=0$)"


def hist_cols(hist: np.ndarray) -> dict[str, np.ndarray]:
    """Support both the original 7-column history and the trapping SI dump."""
    if hist.ndim != 2:
        raise ValueError("history must be 2-D")
    n = hist.shape[1]
    if n >= 16:
        cols = {
            "t": hist[:, 0],
            "t_star": hist[:, 1],
            "t_us": hist[:, 2],
            "r": hist[:, 3],
            "V": hist[:, 6],
            "V_star": hist[:, 7],
            "V_mps": hist[:, 8] if n >= 9 else hist[:, 6],
            "rho": hist[:, 9],
            "k_cgm": hist[:, 12],
            "k_eff": hist[:, 13],
            "dT_k": hist[:, 14],
            "dT_r": hist[:, 15],
            "dT_th": hist[:, 16],
        }
        if n >= 19:
            cols["c_wall"] = hist[:, 17]
            cols["c_wall_over"] = hist[:, 18]
        if n >= 20:
            cols["liquid_frac"] = hist[:, 19]
        if n >= 26:
            cols["c_s"] = hist[:, 22]
            cols["c_l"] = hist[:, 23]
            cols["dT_c"] = hist[:, 24]
            cols["dT_tip"] = hist[:, 25]
        elif "dT_th" in cols:
            cols["dT_c"] = cols["dT_th"] - cols["dT_r"] - cols["dT_k"]
            cols["dT_tip"] = cols["dT_th"]
        return cols
    return {
        "t": hist[:, 0],
        "t_star": hist[:, 1],
        "t_us": hist[:, 0] * 1.0e6,
        "r": hist[:, 2],
        "V": hist[:, 3],
        "V_star": hist[:, 4],
        "rho": hist[:, 5],
        "k_cgm": np.full(hist.shape[0], np.nan),
        "k_eff": np.full(hist.shape[0], np.nan),
    }


def profile_cols(prof: np.ndarray) -> dict[str, np.ndarray]:
    if prof.shape[1] >= 7:
        return {
            "r": prof[:, 0],
            "r_over_d0": prof[:, 1],
            "r_um": prof[:, 2],
            "phi": prof[:, 3],
            "c_over": prof[:, 5],
            "c_atpct": prof[:, 6],
        }
    return {
        "r": prof[:, 0],
        "r_over_d0": prof[:, 1],
        "r_um": prof[:, 0] * 1.0e6,
        "phi": prof[:, 2],
        "c_over": prof[:, 4],
        "c_atpct": prof[:, 4],
    }


def profile_path(root: Path) -> Path:
    ray = root / "ray_profile.tsv"
    return ray if ray.exists() else root / "axis_profile.tsv"


def ls_velocity(t: np.ndarray, r: np.ndarray, window: int = 40) -> np.ndarray:
    v = np.full_like(t, np.nan, dtype=float)
    for i in range(len(t)):
        i0 = max(0, i + 1 - window)
        tt = t[i0 : i + 1]
        rr = r[i0 : i + 1]
        n = float(len(tt))
        if n < 3.0:
            continue
        st = float(tt.sum())
        sr = float(rr.sum())
        stt = float((tt * tt).sum())
        str_ = float((tt * rr).sum())
        den = n * stt - st * st
        if abs(den) < 1.0e-30:
            continue
        v[i] = (n * str_ - st * sr) / den
    return v


def two_point_velocity(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    v = np.full_like(t, np.nan, dtype=float)
    v[1:] = np.diff(r) / np.maximum(np.diff(t), 1.0e-30)
    return v


V_MIN_POINTS = 10
V_MIN_TSTAR = 50.0


def rolling_ls_slope(
    t: np.ndarray, r: np.ndarray, min_points: int = V_MIN_POINTS, min_dt: float = 0.0
) -> np.ndarray:
    """Least-squares dr/dt. Window is ≥ min_points samples and ≥ min_dt in time."""
    n = len(t)
    v = np.full(n, np.nan)
    left = 0
    for i in range(n):
        while left < i:
            nxt = left + 1
            if (i - nxt + 1) >= min_points and (t[i] - t[nxt]) >= min_dt:
                left = nxt
            else:
                break
        m = float(i - left + 1)
        if m < 2.0:
            continue
        tt = t[left : i + 1]
        rr = r[left : i + 1]
        st = float(tt.sum())
        sr = float(rr.sum())
        stt = float((tt * tt).sum())
        str_ = float((tt * rr).sum())
        den = m * stt - st * st
        if abs(den) < 1.0e-30:
            continue
        v[i] = (m * str_ - st * sr) / den
    return v


def smooth_hist_velocity(hc: dict[str, np.ndarray], meta: dict[str, float | str]) -> dict[str, np.ndarray]:
    """LS dr/dt: ≥10 samples, Δt*≥50, and ≥8 Δx of tip travel so grid jitter averages out."""
    d0 = float(meta["d0"])
    D = float(meta["D"])
    dx = float(meta.get("dx", d0))
    t = hc["t"]
    r = hc["r"]
    n = len(t)
    i0 = max(0, n // 2)
    v_g = abs(float(r[-1] - r[i0]) / max(float(t[-1] - t[i0]), 1.0e-30))
    v_g = max(v_g, 1.0e-8)
    min_dt = max(V_MIN_TSTAR * d0 * d0 / D, 8.0 * dx / v_g)
    v = rolling_ls_slope(t, r, V_MIN_POINTS, min_dt)
    out = dict(hc)
    out["V"] = v
    out["V_mps"] = v
    out["V_star"] = v * d0 / D
    dT0 = float(meta.get("dT_scale", 0.0))
    beta0 = float(meta.get("beta0", 0.0))
    out["dT_k"] = dT0 * beta0 * v
    if "dT_c" in out and "dT_r" in out:
        out["dT_tip"] = out["dT_c"] + out["dT_r"] + out["dT_k"]
    if dT0 > 0.0:
        out["Delta_c"] = out["dT_c"] / dT0 if "dT_c" in out else np.full_like(v, np.nan)
        out["Delta_r"] = out["dT_r"] / dT0 if "dT_r" in out else np.full_like(v, np.nan)
        out["Delta_k"] = out["dT_k"] / dT0
        out["Delta_tip"] = out["dT_tip"] / dT0 if "dT_tip" in out else out["Delta_k"]
        if "dT_th" in out:
            out["Delta_th"] = out["dT_th"] / dT0
    return out


def plot_fig1_compare(runs: list[Path], out: Path) -> None:
    fig, (ax, axz) = plt.subplots(
        2, 1, figsize=(6.6, 7.2), gridspec_kw={"height_ratios": [1.35, 1.0]}
    )
    paper = load_paper_xy(FIG1_PAPER) if FIG1_PAPER.exists() else None
    series = []
    for root in runs:
        hist = load_history(history_path(root))
        meta = load_meta(meta_path(root))
        d0 = float(meta["d0"])
        D = float(meta["D"])
        cols = smooth_hist_velocity(hist_cols(hist), meta)
        series.append(
            (float(meta.get("d0_over_W", meta.get("dx", 1.0))), run_label(root), cols["t_star"], cols["V_star"])
        )
    series.sort(key=lambda s: s[0])

    def draw(ax_i, xlim, ylim, ylabel: bool) -> None:
        if paper is not None:
            ax_i.plot(paper[:, 0], paper[:, 1], color="k", lw=3.6, zorder=1)
        for _, label, t_star, v_star in series:
            ax_i.plot(t_star, v_star, lw=1.45, label=label, zorder=2)
        if paper is not None:
            ax_i.plot(
                paper[:, 0],
                paper[:, 1],
                color="k",
                lw=2.4,
                label="Karma 2001 present (digitized)",
                zorder=5,
            )
        ax_i.set_xlim(*xlim)
        ax_i.set_ylim(*ylim)
        ax_i.set_ylabel(r"$V d_0 / D$")
        ax_i.grid(True, alpha=0.25)
        if ylabel:
            ax_i.legend(frameon=False, fontsize=8, loc="upper right")

    draw(ax, (0.0, 10000.0), (0.0, 0.03), True)
    ax.set_title(r"Tip speed vs time (Glasner + Ji, trapping + $\beta_0$)")
    draw(axz, (4000.0, 10000.0), (0.0, 0.03), False)
    axz.set_xlabel(r"$t D / d_0^2$")
    axz.set_title("Late-time zoom")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_fig2_compare(runs: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    if FIG2_PAPER.exists():
        paper = load_paper_xy(FIG2_PAPER)
        ax.plot(paper[:, 0], paper[:, 1], color="k", lw=3.6, zorder=1)
        ax.plot(
            paper[:, 0],
            paper[:, 1],
            color="k",
            lw=2.4,
            label="Karma 2001 present (digitized)",
            zorder=5,
        )
    gt_drawn = False
    runs_sorted = sorted(runs, key=lambda r: float(load_meta(r / "meta.txt").get("dx", 1.0)))
    for root in runs_sorted:
        meta = load_meta(root / "meta.txt")
        prof = load_axis(profile_path(root))
        pc = profile_cols(prof)
        x_over = pc["r_over_d0"]
        phi = pc["phi"]
        c_over = pc["c_over"]
        hist = load_history(root / "tip_history.tsv")
        hc = hist_cols(hist)
        d0 = float(meta["d0"])
        r_tip = float(hc["r"][-1]) / d0
        solid = (phi > 0.95) & (x_over < r_tip - 12.0)
        ax.plot(x_over[solid], c_over[solid], lw=1.5, label=run_label(root), zorder=2)
        rho = hc["rho"][-1]
        k = float(meta["k"])
        if (not gt_drawn) and np.isfinite(rho) and rho > 0.0:
            gt = k * (1.0 - (1.0 - k) * d0 / rho)
            ax.axhline(
                gt,
                ls="--",
                color="0.35",
                lw=1.1,
                label=rf"Gibbs–Thomson ${gt:.3f}$",
                zorder=3,
            )
            gt_drawn = True
    ax.set_xlabel(r"$x / d_0$")
    ax.set_ylabel(r"$c_s / c_l^0$")
    ax.set_title(r"Fig. 2: $d_0/W=0.277$, Glasner, $\theta_0=45^\circ$")
    ax.set_xlim(0.0, 400.0)
    ax.set_ylim(0.0, 0.4)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_velocity_estimator(root: Path, out: Path) -> None:
    """Diagnose whether V* ripples are physical or from differentiating r_tip."""
    hist = load_history(root / "tip_history.tsv")
    meta = load_meta(root / "meta.txt")
    d0 = float(meta["d0"])
    D = float(meta["D"])
    hc = hist_cols(hist)
    t, t_star, r = hc["t"], hc["t_star"], hc["r"]
    v2 = two_point_velocity(t, r) * d0 / D
    v40 = ls_velocity(t, r, 40) * d0 / D
    v80 = ls_velocity(t, r, 80) * d0 / D
    late = t_star > 4000.0
    if not np.any(late):
        late = np.ones_like(t_star, dtype=bool)
    # Residual after a linear trend in r(t) on the late window
    tt, rr = t[late], r[late]
    p = np.polyfit(tt, rr, 1)
    r_lin = np.polyval(p, tt)
    resid = rr - r_lin

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=False)
    ax = axes[0]
    ax.plot(t_star, v2, lw=0.8, alpha=0.7, label="two-point $r_\\mathrm{tip}$")
    ax.plot(t_star, v40, lw=1.6, label="LS window 40")
    ax.plot(t_star, v80, lw=1.6, label="LS window 80")
    if FIG1_PAPER.exists():
        paper = load_paper_xy(FIG1_PAPER)
        ax.plot(paper[:, 0], paper[:, 1], color="k", lw=2.2, zorder=5, label="Karma 2001 present")
    ax.set_xlim(4000, 10000)
    ax.set_ylim(0.0, 0.02)
    ax.set_ylabel(r"$V d_0 / D$")
    ax.set_title(r"Tip-speed estimator (Glasner $45^\circ$, late time)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.plot(t_star[late], resid, lw=1.0)
    ax2.set_xlabel(r"$t D / d_0^2$")
    ax2.set_ylabel(r"$r_\mathrm{tip}$ minus linear trend")
    ax2.set_title(r"Position residual (physical if structured; staircasing if $\sim\Delta x$ noise)")
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    dx = float(meta.get("dx", 1.0))
    print(
        f"estimator {root}: two-pt std={np.nanstd(v2[late]):.5f}  "
        f"LS40 std={np.nanstd(v40[late]):.5f}  LS80 std={np.nanstd(v80[late]):.5f}  "
        f"r residual rms={np.sqrt(np.mean(resid**2)):.4f}  dx={dx:g}"
    )


def _late_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    finite = [v[np.isfinite(v)] for v in values if v.size]
    if not finite:
        return (0.0, 1.0)
    stacked = np.concatenate(finite)
    lo = float(np.min(stacked))
    hi = float(np.max(stacked))
    span = max(hi - lo, 0.08 * max(abs(hi), 1.0e-12))
    pad = 0.15 * span
    return (max(0.0, lo - pad), hi + pad)


def load_contour_segments(path: Path) -> dict[float, list[tuple[float, float, float, float]]]:
    """Map t (μs) → marching-squares φ=0 segments (x,y in μm)."""
    by_t: dict[float, list[tuple[float, float, float, float]]] = {}
    pending: list[tuple[float, float]] = []
    cur_t: float | None = None

    def flush() -> None:
        nonlocal pending
        if len(pending) >= 2 and cur_t is not None:
            by_t.setdefault(cur_t, []).append(
                (pending[0][0], pending[0][1], pending[1][0], pending[1][1])
            )
        pending = []

    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            flush()
            continue
        a = line.split()
        if len(a) < 4:
            continue
        t_us, x, y = float(a[0]), float(a[2]), float(a[3])
        if cur_t is not None and abs(t_us - cur_t) > 1.0e-12:
            flush()
        cur_t = t_us
        pending.append((x, y))
        if len(pending) == 2:
            flush()
    flush()
    return by_t


def _pt_close(a: tuple[float, float], b: tuple[float, float], atol: float) -> bool:
    return abs(a[0] - b[0]) <= atol and abs(a[1] - b[1]) <= atol


def stitch_segment_polylines(
    segs: list[tuple[float, float, float, float]], atol: float = 1.0e-8
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Join cell-edge segments into continuous polylines so dash patterns remain visible."""
    remaining = [[(xa, ya), (xb, yb)] for xa, ya, xb, yb in segs]
    chains: list[list[tuple[float, float]]] = []
    while remaining:
        chain = remaining.pop()
        grown = True
        while grown:
            grown = False
            for i, seg in enumerate(remaining):
                p0, p1 = seg
                if _pt_close(chain[-1], p0, atol):
                    chain.append(p1)
                elif _pt_close(chain[-1], p1, atol):
                    chain.append(p0)
                elif _pt_close(chain[0], p1, atol):
                    chain.insert(0, p0)
                elif _pt_close(chain[0], p0, atol):
                    chain.insert(0, p1)
                else:
                    continue
                remaining.pop(i)
                grown = True
                break
        chains.append(chain)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for chain in chains:
        out.append((np.asarray([p[0] for p in chain]), np.asarray([p[1] for p in chain])))
    return out


def late_contour_polylines(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    by_t = load_contour_segments(path)
    if not by_t:
        return []
    segs = by_t[max(by_t)]
    if not segs:
        return []
    span = max((max(abs(xa), abs(ya), abs(xb), abs(yb)) for xa, ya, xb, yb in segs), default=1.0)
    return stitch_segment_polylines(segs, atol=max(1.0e-8, 1.0e-6 * span))


# Abort is stop_frac=0.80 of L. Clip earlier so V / isolines are not the
# squeezed-corner artifact after the envelope reaches the far Neumann walls.
AM_ENVELOPE_CLIP_FRAC = 0.70


def contour_dump_extent_um(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """t_us and max(|x|, |y|) of φ=0 at each dump (μm)."""
    by_t: dict[float, float] = {}
    if not path.exists() or path.stat().st_size <= 0:
        return np.array([]), np.array([])
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        a = line.split()
        if len(a) < 4:
            continue
        t_us, x, y = float(a[0]), float(a[2]), float(a[3])
        ext = max(abs(x), abs(y))
        prev = by_t.get(t_us)
        by_t[t_us] = ext if prev is None else max(prev, ext)
    if not by_t:
        return np.array([]), np.array([])
    ts = np.array(sorted(by_t))
    return ts, np.array([by_t[t] for t in ts])


def am_envelope_keep(hc: dict[str, np.ndarray], meta: dict[str, float | str], root: Path) -> np.ndarray:
    """True while the φ=0 envelope is inside AM_ENVELOPE_CLIP_FRAC of L."""
    t = hc["t_us"]
    keep = np.ones(len(t), dtype=bool)
    L_um = float(meta.get("L", 0.0)) * 1.0e6
    if not (L_um > 0.0):
        return keep
    clip = AM_ENVELOPE_CLIP_FRAC * L_um
    tt, ext = contour_dump_extent_um(root / "interface_contours.tsv")
    if tt.size >= 2:
        extent = np.interp(t, tt, ext, left=float(ext[0]), right=float(ext[-1]))
        keep &= extent < clip
    elif tt.size == 1:
        keep &= ext[0] < clip
    elif "liquid_frac" in hc:
        R = L_um * np.sqrt(np.maximum(4.0 * (1.0 - hc["liquid_frac"]) / np.pi, 0.0))
        keep &= R < clip
    else:
        keep &= hc["r"] * 1.0e6 < clip
    return keep


def last_interior_contour_polylines(path: Path, L_um: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Last φ=0 dump whose axis extent is still inside the clip fraction of L."""
    by_t = load_contour_segments(path)
    if not by_t or not (L_um > 0.0):
        return late_contour_polylines(path)
    clip = AM_ENVELOPE_CLIP_FRAC * L_um
    chosen = None
    for t in sorted(by_t):
        segs = by_t[t]
        if not segs:
            continue
        ext = max(max(abs(xa), abs(ya), abs(xb), abs(yb)) for xa, ya, xb, yb in segs)
        if ext < clip:
            chosen = segs
    if chosen is None:
        return []
    span = max((max(abs(xa), abs(ya), abs(xb), abs(yb)) for xa, ya, xb, yb in chosen), default=1.0)
    return stitch_segment_polylines(chosen, atol=max(1.0e-8, 1.0e-6 * span))


def plot_trapping_convergence(runs: list[Path], out: Path) -> None:
    """W0-convergence of tip speed, trapping, and (isothermal) SI V(t)."""
    metas = [load_meta(meta_path(r)) for r in runs]
    am = any(float(m.get("Tdot", m.get("Tdot", 0.0))) > 0.0 for m in metas)
    t_mid_star = 1000.0
    t_cut_star = 4000.0
    if am:
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2), layout="constrained")
        ax_vs, ax_v, ax_wall = axes[0]
        ax_c, ax_iso, ax_k = axes[1]
        ax_mid = ax_late = ax_dt = ax_r = None
    else:
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2), layout="constrained")
        ax_vs, ax_mid, ax_late = axes[0]
        ax_c, ax_iso, ax_dt = axes[1]
        ax_v = ax_wall = ax_r = ax_k = None
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            float(load_meta(meta_path(r)).get("W0", 0.0)),
            0 if is_notrap(load_meta(meta_path(r))) else 1,
            float(load_meta(meta_path(r)).get("noise_F0", 0.0)),
        ),
        reverse=False,
    )
    w_seen: dict[float, str] = {}
    t_max = 0.0
    late_v: list[np.ndarray] = []
    mid_v: list[np.ndarray] = []
    am_kept: list[tuple[np.ndarray, np.ndarray]] = []
    d0 = float(metas[0].get("d0", 12.17e-9))
    D = float(metas[0].get("D", 4.4e-9))
    for root in runs_sorted:
        meta = load_meta(meta_path(root))
        label = run_label(root)
        w_nm = float(meta.get("W0_nm", float(meta.get("W0", 0.0)) * 1.0e9))
        if w_nm not in w_seen:
            w_seen[w_nm] = f"C{len(w_seen) % 10}"
        color = w_seen[w_nm]
        ls = "--" if is_notrap(meta) else "-"
        hc = smooth_hist_velocity(hist_cols(load_history(history_path(root))), meta)
        keep = am_envelope_keep(hc, meta, root) if am else np.ones(len(hc["t"]), dtype=bool)
        tplot = hc["t_us"] if am else hc["t_star"]
        t_max = max(t_max, float(tplot[keep][-1]) if np.any(keep) else float(tplot[-1]))
        vplot = hc["V"] if am else hc["V_star"]
        t_draw = tplot[keep]
        v_draw = vplot[keep]
        ax_vs.plot(t_draw, v_draw, lw=1.6, ls=ls, color=color, label=label)
        if am:
            ax_v.plot(t_draw, v_draw, lw=1.6, ls=ls, color=color, label=label)
            if np.any(keep):
                am_kept.append((t_draw, v_draw))
            if np.any(~keep):
                t_clip = float(tplot[~keep][0])
                ax_vs.axvline(t_clip, color=color, lw=0.9, ls=":", alpha=0.8)
                ax_v.axvline(t_clip, color=color, lw=0.9, ls=":", alpha=0.8)
            cpath = root / "interface_contours.tsv"
            L_um = float(meta.get("L", 0.0)) * 1.0e6
            if cpath.exists() and cpath.stat().st_size > 0:
                polys = last_interior_contour_polylines(cpath, L_um)
                ls_iso: str | tuple = (0, (5.5, 2.8)) if is_notrap(meta) else "-"
                z_iso = 3.5 if is_notrap(meta) else 2.5
                for xs, ys in polys:
                    ax_iso.plot(
                        xs, ys, lw=1.55, ls=ls_iso, color=color, zorder=z_iso, solid_capstyle="round"
                    )
        else:
            v_si = hc.get("V_mps", hc["V"])
            t_si = hc["t"]
            m_mid = tplot >= t_mid_star
            m_late = tplot >= t_cut_star
            ax_mid.plot(t_si[m_mid], v_si[m_mid], lw=1.6, ls=ls, color=color, label=label)
            ax_late.plot(t_si[m_late], v_si[m_late], lw=1.6, ls=ls, color=color, label=label)
            mid_v.append(v_si[m_mid] if np.any(m_mid) else v_si[-max(5, len(v_si) // 5) :])
            late_v.append(v_si[m_late] if np.any(m_late) else v_si[-max(5, len(v_si) // 5) :])
            if "Delta_tip" in hc:
                ax_dt.plot(tplot, hc["Delta_tip"], lw=1.6, ls=ls, color=color)
                ax_dt.plot(tplot, hc["Delta_c"], lw=1.0, ls="-.", color=color, alpha=0.85)
                ax_dt.plot(tplot, hc["Delta_r"], lw=1.0, ls=":", color=color, alpha=0.85)
                ax_dt.plot(tplot, hc["Delta_k"], lw=1.0, ls=(0, (4, 1, 1, 1)), color=color, alpha=0.9)
            elif "dT_k" in hc:
                ax_dt.plot(tplot, hc["dT_k"], lw=1.5, ls=ls, color=color)
            if "dT_tip" in hc and "Delta_tip" not in hc:
                ax_dt.plot(tplot, hc["dT_tip"], lw=1.0, ls=":", color=color)
            cpath = root / "interface_contours.tsv"
            polys = late_contour_polylines(cpath) if cpath.exists() else []
            ls_iso: str | tuple = (0, (5.5, 2.8)) if is_notrap(meta) else "-"
            z_iso = 3.5 if is_notrap(meta) else 2.5
            for xs, ys in polys:
                ax_iso.plot(
                    xs, ys, lw=1.55, ls=ls_iso, color=color, zorder=z_iso, solid_capstyle="round"
                )
        ppath = profile_path(root)
        if ppath.exists() and (not am or str(meta.get("stop_reason", "")) != "wall"):
            pc = profile_cols(load_axis(ppath))
            ax_c.plot(pc["r_over_d0"], pc["c_over"], lw=1.5, ls=ls, color=color, label=label)
        if am and "dT_th" in hc:
            if "Delta_th" in hc:
                ax_k.plot(t_draw, hc["Delta_th"][keep], lw=1.6, ls=ls, color=color, label=label + r" $\Delta_\mathrm{bulk}$")
                if "Delta_c" in hc:
                    ax_k.plot(t_draw, hc["Delta_c"][keep], lw=1.2, ls="-.", color=color, alpha=0.85)
                if "Delta_tip" in hc:
                    ax_k.plot(t_draw, hc["Delta_tip"][keep], lw=1.0, ls=":", color=color)
                if "Delta_k" in hc:
                    ax_k.plot(t_draw, hc["Delta_k"][keep], lw=1.0, ls=(0, (4, 1, 1, 1)), color=color)
            else:
                ax_k.plot(t_draw, hc["dT_th"][keep], lw=1.6, ls=ls, color=color, label=label + r" $\Delta T_\mathrm{bulk}$")
                if "dT_c" in hc:
                    ax_k.plot(t_draw, hc["dT_c"][keep], lw=1.2, ls=ls, color=color, alpha=0.85, label=label + r" $\Delta T_c$")
                if "dT_tip" in hc:
                    ax_k.plot(t_draw, hc["dT_tip"][keep], lw=1.0, ls=":", color=color, label=label + r" $\Delta T_\mathrm{tip}$")
        if am and ax_wall is not None:
            wall = hc.get("c_wall_over")
            if wall is not None:
                ax_wall.plot(t_draw, wall[keep], lw=1.5, ls=ls, color=color, label=label)
        if am and ax_r is not None:
            ax_r.plot(t_draw, hc["r"][keep] * 1.0e6, lw=1.5, ls=ls, color=color, label=label)
    if am:
        t0 = 0.6 * t_max
        for t_k, v_k in am_kept:
            late_v.append(
                v_k[t_k >= t0] if t_k[-1] >= t0 else v_k[-max(5, len(v_k) // 5) :]
            )
    y0, y1 = _late_ylim(late_v)
    if am:
        xlab = r"$t$ ($\mu$s)"
        ax_vs.set_xlim(0.0, max(t_max, 1.0e-9))
        ax_vs.set_ylim(bottom=0.0)
        ax_vs.axhspan(0.05, 0.5, color="0.85", zorder=0)
        ax_v.set_xlim(t0, t_max)
        ax_v.set_ylim(y0, y1)
        ax_v.axhspan(0.05, 0.5, color="0.85", zorder=0)
        clip_pct = int(round(AM_ENVELOPE_CLIP_FRAC * 100))
        ax_v.set_title(
            rf"Late-time zoom ($t>{t0:.1f}\,\mu\mathrm{{s}}$; drop envelope $>{clip_pct}\%\,L$)"
        )
        fig.suptitle(am_cooling_title(metas[0]), fontsize=10)
        ax_vs.set_ylabel(r"$V$ (m/s)")
        ax_v.set_ylabel(r"$V$ (m/s)")
        ax_wall.set_xlabel(xlab)
        ax_wall.set_ylabel(r"$c_\mathrm{wall}/c_l^0$")
        ax_wall.set_title(r"Far-wall $c$ (Neumann; pile-up $\Rightarrow$ BC interaction)")
        ax_wall.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax_wall.grid(True, alpha=0.25)
        ax_wall.legend(frameon=False, fontsize=7)
        ax_iso.set_aspect("equal", adjustable="box")
        ax_iso.set_xlabel(r"$x$ ($\mu$m)")
        ax_iso.set_ylabel(r"$y$ ($\mu$m)")
        ax_iso.set_title(rf"Last $\phi=0$ inside ${clip_pct}\%\,L$")
        ax_iso.grid(True, alpha=0.25)
        xmax = ymax = 0.0
        for ln in ax_iso.lines:
            xd = np.asarray(ln.get_xdata(), dtype=float)
            yd = np.asarray(ln.get_ydata(), dtype=float)
            if xd.size:
                xmax = max(xmax, float(np.nanmax(xd)))
                ymax = max(ymax, float(np.nanmax(yd)))
        lim = max(xmax, ymax, 0.5) * 1.05
        ax_iso.set_xlim(0.0, lim)
        ax_iso.set_ylim(0.0, lim)
        ax_iso.legend(
            handles=[
                Line2D([0], [0], color="k", lw=1.5, ls="-", label="trap"),
                Line2D([0], [0], color="k", lw=1.5, ls=(0, (5.5, 2.8)), label="no trap"),
            ],
            frameon=False,
            fontsize=7,
            loc="upper left",
        )
        if not ax_iso.lines:
            ax_iso.text(0.5, 0.5, r"no $\phi=0$ dumps", ha="center", va="center", transform=ax_iso.transAxes)
        ax_k.set_xlabel(xlab)
        ax_k.set_ylabel(r"$\Delta=(T_L-T)/[|m_l^e|(1-k_e)c_l^0]$")
        ax_k.set_title(r"Undercooling vs $\Omega=0.55$")
        ax_k.axhline(0.55, color="0.25", ls="--", lw=1.0, label=r"Karma $\Omega=0.55$")
        ax_k.set_ylim(bottom=0.0)
        ax_k.legend(frameon=False, fontsize=6)
        ax_k.grid(True, alpha=0.25)
        ax_vs.set_xlabel(xlab)
        ax_vs.set_title("Tip speed (full range)")
        ax_vs.grid(True, alpha=0.25)
        ax_vs.legend(frameon=False, fontsize=6)
        ax_v.set_xlabel(xlab)
        ax_v.grid(True, alpha=0.25)
    else:
        t_unit = d0 * d0 / D
        v_unit = D / d0
        t_si_max = max(t_max, 1.0) * t_unit
        ax_vs.set_xlim(0.0, 10000.0)
        ax_vs.set_ylim(bottom=0.0)
        ax_t = ax_vs.secondary_xaxis(
            "top", functions=(lambda ts: np.asarray(ts) * t_unit, lambda t: np.asarray(t) / t_unit)
        )
        ax_V = ax_vs.secondary_yaxis(
            "right", functions=(lambda vs: np.asarray(vs) * v_unit, lambda v: np.asarray(v) / v_unit)
        )
        ax_t.set_xlabel(r"$t$ (s)")
        ax_V.set_ylabel(r"$V$ (m/s)")
        for ax_si_t in (ax_t, ax_mid, ax_late):
            ax_si_t.ticklabel_format(axis="x", style="sci", scilimits=(-4, -3), useMathText=True)
        ax_vs.set_xlabel(r"$t D_L / d_0^2$")
        ax_vs.set_ylabel(r"$V d_0 / D_L$")
        ax_vs.set_title("Tip speed (full range)")
        ax_vs.grid(True, alpha=0.25)
        ax_vs.legend(frameon=False, fontsize=6)
        y_mid0, y_mid1 = _late_ylim(mid_v)
        ax_mid.set_xlim(t_mid_star * t_unit, t_si_max)
        ax_mid.set_ylim(y_mid0, y_mid1)
        ax_mid.set_xlabel(r"$t$ (s)")
        ax_mid.set_ylabel(r"$V$ (m/s)")
        ax_mid.set_title(r"Mid-time zoom ($t^\ast>1000$)")
        ax_mid.grid(True, alpha=0.25)
        ax_late.set_xlim(t_cut_star * t_unit, t_si_max)
        ax_late.set_ylim(y0, y1)
        ax_late.set_xlabel(r"$t$ (s)")
        ax_late.set_ylabel(r"$V$ (m/s)")
        ax_late.set_title(r"Late-time zoom ($t^\ast>4000$)")
        ax_late.grid(True, alpha=0.25)
        ax_dt.set_xlabel(r"$t D_L / d_0^2$")
        ax_dt.set_ylabel(r"$\Delta=(T_L-T)/[|m_l^e|(1-k_e)c_l^0]$")
        ax_dt.set_title(r"Tip budget vs $\Omega=0.55$")
        ax_dt.axhline(0.55, color="0.2", ls="--", lw=1.1, label=r"Karma $\Omega=0.55$")
        ax_dt.set_ylim(0.0, 0.62)
        ax_dt.grid(True, alpha=0.25)
        term_h = [
            Line2D([0], [0], color="k", lw=1.6, ls="-", label=r"$\Delta_\mathrm{tip}$"),
            Line2D([0], [0], color="k", lw=1.0, ls="-.", label=r"$\Delta_c$"),
            Line2D([0], [0], color="k", lw=1.0, ls=":", label=r"$\Delta_\Gamma=d_0/\rho$"),
            Line2D([0], [0], color="k", lw=1.0, ls=(0, (4, 1, 1, 1)), label=r"$\Delta_k=\beta_0 V$"),
        ]
        ax_dt.legend(handles=term_h, frameon=False, fontsize=6)
        ax_iso.set_aspect("equal", adjustable="box")
        ax_iso.set_xlabel(r"$x$ ($\mu$m)")
        ax_iso.set_ylabel(r"$y$ ($\mu$m)")
        ax_iso.set_title(r"Late-stage $\phi=0$")
        ax_iso.grid(True, alpha=0.25)
        ax_iso.legend(
            handles=[
                Line2D([0], [0], color="k", lw=1.5, ls="-", label="trap"),
                Line2D([0], [0], color="k", lw=1.5, ls=(0, (5.5, 2.8)), label="no trap"),
            ],
            frameon=False,
            fontsize=7,
            loc="upper left",
        )
        if not ax_iso.lines:
            ax_iso.text(0.5, 0.5, r"no $\phi=0$ dumps", ha="center", va="center", transform=ax_iso.transAxes)
        fig.suptitle(r"Trapping vs no-trapping $W_0$ (Glasner + Ji $\bar S_{2,1}$)")
    ax_c.set_xlabel(r"$r / d_0$ along [100]")
    ax_c.set_ylabel(r"$c / c_l^0$")
    ax_c.set_title("Concentration along [100]")
    ax_c.grid(True, alpha=0.25)
    c_inf_over = None
    for root in runs_sorted:
        meta = load_meta(meta_path(root))
        cl0 = float(meta.get("cl0", 1.0))
        c_inf = float(meta.get("c_inf", np.nan))
        if cl0 > 0.0 and np.isfinite(c_inf):
            c_inf_over = c_inf / cl0
            break
    if c_inf_over is not None:
        ax_c.axhline(c_inf_over, color="0.4", lw=0.8, ls="--", label=r"$c_\infty/c_l^0$")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_am_diagnostics(runs: list[Path], out: Path) -> None:
    """Velocity, [100] concentration, tip undercooling, and far-wall solute."""
    fig, (ax_v, ax_c, ax_dt, ax_w) = plt.subplots(1, 4, figsize=(16.4, 4.1))
    w_seen: dict[float, str] = {}
    for root in sorted(runs, key=lambda r: float(load_meta(meta_path(r)).get("W0", 0.0)), reverse=True):
        meta = load_meta(meta_path(root))
        label = run_label(root)
        w_nm = float(meta.get("W0_nm", float(meta.get("W0", 0.0)) * 1.0e9))
        if w_nm not in w_seen:
            w_seen[w_nm] = f"C{len(w_seen) % 10}"
        color = w_seen[w_nm]
        hc = smooth_hist_velocity(hist_cols(load_history(history_path(root))), meta)
        keep = am_envelope_keep(hc, meta, root)
        t = hc["t_us"][keep]
        v = hc.get("V_mps", hc["V"])[keep]
        ax_v.plot(t, v, lw=1.6, color=color, label=label)
        ppath = profile_path(root)
        if ppath.exists() and str(meta.get("stop_reason", "")) != "wall":
            pc = profile_cols(load_axis(ppath))
            ax_c.plot(pc["r_over_d0"], pc["c_over"], lw=1.5, color=color, label=label)
        if "Delta_th" in hc:
            ax_dt.plot(t, hc["Delta_th"][keep], lw=1.6, color=color, label=label + r" $\Delta_\mathrm{bulk}$")
        if "Delta_tip" in hc:
            ax_dt.plot(t, hc["Delta_tip"][keep], lw=1.1, ls="--", color=color, label=label + r" $\Delta_\mathrm{tip}$")
        if "Delta_c" in hc:
            ax_dt.plot(t, hc["Delta_c"][keep], lw=1.0, ls="-.", color=color, alpha=0.85)
        if "Delta_k" in hc:
            ax_dt.plot(t, hc["Delta_k"][keep], lw=1.0, ls=":", color=color)
        wall = hc.get("c_wall_over")
        if wall is not None:
            ax_w.plot(t, wall[keep], lw=1.5, color=color, label=label)
    ax_v.set_xlabel(r"$t$ ($\mu$s)")
    ax_v.set_ylabel(r"$V$ (m/s)")
    ax_v.set_title(r"Tip speed")
    ax_v.set_ylim(bottom=0.0)
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(frameon=False, fontsize=7)
    ax_c.set_xlabel(r"$r/d_0$ along [100]")
    ax_c.set_ylabel(r"$c/c_l^0$")
    ax_c.set_title(r"Concentration along [100]")
    ax_c.grid(True, alpha=0.25)
    ax_dt.set_xlabel(r"$t$ ($\mu$s)")
    ax_dt.set_ylabel(r"$\Delta=(T_L-T)/[|m_l^e|(1-k_e)c_l^0]$")
    ax_dt.set_title(r"Tip undercooling")
    ax_dt.set_ylim(bottom=0.0)
    ax_dt.axhline(0.55, color="0.3", ls="--", lw=1.0, label=r"Karma $\Omega=0.55$")
    ax_dt.axhline(1.0, color="0.5", ls=":", lw=0.9, label=r"$\Delta=1$")
    ax_dt.grid(True, alpha=0.25)
    ax_dt.legend(frameon=False, fontsize=6)
    ax_w.set_xlabel(r"$t$ ($\mu$s)")
    ax_w.set_ylabel(r"$c_\mathrm{wall}/c_l^0$")
    ax_w.set_title(r"Far-wall $c$ (pile-up $\Rightarrow$ BC)")
    ax_w.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend(frameon=False, fontsize=7)
    fig.suptitle(am_cooling_title(load_meta(meta_path(runs[0]))), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_bc_wall_solute(runs: list[Path], out: Path) -> None:
    """Far-wall c/c_l^0 vs time. A rise above c_∞ means Neumann pile-up."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    am = False
    for root in runs:
        meta = load_meta(meta_path(root))
        am = am or float(meta.get("Tdot", 0.0)) > 0.0
        hc = hist_cols(load_history(history_path(root)))
        wall = hc.get("c_wall_over")
        if wall is None:
            continue
        t = hc["t_us"] if am else hc["t_star"]
        ax.plot(t, wall, lw=1.6, label=run_label(root))
        cl0 = float(meta.get("cl0", 1.0))
        c_inf = float(meta.get("c_inf", np.nan))
        if cl0 > 0.0 and np.isfinite(c_inf):
            ax.axhline(c_inf / cl0, color="0.45", lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel(r"$t$ ($\mu$s)" if am else r"$t D_L / d_0^2$")
    ax.set_ylabel(r"$c_\mathrm{wall}/c_l^0$")
    ax.set_title(r"Far Neumann wall solute (flat at $c_\infty$ $\Rightarrow$ no BC interaction)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_undercooling(runs: list[Path], out: Path) -> None:
    """Imposed bulk ΔT and tip budget ΔT_c + ΔT_r + ΔT_k vs time."""
    nax = max(1, len(runs))
    fig, axes = plt.subplots(1, nax, figsize=(4.2 * nax, 4.0), squeeze=False)
    for ax, root in zip(axes[0], runs):
        meta = load_meta(root / "meta.txt")
        hc = smooth_hist_velocity(hist_cols(load_history(root / "tip_history.tsv")), meta)
        if "dT_th" not in hc:
            ax.set_title(run_label(root) + " (no ΔT)")
            continue
        keep = am_envelope_keep(hc, meta, root)
        t = hc["t_us"][keep]
        if "Delta_th" in hc:
            ax.plot(t, hc["Delta_th"][keep], lw=2.0, color="k", label=r"$\Delta_\mathrm{bulk}$")
            ax.plot(t, hc["Delta_c"][keep], lw=1.5, label=r"$\Delta_c$")
            ax.plot(t, hc["Delta_r"][keep], lw=1.5, label=r"$\Delta_\Gamma$")
            ax.plot(t, hc["Delta_k"][keep], lw=1.5, label=r"$\Delta_k$")
            ax.plot(t, hc["Delta_tip"][keep], lw=1.0, ls="--", color="0.3", label=r"$\Delta_\mathrm{tip}$")
            ax.axhline(0.55, color="0.35", ls="--", lw=1.0, label=r"Karma $\Omega=0.55$")
            ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label=r"$\Delta=1$")
            ax.set_ylabel(r"$\Delta=(T_L-T)/[|m_l^e|(1-k_e)c_l^0]$")
        else:
            ax.plot(t, hc["dT_th"][keep], lw=2.0, color="k", label=r"$\Delta T_\mathrm{bulk}=T_L-T(t)$")
            ax.plot(t, hc["dT_c"][keep], lw=1.5, label=r"$\Delta T_c$")
            ax.plot(t, hc["dT_r"][keep], lw=1.5, label=r"$\Delta T_\Gamma$")
            ax.plot(t, hc["dT_k"][keep], lw=1.5, label=r"$\Delta T_k$")
            ax.set_ylabel(r"$\Delta T$ (K)")
        ax.set_title(run_label(root), fontsize=8)
        ax.set_xlabel(r"$t$ ($\mu$s)")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=6)
    fig.suptitle(r"Tip undercooling budget (spatially uniform $T(t)$)")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_interface_isolines(runs: list[Path], out: Path) -> None:
    """Superimposed φ=0 contours at equally spaced times.

    Every panel uses the same physical (x, y) window (metres, plotted in μm)
    so different W0 are comparable. Limits are the largest box L among the
    runs, not each front's own bounding box.
    """
    have = [r for r in runs if (r / "interface_contours.tsv").exists()]
    if not have:
        return
    n = len(have)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False, sharex=True, sharey=True)
    cmap = plt.cm.viridis
    lim_um = 0.0
    for root in have:
        meta = load_meta(meta_path(root))
        L = float(meta.get("L", 0.0))
        if L > 0.0:
            lim_um = max(lim_um, L * 1.0e6)
    for ax, root in zip(axes[0], have):
        meta = load_meta(meta_path(root))
        L_um = float(meta.get("L", 0.0)) * 1.0e6
        clip = AM_ENVELOPE_CLIP_FRAC * L_um if L_um > 0.0 else None
        tt, ext = contour_dump_extent_um(root / "interface_contours.tsv")
        segs: list[tuple[float, list[tuple[float, float]]]] = []
        cur_t: float | None = None
        pts: list[tuple[float, float]] = []
        times: set[float] = set()
        for line in (root / "interface_contours.tsv").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                if pts and cur_t is not None:
                    segs.append((cur_t, pts))
                    pts = []
                continue
            a = line.split()
            if len(a) < 4:
                continue
            t_us, x, y = float(a[0]), float(a[2]), float(a[3])
            times.add(t_us)
            if cur_t is None:
                cur_t = t_us
            if abs(t_us - cur_t) > 1.0e-12:
                if pts:
                    segs.append((cur_t, pts))
                pts = [(x, y)]
                cur_t = t_us
            else:
                pts.append((x, y))
        if pts and cur_t is not None:
            segs.append((cur_t, pts))
        if clip is not None and tt.size:
            segs = [
                (t_us, xy)
                for t_us, xy in segs
                if ext[int(np.argmin(np.abs(tt - t_us)))] < clip
            ]
        times = {t_us for t_us, _ in segs}
        tlist = sorted(times)
        t0, t1 = (tlist[0], tlist[-1]) if tlist else (0.0, 1.0)
        span = max(t1 - t0, 1.0e-9)
        for t_us, xy in segs:
            col = cmap((t_us - t0) / span)
            ax.plot([p[0] for p in xy], [p[1] for p in xy], color=col, lw=0.8)
        if segs:
            xmax = max(p[0] for _, xy in segs for p in xy)
            ymax = max(p[1] for _, xy in segs for p in xy)
            # do not grow the shared window past L from a clipped dump
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
        ax.set_title(run_label(root), fontsize=8)
        ax.grid(True, alpha=0.2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t0, t1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=r"$t$ ($\mu$s)")
    lim = max(lim_um, 0.5)
    for ax in axes[0]:
        ax.set_xlim(0.0, lim)
        ax.set_ylim(0.0, lim)
    fig.suptitle(
        rf"Solid–liquid interface ($\phi=0$) at equally spaced times "
        rf"(same window; drop dumps with envelope $>{int(round(AM_ENVELOPE_CLIP_FRAC * 100))}\%\,L$)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="result directories")
    parser.add_argument("--out-dir", default="results/alloy_pf_karma2001_benchmark_figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in args.roots]
    runs.sort(
        key=lambda r: float(
            load_meta(meta_path(r)).get("W0", float(load_meta(meta_path(r)).get("d0_over_W", 0.0)))
        )
    )
    am = any(float(load_meta(meta_path(r)).get("Tdot", 0.0)) > 0.0 for r in runs)
    if not am:
        plot_fig1_compare(runs, out_dir / "fig1_compare_present.png")
        plot_fig2_compare(runs, out_dir / "fig2_compare_present.png")
        print(f"wrote {out_dir / 'fig1_compare_present.png'}")
        print(f"wrote {out_dir / 'fig2_compare_present.png'}")
    plot_trapping_convergence(runs, out_dir / "trapping_w0_convergence.png")
    print(f"wrote {out_dir / 'trapping_w0_convergence.png'}")
    plot_bc_wall_solute(runs, out_dir / "bc_wall_solute.png")
    print(f"wrote {out_dir / 'bc_wall_solute.png'}")
    if am:
        plot_am_diagnostics(runs, out_dir / "am_v_c_undercooling.png")
        print(f"wrote {out_dir / 'am_v_c_undercooling.png'}")
        plot_undercooling(runs, out_dir / "am_undercooling.png")
        print(f"wrote {out_dir / 'am_undercooling.png'}")
        plot_interface_isolines(runs, out_dir / "am_interface_isolines.png")
        iso = out_dir / "am_interface_isolines.png"
        if iso.exists():
            print(f"wrote {iso}")
    if not am:
        est_root = runs[0]
        plot_velocity_estimator(est_root, out_dir / "fig1_velocity_estimator.png")
        print(f"wrote {out_dir / 'fig1_velocity_estimator.png'}")


if __name__ == "__main__":
    main()
