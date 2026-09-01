#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lag, Δ(t), and full-box c maps for the G=3e6 K/m, Vp=0.4 m/s FTA series."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTROOT = ROOT / "results" / "alloy_pf_directional_ds" / "G3e6_V0.4_static"
OUTROOT_WIDE = ROOT / "results" / "alloy_pf_directional_ds" / "G3e6_V0.4_Ly3.2"
CASES_NARROW = (
    (r"$W_0=40$ nm", "w0_40nm_dx1"),
    (r"$W_0=20$ nm", "w0_20nm_dx1"),
    (r"$W_0=10$ nm", "w0_10nm_dx1"),
    (r"$W_0=10$ nm, $\Delta x=0.5\,W_0$", "w0_10nm_dx0.5"),
    (r"$W_0=5$ nm", "w0_5nm_dx1"),
    (r"$W_0=2.5$ nm, $0.2\tau_0$", "w0_2.5nm_dx1"),
)
LAG_ONLY_NARROW = (
    (r"$W_0=2.5$ nm, $0.1\tau_0$ (running)", "w0_2.5nm_dx1_dt0.1"),
)
CASES_WIDE = (
    (r"$W_0=20$ nm", "w0_20nm_dx1"),
    (r"$W_0=10$ nm", "w0_10nm_dx1"),
    (r"$W_0=5$ nm", "w0_5nm_dx1"),
)
TIMES_US = (0.0, 24.0, 48.0, 72.0, 96.0, 120.0)


def load_cgm():
    spec = importlib.util.spec_from_file_location(
        "cgm_delta", Path(__file__).with_name("cgm_delta.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_meta(path: Path) -> dict:
    meta: dict = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            meta[parts[0]] = float(parts[1])
        except ValueError:
            meta[parts[0]] = parts[1]
    return meta


def load_run(root: Path) -> dict | None:
    if not (root / "history.tsv").exists() or not (root / "meta.txt").exists():
        return None
    meta = load_meta(root / "meta.txt")
    hist = np.loadtxt(root / "history.tsv", comments="#")
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)
    nx, ny = int(meta["Nx"]), int(meta["Ny"])
    dx = float(meta["dx"])
    out = {"root": root, "meta": meta, "hist": hist, "nx": nx, "ny": ny, "dx": dx}
    if (root / "phi_final.raw").exists():
        out["phi"] = np.fromfile(root / "phi_final.raw", dtype=np.float64).reshape(ny, nx)
    if (root / "c_final.raw").exists():
        out["c"] = np.fromfile(root / "c_final.raw", dtype=np.float64).reshape(ny, nx)
    return out


def lag_and_delta(run: dict, cgm):
    meta, hist = run["meta"], run["hist"]
    t = hist[:, 0]
    xt = hist[:, 2]
    G = float(meta["G"])
    Vp = float(meta["Vp"])
    x_tl = float(meta["x_tl"])
    lag = x_tl + Vp * t - xt
    dlt = cgm.delta_from_xtip(xt, t, x_tl, Vp, G=G)
    c_wall = hist[:, 10] if hist.shape[1] >= 11 else None
    c_wall_max = hist[:, 11] if hist.shape[1] >= 12 else None
    return t, xt, lag, dlt, c_wall, c_wall_max


def plot_lag_delta(runs: list[tuple[str, dict]], cgm, figdir: Path, Vp: float) -> None:
    d_cgm = cgm.delta_eq19(Vp, drag=0.38)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.4), sharex=True)
    ax_lag, ax_d, ax_w = axes
    for label, run in runs:
        t, xt, lag, dlt, c_wall, c_wall_max = lag_and_delta(run, cgm)
        tu = t * 1e6
        ax_lag.plot(tu, lag * 1e6, lw=1.6, label=label)
        ax_d.plot(tu, dlt, lw=1.6, label=label)
        if c_wall is not None:
            ax_w.plot(tu, c_wall, lw=1.4, label=label)
            if c_wall_max is not None:
                ax_w.plot(
                    tu,
                    float(run["meta"]["clo"]) + c_wall_max,
                    lw=0.8,
                    ls=":",
                    color=ax_w.lines[-1].get_color(),
                )
    ax_lag.set_ylabel(r"lag $\delta=x_s+V_p t-x_{\mathrm{tip}}$ (μm)")
    ax_lag.set_title("Pulling-frame lag (QSS if this plateaus)")
    ax_lag.grid(True, alpha=0.25)
    ax_lag.legend(frameon=False, fontsize=8)
    ax_d.axhline(d_cgm, color="k", ls="--", lw=1.0, label=rf"CGM $\Delta(V_p)={d_cgm:.2f}$")
    ax_d.set_ylabel(r"$\Delta(t)$")
    ax_d.set_title("Dimensionless tip undercooling")
    ax_d.grid(True, alpha=0.25)
    ax_d.legend(frameon=False, fontsize=8)
    clo = float(runs[0][1]["meta"]["clo"]) if runs else 4.5
    ax_w.axhline(clo, color="k", ls="--", lw=1.0, label=rf"$c_\infty={clo:g}$")
    ax_w.set_xlabel(r"$t$ (μs)")
    ax_w.set_ylabel(r"right-wall $c$ (at%)")
    ax_w.set_title("Far-liquid wall (mean; dotted = $c_\\infty$ + max |dev|)")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "ds_lag_delta.png", dpi=140)
    plt.close(fig)


def snapshot_steps(root: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in root.glob("output_c_phi_*.png"):
        try:
            out.append((int(p.stem.rsplit("_", 1)[1]), p))
        except ValueError:
            continue
    out.sort()
    return out


def png_c_top(path: Path, ny: int, clo: float) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 3:
        img = np.asarray(img[..., 0], dtype=np.float64)
    else:
        img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    return img[:ny, :] * (2.0 * clo)


def box_um(run: dict) -> tuple[float, float]:
    return run["nx"] * run["dx"] * 1e6, run["ny"] * run["dx"] * 1e6


def nearest_snapshots(run: dict, times_us: tuple[float, ...]) -> list[tuple[float, np.ndarray]]:
    dt = float(run["meta"]["dt"])
    clo = float(run["meta"]["clo"])
    t_end = float(run["meta"].get("t_stop", run["hist"][-1, 0]))
    snaps = snapshot_steps(run["root"])
    if not snaps:
        if "c" in run:
            return [(t_end * 1e6, run["c"])]
        return []
    steps = [s for s, _ in snaps]
    picked: list[tuple[float, np.ndarray]] = []
    used: set[int] = set()
    for tus in times_us:
        t = min(tus * 1e-6, t_end)
        step = min(steps, key=lambda s: abs(s * dt - t))
        if step in used:
            continue
        used.add(step)
        path = dict(snaps)[step]
        picked.append((step * dt * 1e6, png_c_top(path, run["ny"], clo)))
    if "c" in run and t_end * 1e6 - picked[-1][0] > 1.0:
        picked.append((t_end * 1e6, run["c"]))
    return picked


def _equal_ax(ax, lx: float, ly: float) -> None:
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)
    ax.set_aspect("equal", adjustable="box")


def plot_c_maps(runs: list[tuple[str, dict]], figdir: Path) -> None:
    have = [(lab, r) for lab, r in runs if "c" in r]
    if not have:
        return
    clo = float(have[0][1]["meta"]["clo"])
    lx = max(box_um(r)[0] for _, r in have)
    ly = max(box_um(r)[1] for _, r in have)
    n = len(have)
    fig_w = 10.8
    row_h = fig_w * ly / lx + 0.55
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, row_h * n), squeeze=False)
    im = None
    for ax, (label, run) in zip(axes[:, 0], have):
        rlx, rly = box_um(run)
        im = ax.imshow(
            run["c"],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=2.0 * clo,
            extent=(0.0, rlx, 0.0, rly),
            aspect="equal",
        )
        if "phi" in run:
            dx = run["dx"]
            ny, nx = run["phi"].shape
            yy, xx = np.mgrid[0:ny, 0:nx]
            ax.contour(
                (0.5 * dx + xx * dx) * 1e6,
                (0.5 * dx + yy * dx) * 1e6,
                run["phi"],
                levels=[0.0],
                colors="w",
                linewidths=0.5,
            )
        tstop = float(run["meta"].get("t_stop", 0.0)) * 1e6
        _equal_ax(ax, lx, ly)
        ax.set_ylabel("y (μm)")
        ax.set_title(f"{label}  $t={tstop:.1f}$ μs", loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("x (μm)")
    fig.colorbar(im, ax=axes[:, 0], fraction=0.02, pad=0.02, label="c (at%)")
    fig.suptitle("Late $c$ on the stored grid (true aspect; white = φ=0)", fontsize=12)
    fig.savefig(figdir / "ds_c_late.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_phi_maps(runs: list[tuple[str, dict]], figdir: Path) -> None:
    have = [(lab, r) for lab, r in runs if "phi" in r]
    if not have:
        return
    lx = max(box_um(r)[0] for _, r in have)
    ly = max(box_um(r)[1] for _, r in have)
    n = len(have)
    fig_w = 10.8
    row_h = fig_w * ly / lx + 0.55
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, row_h * n), squeeze=False)
    im = None
    for ax, (label, run) in zip(axes[:, 0], have):
        rlx, rly = box_um(run)
        im = ax.imshow(
            run["phi"],
            origin="lower",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            extent=(0.0, rlx, 0.0, rly),
            aspect="equal",
        )
        dx = run["dx"]
        ny, nx = run["phi"].shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        ax.contour(
            (0.5 * dx + xx * dx) * 1e6,
            (0.5 * dx + yy * dx) * 1e6,
            run["phi"],
            levels=[0.0],
            colors="k",
            linewidths=0.4,
        )
        tstop = float(run["meta"].get("t_stop", 0.0)) * 1e6
        _equal_ax(ax, lx, ly)
        ax.set_ylabel("y (μm)")
        ax.set_title(f"{label}  $t={tstop:.1f}$ μs", loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("x (μm)")
    fig.colorbar(im, ax=axes[:, 0], fraction=0.02, pad=0.02, label=r"$\varphi$")
    fig.suptitle(r"Late $\varphi$ (true aspect; black $=\varphi=0$)", fontsize=12)
    fig.savefig(figdir / "ds_phi_late.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_c_lineouts(runs: list[tuple[str, dict]], figdir: Path) -> None:
    have = [(lab, r) for lab, r in runs if "c" in r]
    if not have:
        return
    clo = float(have[0][1]["meta"]["clo"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for label, run in have:
        mid = run["ny"] // 2
        x = (0.5 + np.arange(run["nx"])) * run["dx"] * 1e6
        ax.plot(x, run["c"][mid, :], lw=1.2, label=label)
    ax.axhline(clo, color="k", ls="--", lw=0.8, label=rf"$c_\infty={clo:g}$")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("c (at%) at mid-y")
    ax.set_title("Late axial microsegregation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "ds_c_line.png", dpi=150)
    plt.close(fig)


def plot_front_xy(runs: list[tuple[str, dict]], figdir: Path) -> None:
    have = [(lab, r) for lab, r in runs if "phi" in r]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for label, run in have:
        phi, dx = run["phi"], run["dx"]
        ny, nx = phi.shape
        xs = np.full(ny, np.nan)
        for j in range(ny):
            row = phi[j]
            for i in range(nx - 2, 0, -1):
                p0, p1 = row[i], row[i + 1]
                if p0 * p1 < 0.0:
                    t = p0 / (p0 - p1)
                    xs[j] = (i + 0.5 + t) * dx
                    break
        y = (np.arange(ny) + 0.5) * dx * 1e6
        ax.plot(xs * 1e6, y, lw=1.4, label=label)
    ax.set_xlabel(r"rightmost $\varphi=0$ (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(r"Leading $\varphi=0$ vs $y$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "ds_front_xy.png", dpi=150)
    plt.close(fig)


def plot_c_evolution(
    runs: list[tuple[str, dict]], figdir: Path, clo: float, lx: float, ly: float
) -> None:
    nrun = len(runs)
    frames = [nearest_snapshots(run, TIMES_US) for _, run in runs]
    nt = max(len(f) for f in frames)
    if nt == 0:
        return
    fig_w = 5.4 * nrun
    row_h = (fig_w / nrun) * ly / lx + 0.42
    fig, axes = plt.subplots(nt, nrun, figsize=(fig_w, row_h * nt), squeeze=False)
    im = None
    for col, ((label, run), seq) in enumerate(zip(runs, frames)):
        rlx, rly = box_um(run)
        for row in range(nt):
            ax = axes[row, col]
            if row >= len(seq):
                ax.axis("off")
                continue
            tus, field = seq[row]
            im = ax.imshow(
                field,
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=2.0 * clo,
                extent=(0.0, rlx, 0.0, rly),
                aspect="equal",
            )
            _equal_ax(ax, lx, ly)
            if col == 0:
                ax.set_ylabel(f"$t={tus:.0f}$ μs\ny (μm)", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(label, loc="left", fontsize=10)
            if row == nt - 1:
                ax.set_xlabel("x (μm)")
            else:
                ax.set_xticklabels([])
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="c (at%)")
    fig.suptitle("Full-box $c(x,y)$ vs time (static grid, true aspect)", fontsize=12)
    fig.savefig(figdir / "ds_c_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=("narrow", "wide"), default="narrow")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.series == "wide":
        cases, lag_only = CASES_WIDE, ()
        default_root = OUTROOT_WIDE
    else:
        cases, lag_only = CASES_NARROW, LAG_ONLY_NARROW
        default_root = OUTROOT
    root = args.root if args.root is not None else default_root
    figdir = args.out_dir if args.out_dir is not None else root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cgm = load_cgm()
    runs = []
    for label, name in cases:
        loaded = load_run(root / name)
        if loaded is not None and (root / name / "phi_final.raw").exists():
            runs.append((label, loaded))
    for label, name in lag_only:
        loaded = load_run(root / name)
        if loaded is not None:
            runs.append((label, loaded))
    if not runs:
        print(f"no finished cases under {root}", file=sys.stderr)
        sys.exit(1)
    Vp = float(runs[0][1]["meta"]["Vp"])
    plot_lag_delta(runs, cgm, figdir, Vp)
    plot_c_maps(runs, figdir)
    plot_phi_maps(runs, figdir)
    plot_c_lineouts(runs, figdir)
    plot_front_xy(runs, figdir)
    print(f"wrote figures under {figdir}")


if __name__ == "__main__":
    main()
