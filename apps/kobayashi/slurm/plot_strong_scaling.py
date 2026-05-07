#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strong scaling plot from kobayashi_scaling_<jobid>/summary.tsv (SVG; optional PNG if matplotlib)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Tuple


def read_summary_tsv(path: str) -> Tuple[str, List[int], List[float]]:
    """Returns (first-column header name, x values, wall times)."""
    xs: List[int] = []
    wall: List[float] = []
    with open(path, encoding="utf-8") as f:
        header_line = next(f, None)
        if not header_line:
            raise ValueError(f"empty file: {path}")
        header = header_line.strip().split("\t")
        x_key = header[0] if header else "nproc"
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            xs.append(int(parts[0]))
            wall.append(float(parts[1]))
    if not xs:
        raise ValueError(f"no data rows in {path}")
    pairs = sorted(zip(xs, wall), key=lambda x: x[0])
    return x_key, [p[0] for p in pairs], [p[1] for p in pairs]


def axis_label_from_tsv_key(x_key: str) -> str:
    if x_key.strip() == "nthreads":
        return "OpenMP threads"
    return "MPI ranks"


def linmap(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2.0
    return a + (v - lo) / (hi - lo) * (b - a)


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(
    out_path: str,
    nproc: List[int],
    wall: List[float],
    title: str,
    x_axis_label: str,
) -> None:
    t1 = wall[0]
    speedup = [t1 / w if w > 0 else float("nan") for w in wall]
    ideal = list(nproc)
    eff_pct = [100.0 * speedup[i] / nproc[i] for i in range(len(nproc))]

    W, H = 920, 540
    m = dict(l=72, r=48, t=56, b=52)
    pw = W - m["l"] - m["r"]
    ph = (H - m["t"] - m["b"]) / 2 - 16

    xmin, xmax = math.log10(min(nproc)), math.log10(max(nproc))
    pad = 0.06 * (xmax - xmin)
    xmin -= pad
    xmax += pad

    wmax = max(wall) * 1.08
    wmin = 0.0
    smax = max(max(speedup), max(ideal)) * 1.05
    smin = 0.0
    emax = max(105.0, max(eff_pct) * 1.08)
    emin = 0.0

    def x_px(n: int) -> float:
        return m["l"] + linmap(math.log10(n), xmin, xmax, 0.0, pw)

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">')
    lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="18" font-weight="600">{escape_xml(title)}</text>')

    # --- Panel 1: wall time ---
    y0 = m["t"]
    lines.append(f'<text x="{m["l"]}" y="{y0 - 8}" font-size="13" font-weight="600">Wall time (integration loop)</text>')
    lines.append(
        f'<rect x="{m["l"]}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#ccc" stroke-width="1"/>'
    )

    def y1_wall(t: float) -> float:
        return y0 + linmap(t, wmin, wmax, ph, 0.0)

    # grid + y ticks
    for tick in _nice_ticks(wmin, wmax, 5):
        yy = y1_wall(tick)
        lines.append(f'<line x1="{m["l"]}" y1="{yy}" x2="{m["l"]+pw}" y2="{yy}" stroke="#eee" stroke-width="1"/>')
        lines.append(
            f'<text x="{m["l"]-6}" y="{yy+4}" text-anchor="end" font-size="10" fill="#444">{tick:.1f}s</text>'
        )

    for n in nproc:
        xx = x_px(n)
        lines.append(f'<line x1="{xx}" y1="{y0}" x2="{xx}" y2="{y0+ph}" stroke="#f5f5f5" stroke-width="1"/>')
        lines.append(f'<text x="{xx}" y="{y0+ph+14}" text-anchor="middle" font-size="10" fill="#444">{n}</text>')

    pts = " ".join(f"{x_px(n)},{y1_wall(w)}" for n, w in zip(nproc, wall))
    lines.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{pts}"/>')
    for n, w in zip(nproc, wall):
        lines.append(
            f'<circle cx="{x_px(n)}" cy="{y1_wall(w)}" r="4" fill="#1f77b4" stroke="white" stroke-width="1"/>'
        )
    lines.append(
        f'<text x="{m["l"]+pw/2}" y="{y0+ph+36}" text-anchor="middle" font-size="11" fill="#333">{escape_xml(x_axis_label)} (log scale)</text>'
    )

    # --- Panel 2: speedup + efficiency ---
    y0b = y0 + ph + 44
    lines.append(f'<text x="{m["l"]}" y="{y0b - 8}" font-size="13" font-weight="600">Speedup vs ideal · Parallel efficiency</text>')
    lines.append(
        f'<rect x="{m["l"]}" y="{y0b}" width="{pw}" height="{ph}" fill="none" stroke="#ccc" stroke-width="1"/>'
    )

    def y2_s(s: float) -> float:
        return y0b + linmap(s, smin, smax, ph, 0.0)

    def y2_e(e: float) -> float:
        return y0b + linmap(e, emin, emax, ph, 0.0)

    for tick in _nice_ticks(smin, smax, 5):
        yy = y2_s(tick)
        lines.append(f'<line x1="{m["l"]}" y1="{yy}" x2="{m["l"]+pw}" y2="{yy}" stroke="#eee" stroke-width="1"/>')
        lines.append(
            f'<text x="{m["l"]-6}" y="{yy+4}" text-anchor="end" font-size="10" fill="#2ca02c">{tick:.0f}×</text>'
        )

    second_axis_x = m["l"] + pw + 8
    for tick in _nice_ticks(emin, emax, 5):
        yy = y2_e(tick)
        lines.append(
            f'<text x="{second_axis_x}" y="{yy+4}" font-size="10" fill="#ff7f0e">{tick:.0f}%</text>'
        )

    ideal_line_pts = " ".join(f"{x_px(n)},{y2_s(float(n))}" for n in nproc)
    lines.append(f'<polyline fill="none" stroke="#aaa" stroke-width="2" stroke-dasharray="6 4" points="{ideal_line_pts}"/>')

    su_pts = " ".join(f"{x_px(n)},{y2_s(s)}" for n, s in zip(nproc, speedup))
    lines.append(f'<polyline fill="none" stroke="#2ca02c" stroke-width="2.5" points="{su_pts}"/>')
    for n, s in zip(nproc, speedup):
        lines.append(
            f'<circle cx="{x_px(n)}" cy="{y2_s(s)}" r="4" fill="#2ca02c" stroke="white" stroke-width="1"/>'
        )

    e_pts = " ".join(f"{x_px(n)},{y2_e(e)}" for n, e in zip(nproc, eff_pct))
    lines.append(f'<polyline fill="none" stroke="#ff7f0e" stroke-width="2" points="{e_pts}"/>')
    for n, e in zip(nproc, eff_pct):
        lines.append(
            f'<circle cx="{x_px(n)}" cy="{y2_e(e)}" r="3" fill="#ff7f0e" stroke="white" stroke-width="1"/>'
        )

    lines.append(
        f'<text x="{m["l"]}" y="{H-18}" font-size="10" fill="#666">Green: measured speedup (T₁/Tₙ). Gray dashed: ideal. Orange: efficiency 100·T₁/(n·Tₙ).</text>'
    )
    lines.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _nice_ticks(lo: float, hi: float, n: int) -> List[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    p10 = 10.0 ** math.floor(math.log10(max(raw, 1e-12)))
    step = max(p10, p10 * 2, p10 * 5)
    for cand in (p10, 2 * p10, 5 * p10, 10 * p10):
        if (hi - lo) / cand <= n + 2:
            step = cand
            break
    ticks = []
    x = math.ceil(lo / step) * step
    while x <= hi + 1e-9:
        ticks.append(x)
        x += step
    return ticks if ticks else [lo, hi]


def try_matplotlib_png(
    png_path: str,
    nproc: List[int],
    wall: List[float],
    title: str,
    x_axis_label: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    t1 = wall[0]
    speedup = [t1 / w for w in wall]
    ideal = [float(n) for n in nproc]
    eff = [100.0 * speedup[i] / nproc[i] for i in range(len(nproc))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="600")

    ax1.loglog(nproc, wall, "o-", color="#1f77b4", lw=2, ms=6)
    ax1.set_ylabel("Wall time (s)")
    ax1.set_title("Strong scaling — Kobayashi FD (512×512, 5000 steps)")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(nproc, speedup, "o-", color="#2ca02c", lw=2, ms=6, label="Measured speedup")
    ax2.plot(nproc, ideal, "--", color="#999", lw=2, label="Ideal")
    ax2.set_xscale("log")
    ax2.set_xticks(nproc)
    ax2.set_xticklabels([str(n) for n in nproc])
    ax2.set_xlabel(x_axis_label)
    ax2.set_ylabel("Speedup")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    ax2b = ax2.twinx()
    ax2b.plot(nproc, eff, "s-", color="#ff7f0e", lw=1.5, ms=4, alpha=0.85, label="Efficiency %")
    ax2b.set_ylabel("Efficiency (%)")
    ax2b.set_ylim(0, min(115, max(eff) * 1.15))

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        help="summary.tsv or directory containing it",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Output SVG path (default: <dir>/strong_scaling.svg)",
    )
    ap.add_argument(
        "--png",
        action="store_true",
        help="Also write PNG if matplotlib is installed",
    )
    ap.add_argument("--title", default="", help="Figure title")
    args = ap.parse_args()

    path = args.path
    if os.path.isdir(path):
        summary = os.path.join(path, "summary.tsv")
    else:
        summary = path
    if not os.path.isfile(summary):
        print(f"missing {summary}", file=sys.stderr)
        return 1

    x_key, nproc, wall = read_summary_tsv(summary)
    base_dir = os.path.dirname(os.path.abspath(summary))
    out_svg = args.output or os.path.join(base_dir, "strong_scaling.svg")
    x_lab = axis_label_from_tsv_key(x_key)
    title = args.title or f"Kobayashi strong scaling ({x_lab}, summary.tsv)"

    write_svg(out_svg, nproc, wall, title, x_lab)
    print(f"wrote {out_svg}")

    if args.png:
        out_png = os.path.splitext(out_svg)[0] + ".png"
        if try_matplotlib_png(out_png, nproc, wall, title, x_lab):
            print(f"wrote {out_png}")
        else:
            print("matplotlib not available; skipped PNG", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
