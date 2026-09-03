#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Digitize Karma, PRL 87, 115701 (2001) FIG. 1 present-model V*(t*) from a raster.

The published axes are t D/d0^2 in [0, 10000] and V d0/D in [0, 0.08]. The
present-model pair (d0/W = 0.544 solid, 0.277 dotted) overlap; this traces that
lower envelope. The initial seed spike is clipped by the frame at V*=0.08.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Inner plot rectangle on the 400 dpi raster of arXiv cond-mat/0103289 page 4.
# Measured from the axis spines (not the tick labels).
BOX = dict(left=633, right=1485, top=312, bottom=1019)
XMIN, XMAX = 0.0, 10000.0
YMIN, YMAX = 0.0, 0.08


def x_of(px: float) -> float:
    return XMIN + (px - BOX["left"]) / (BOX["right"] - BOX["left"]) * (XMAX - XMIN)


def y_of(py: float) -> float:
    return YMAX - (py - BOX["top"]) / (BOX["bottom"] - BOX["top"]) * (YMAX - YMIN)


def in_legend(px: int, py: int) -> bool:
    """Legend sits in the upper-right; curves at those t* are near V*~0.018."""
    t = x_of(px)
    v = y_of(py)
    return t > 4800.0 and v > 0.030


def is_left_tick(px: int, py: int) -> bool:
    """Major Y ticks (0.02, 0.04, 0.06, 0.08) leak a few pixels into the plot."""
    if px > BOX["left"] + 22:
        return False
    v = y_of(py)
    return any(abs(v - tick) < 0.0018 for tick in (0.02, 0.04, 0.06, 0.08, 0.0))


def clusters(ys: np.ndarray, gap: int = 5) -> list[tuple[float, int]]:
    if ys.size == 0:
        return []
    ys = np.sort(ys)
    out: list[tuple[int, int]] = []
    s = int(ys[0])
    p = int(ys[0])
    for y in ys[1:]:
        y = int(y)
        if y <= p + gap:
            p = y
        else:
            out.append((s, p))
            s = p = y
    out.append((s, p))
    return [(0.5 * (a + b), b - a + 1) for a, b in out]


def digitize(gray: np.ndarray, ink_thr: float = 108.0) -> np.ndarray:
    left, right, top, bot = BOX["left"], BOX["right"], BOX["top"], BOX["bottom"]
    rows: list[tuple[float, float]] = []
    y_hat = float(bot - 12)  # seed: lowest present curve at late time
    y_prev = y_hat
    # Walk right-to-left so we lock onto V*~0.018 before the steep transient.
    # Stay inside the left spine (the axis itself is solid ink).
    for px in range(right - 4, left + 6, -1):
        ys = []
        for py in range(top + 3, bot - 6):
            if gray[py, px] >= ink_thr:
                continue
            if in_legend(px, py) or is_left_tick(px, py):
                continue
            v = y_of(py)
            if not (0.012 < v < 0.0805):
                continue
            ys.append(py)
        cl = clusters(np.array(ys, dtype=int)) if ys else []
        t = x_of(px)
        if cl:
            dpy = y_hat - y_prev
            y_pred = y_hat + dpy
            # Steep seed drop: ~40 py per px; keep the gate on the predicted path.
            gate = max(12.0, min(80.0, 4.0 * abs(dpy) + (40.0 if t < 600.0 else 8.0)))
            mids = np.array([c[0] for c in cl])
            near = np.abs(mids - y_pred) <= gate
            pick = mids[near] if np.any(near) else mids
            y_prev = y_hat
            # Lowest on the page = largest py = smallest V* (present envelope).
            y_hat = float(np.max(pick))
        rows.append((t, min(YMAX, y_of(y_hat))))
    rows.reverse()
    arr = np.array(rows)
    # Frame clips at 0.08; the bundle enters at the top-left corner.
    if len(arr) == 0 or arr[0, 0] > 1.0:
        arr = np.vstack(([0.0, YMAX], arr))
    elif arr[0, 1] < 0.075:
        arr[0, 1] = YMAX
        if arr[0, 0] > 0.0:
            arr = np.vstack(([0.0, YMAX], arr))
    step = 6
    keep = np.zeros(len(arr), dtype=bool)
    keep[0] = keep[-1] = True
    keep[arr[:, 0] < 400.0] = True
    keep[::step] = True
    return arr[keep]


def overlay(rgb: Image.Image, xy: np.ndarray, old: np.ndarray | None, out: Path) -> None:
    left, right, top, bot = BOX["left"], BOX["right"], BOX["top"], BOX["bottom"]
    draw = ImageDraw.Draw(rgb)
    draw.rectangle([left, top, right, bot], outline=(0, 160, 0))

    def to_px(t: float, v: float) -> tuple[int, int]:
        px = int(round(left + (t - XMIN) / (XMAX - XMIN) * (right - left)))
        py = int(round(top + (YMAX - v) / (YMAX - YMIN) * (bot - top)))
        return px, py

    if old is not None and old.size:
        for t, v in old:
            px, py = to_px(float(t), float(v))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], outline=(40, 90, 220))
    for t, v in xy:
        px, py = to_px(float(t), float(v))
        draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(200, 30, 30))
    crop = rgb.crop((left - 70, top - 28, right + 18, bot + 78))
    crop.save(out)


def write_tsv(xy: np.ndarray, path: Path) -> None:
    header = (
        "# Digitised from Karma, Phys. Rev. Lett. 87, 115701 (2001) FIG. 1\n"
        "# arXiv cond-mat/0103289 page-4 raster at 400 dpi; axis spines as origin.\n"
        "# Present-model envelope (d0/W = 0.544 solid and 0.277 dotted overlap).\n"
        "# Published Y max is 0.08: the seed spike is clipped by the frame.\n"
        "# Uncertainty ~0.001 in V d0/D (line thickness + raster).\n"
        "# tD/d0^2  Vd0/D\n"
    )
    with path.open("w") as f:
        f.write(header)
        for t, v in xy:
            f.write(f"{t:.6f} {v:.6f}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raster", type=Path, help="400 dpi PNG of arXiv page 4")
    p.add_argument("--out-tsv", type=Path, required=True)
    p.add_argument("--out-overlay", type=Path, required=True)
    p.add_argument("--old-tsv", type=Path, default=None)
    args = p.parse_args()
    im = Image.open(args.raster).convert("RGB")
    gray = np.asarray(im.convert("L"), dtype=float)
    xy = digitize(gray)
    old = np.loadtxt(args.old_tsv, comments="#") if args.old_tsv and args.old_tsv.exists() else None
    overlay(im, xy, old, args.out_overlay)
    write_tsv(xy, args.out_tsv)
    print(f"wrote {args.out_tsv}  n={len(xy)}  t*[{xy[0,0]:.1f},{xy[-1,0]:.1f}]  "
          f"V*[{xy[0,1]:.4f}..{xy[-1,1]:.4f}]")
    print(f"wrote {args.out_overlay}")
    for ts in (0, 80, 150, 300, 500, 1000, 2000, 4000, 10000):
        print(f"  t*={ts:5d}  V*={np.interp(ts, xy[:,0], xy[:,1]):.4f}")


if __name__ == "__main__":
    main()
