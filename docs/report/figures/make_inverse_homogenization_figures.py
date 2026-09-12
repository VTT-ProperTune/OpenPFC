#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PNG/SVG figures for the inverse-homogenization report chapter.

No matplotlib: dumps are greyscale fields; we write PNG via zlib and SVG as
text. Labels are baked into the PNGs so the PDF build does not depend on
external image hrefs inside SVG.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = Path("/scratch/project_462001519/juaho/inverse-homogenization")

NAVY = (18, 22, 36)
CREAM = (238, 233, 220)
GAP = (92, 96, 104)
TEAL = (46, 140, 148)
RUST = (176, 72, 48)
GOLD = (212, 168, 72)
WHITE = (255, 255, 255)
INK = (16, 16, 20)


def load_h(path: Path) -> tuple[int, int, int, list[float]]:
    toks = path.read_text().split()
    nx, ny, nz = map(int, toks[:3])
    vals = [float(x) for x in toks[3:]]
    assert len(vals) == nx * ny * nz, (path, len(vals), nx * ny * nz)
    return nx, ny, nz, vals


def slice2d(nx: int, ny: int, nz: int, vals: list[float], k: int = 0) -> list[list[float]]:
    out = [[0.0] * nx for _ in range(ny)]
    for j in range(ny):
        for i in range(nx):
            out[j][i] = vals[(k * ny + j) * nx + i]
    return out


def tile(a: list[list[float]], tx: int, ty: int) -> list[list[float]]:
    ny, nx = len(a), len(a[0])
    return [[a[j % ny][i % nx] for i in range(nx * tx)] for j in range(ny * ty)]


def sample(a: list[list[float]], x: float, y: float) -> float:
    ny, nx = len(a), len(a[0])
    i = int(x) % nx
    j = int(y) % ny
    return a[j][i]


def warp(a: list[list[float]], ex: float, ey: float, scale: int = 3) -> list[list[float]]:
    """Pull along y by ey; lateral stretch ex (auxetic if ex>0 with ey>0)."""
    ny, nx = len(a), len(a[0])
    out_nx = max(1, int(round(nx * scale * (1.0 + ex))))
    out_ny = max(1, int(round(ny * scale * (1.0 + ey))))
    out = [[0.0] * out_nx for _ in range(out_ny)]
    for j in range(out_ny):
        y0 = j / (1.0 + ey) / scale
        for i in range(out_nx):
            x0 = i / (1.0 + ex) / scale
            out[j][i] = sample(a, x0, y0)
    return out


def upsample(a: list[list[float]], s: int) -> list[list[float]]:
    ny, nx = len(a), len(a[0])
    return [[a[j // s][i // s] for i in range(nx * s)] for j in range(ny * s)]


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return tuple(int(a[k] + t * (b[k] - a[k])) for k in range(3))  # type: ignore[return-value]


def rgb_of(h: float, solid=NAVY, void=CREAM) -> tuple[int, int, int]:
    t = 0.0 if h < 0.0 else 1.0 if h > 1.0 else h
    return lerp_rgb(void, solid, t)


def to_rgb(rows: list[list[float]], solid=NAVY, void=CREAM) -> list[list[tuple[int, int, int]]]:
    return [[rgb_of(v, solid, void) for v in row] for row in rows]


def write_png(path: Path, rows: list[list[tuple[int, int, int]]]) -> None:
    h = len(rows)
    w = len(rows[0])
    raw = bytearray()
    for j in range(h):
        raw.append(0)
        for i in range(w):
            raw.extend(rows[j][i])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    print(f"wrote {path} {w}x{h}")


def canvas(w: int, h: int, color: tuple[int, int, int]) -> list[list[tuple[int, int, int]]]:
    return [[color] * w for _ in range(h)]


def blit(
    dst: list[list[tuple[int, int, int]]],
    src: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
) -> None:
    dh, dw = len(dst), len(dst[0])
    for j, row in enumerate(src):
        y = y0 + j
        if y < 0 or y >= dh:
            continue
        for i, pix in enumerate(row):
            x = x0 + i
            if 0 <= x < dw:
                dst[y][x] = pix


def fill_rect(
    dst: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    dh, dw = len(dst), len(dst[0])
    for j in range(h):
        y = y0 + j
        if y < 0 or y >= dh:
            continue
        for i in range(w):
            x = x0 + i
            if 0 <= x < dw:
                dst[y][x] = color


def hstack_rgb(
    imgs: list[list[list[tuple[int, int, int]]]],
    gap: int = 8,
    gap_color: tuple[int, int, int] = GAP,
) -> list[list[tuple[int, int, int]]]:
    h = max(len(im) for im in imgs)
    w = sum(len(im[0]) for im in imgs) + gap * (len(imgs) - 1)
    out = canvas(w, h, gap_color)
    x = 0
    for k, im in enumerate(imgs):
        blit(out, im, x, (h - len(im)) // 2)
        x += len(im[0]) + (gap if k < len(imgs) - 1 else 0)
    return out


def vstack_rgb(
    imgs: list[list[list[tuple[int, int, int]]]],
    gap: int = 8,
    gap_color: tuple[int, int, int] = GAP,
) -> list[list[tuple[int, int, int]]]:
    w = max(len(im[0]) for im in imgs)
    h = sum(len(im) for im in imgs) + gap * (len(imgs) - 1)
    out = canvas(w, h, gap_color)
    y = 0
    for k, im in enumerate(imgs):
        blit(out, im, (w - len(im[0])) // 2, y)
        y += len(im) + (gap if k < len(imgs) - 1 else 0)
    return out


# 5x7 glyphs, bit rows MSB left. Uppercase + digits + a few marks.
_FONT: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04),
    ",": (0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08),
    ":": (0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "(": (0x04, 0x08, 0x10, 0x10, 0x10, 0x08, 0x04),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "?": (0x0E, 0x11, 0x01, 0x06, 0x04, 0x00, 0x04),
}


def text_size(text: str, scale: int) -> tuple[int, int]:
    return (len(text) * 6 - 1) * scale, 7 * scale


def blit_text(
    dst: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    text: str,
    color: tuple[int, int, int],
    scale: int = 2,
) -> None:
    for ci, ch in enumerate(text.upper()):
        glyph = _FONT.get(ch, _FONT["?"])
        for r, bits in enumerate(glyph):
            for c in range(5):
                if bits & (1 << (4 - c)):
                    fill_rect(dst, x0 + (ci * 6 + c) * scale, y0 + r * scale, scale, scale, color)


def captioned(
    im: list[list[tuple[int, int, int]]],
    lines: list[str],
    accent: tuple[int, int, int] = TEAL,
    bar: tuple[int, int, int] = NAVY,
) -> list[list[tuple[int, int, int]]]:
    scale = 2
    bar_h = 14 + len(lines) * (7 * scale + 8)
    text_w = max(text_size(line, scale)[0] for line in lines) + 28
    w = max(len(im[0]), text_w)
    h = len(im)
    out = canvas(w, h + bar_h, bar)
    blit(out, im, (w - len(im[0])) // 2, 0)
    fill_rect(out, 0, h, 6, bar_h, accent)
    y = h + 8
    for line in lines:
        blit_text(out, 16, y, line, CREAM, scale)
        y += 7 * scale + 8
    return out


def frame(
    im: list[list[tuple[int, int, int]]],
    pad: int = 6,
    color: tuple[int, int, int] = NAVY,
) -> list[list[tuple[int, int, int]]]:
    w, h = len(im[0]), len(im)
    out = canvas(w + 2 * pad, h + 2 * pad, color)
    blit(out, im, pad, pad)
    return out


def place_in(
    im: list[list[tuple[int, int, int]]],
    box_w: int,
    box_h: int,
    bg: tuple[int, int, int] = CREAM,
    align: str = "center",
) -> list[list[tuple[int, int, int]]]:
    out = canvas(box_w, box_h, bg)
    x = (box_w - len(im[0])) // 2 if align == "center" else 0
    y = (box_h - len(im)) // 2
    blit(out, im, x, y)
    return out


def width_bar(
    dst: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    width: int,
    color: tuple[int, int, int],
) -> None:
    fill_rect(dst, x0, y0, width, 3, color)
    fill_rect(dst, x0, y0 - 6, 3, 15, color)
    fill_rect(dst, x0 + width - 3, y0 - 6, 3, 15, color)


def stiffness_svg(path: Path, C: list[list[float]], title: str) -> None:
    n = 6
    cell = 54
    left, top = 56, 48
    labels = ["11", "22", "33", "23", "13", "12"]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + n * cell + 28}" height="{top + n * cell + 48}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{(left + n * cell) / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="13">{title}</text>',
        '<text x="210" y="38" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#2e8c94">gold outline: in-plane C12 &lt; 0 (auxetic)</text>',
    ]
    for i in range(n):
        parts.append(
            f'<text x="{left - 8}" y="{top + i * cell + 32}" text-anchor="end" font-family="sans-serif" font-size="10">{labels[i]}</text>'
        )
        parts.append(
            f'<text x="{left + i * cell + 27}" y="{top + n * cell + 18}" text-anchor="middle" font-family="sans-serif" font-size="10">{labels[i]}</text>'
        )
        for j in range(n):
            v = C[i][j]
            # Compress so the small negative C12 still reads as teal, not white.
            t = v / 0.05
            t = max(-1.0, min(1.0, t))
            if t >= 0:
                r, g, b = lerp_rgb(WHITE, RUST, t)
            else:
                r, g, b = lerp_rgb(WHITE, TEAL, -t)
            fill = f"#{r:02x}{g:02x}{b:02x}"
            x, y = left + j * cell, top + i * cell
            highlight = (i, j) in ((0, 1), (1, 0))
            stroke = "#c8a048" if highlight else "#333"
            sw = 2.4 if highlight else 0.4
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
            )
            ink = "#111" if abs(t) < 0.55 else "#fff"
            parts.append(
                f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 + 4}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="{ink}">{v:.3f}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


def isometric_surface(
    nx: int, ny: int, nz: int, vals: list[float], pix: int = 3
) -> list[list[tuple[int, int, int]]]:
    solid = [v > 0.5 for v in vals]

    def at(i: int, j: int, k: int) -> bool:
        return solid[(k * ny + j) * nx + i]

    neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    w = (nx + nz + 2) * pix
    h = (ny + nz // 2 + 2) * pix
    img = canvas(w, h, CREAM)
    # Painter: far (k=0) first, near last. Offset grows with k.
    for k in range(nz):
        zf = k / max(nz - 1, 1)
        col = lerp_rgb(NAVY, TEAL, 0.15 + 0.85 * zf)
        top = lerp_rgb(col, WHITE, 0.18)
        for j in range(ny):
            for i in range(nx):
                if not at(i, j, k):
                    continue
                surf = False
                for di, dj, dk in neigh:
                    if not at((i + di) % nx, (j + dj) % ny, (k + dk + nz) % nz):
                        surf = True
                        break
                if not surf:
                    continue
                x = (i + k + 1) * pix
                y = (j + k // 2 + 1) * pix
                face = top if (k + 1 >= nz or not at(i, j, k + 1)) else col
                south = lerp_rgb(col, INK, 0.35)
                east = lerp_rgb(col, INK, 0.18)
                drop = pix // 2 + 1
                for dy in range(drop):
                    for dx in range(pix):
                        yy, xx = y + pix + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            img[yy][xx] = south
                    for dx in range(drop):
                        yy, xx = y + dx, x + pix + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            img[yy][xx] = east
                        yy, xx = y + pix + dx, x + pix + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            img[yy][xx] = south
                for dy in range(pix):
                    for dx in range(pix):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            img[yy][xx] = face
    return img


def stacked_slices(
    nx: int, ny: int, nz: int, vals: list[float], nshow: int = 8, scale: int = 3
) -> list[list[tuple[int, int, int]]]:
    ks = [int(round(i * (nz - 1) / (nshow - 1))) for i in range(nshow)]
    sx, sy = 14, 10
    card_w, card_h = nx * scale, ny * scale
    w = card_w + (nshow - 1) * sx + 8
    h = card_h + (nshow - 1) * sy + 8
    img = canvas(w, h, GAP)
    for idx, k in enumerate(ks):
        sl = slice2d(nx, ny, nz, vals, k)
        sl_bin = [[1.0 if p > 0.5 else 0.0 for p in row] for row in sl]
        rgb = to_rgb(upsample(sl_bin, scale))
        ox = 4 + idx * sx
        oy = 4 + idx * sy
        # drop shadow
        fill_rect(img, ox + 3, oy + 3, card_w, card_h, (40, 42, 48))
        blit(img, rgb, ox, oy)
        fill_rect(img, ox, oy, card_w, 3, TEAL if idx == nshow - 1 else NAVY)
    return img


def main() -> None:
    rot_ok = SCRATCH / "auxgeo_21956076/rotating_h.txt"
    rot_lo = SCRATCH / "auxgeo_21955691/rotating_h.txt"
    rot_hi = SCRATCH / "auxgeo_21955873/rotating_h.txt"
    seed_a = SCRATCH / "auxetic_21955207/continue_h.txt"
    seed_b = SCRATCH / "auxetic_21955207/seed7_h.txt"
    ch_p = SCRATCH / "stage6_21958468/ch_c0.50_k1_ay1.0_h.txt"
    vol_p = SCRATCH / "stage8_21964843/inv64_h.txt"

    def field2d(path: Path) -> list[list[float]]:
        nx, ny, nz, v = load_h(path)
        return slice2d(nx, ny, nz, v)

    R_ok = field2d(rot_ok)
    R_lo = field2d(rot_lo)
    R_hi = field2d(rot_hi)
    CH = field2d(ch_p)

    # Hero wallpaper: 3x3 hinged rotating squares.
    write_png(
        ROOT / "inverse_homogenization_rotating_tile.png",
        to_rgb(upsample(tile(R_ok, 3, 3), 4)),
    )

    # Hinge-width family: the geometry that can actually go auxetic.
    hinges = hstack_rgb(
        [
            captioned(
                frame(to_rgb(upsample(tile(R_lo, 2, 2), 3))),
                ["DISCONNECTED  HALF=0.185", "FOUR ISLANDS   NU > 0"],
                accent=RUST,
            ),
            captioned(
                frame(to_rgb(upsample(tile(R_ok, 2, 2), 3))),
                ["HINGES        HALF=0.200", "PERCOLATING   NU = -0.123"],
                accent=TEAL,
            ),
            captioned(
                frame(to_rgb(upsample(tile(R_hi, 2, 2), 3))),
                ["FUSED         HALF=0.215", "PLATE+HOLES   NU = +0.14"],
                accent=RUST,
            ),
        ],
        gap=14,
        gap_color=CREAM,
    )
    write_png(ROOT / "inverse_homogenization_hinges.png", hinges)

    write_png(
        ROOT / "inverse_homogenization_spinodal_tile.png",
        to_rgb(upsample(tile(CH, 3, 3), 3)),
    )

    # Kinematic money figure. Poisson effect x5 so the SIGN is visible;
    # measured nu is in the caption bar.
    ey = 0.22
    rot_rest = to_rgb(upsample(tile(R_ok, 3, 3), 2))
    # Threshold the CH field so the kinematic comparison is morphology, not grey.
    CH_bin = [[1.0 if p > 0.5 else 0.0 for p in row] for row in CH]
    ch_rest = to_rgb(upsample(tile(CH_bin, 3, 3), 2))
    rot_def = to_rgb(warp(tile(R_ok, 3, 3), ex=5.0 * 0.123 * ey, ey=ey, scale=2))
    ch_def = to_rgb(warp(tile(CH_bin, 3, 3), ex=-5.0 * 0.23 * ey, ey=ey, scale=2))

    def pair(
        rest: list[list[tuple[int, int, int]]],
        stretched: list[list[tuple[int, int, int]]],
        title: list[str],
        accent: tuple[int, int, int],
        rest_tag: str,
        stretch_tag: str,
    ) -> list[list[tuple[int, int, int]]]:
        box_w = max(len(rest[0]), len(stretched[0])) + 16
        box_h = max(len(rest), len(stretched)) + 28
        a = place_in(rest, box_w, box_h, CREAM)
        b = place_in(stretched, box_w, box_h, CREAM)
        width_bar(a, 8, box_h - 10, len(rest[0]), NAVY)
        width_bar(b, 8, box_h - 10, len(stretched[0]), accent)
        blit_text(a, 10, 6, rest_tag, NAVY, 2)
        blit_text(b, 10, 6, stretch_tag, accent, 2)
        return captioned(hstack_rgb([frame(a, 4, NAVY), frame(b, 4, accent)], gap=18, gap_color=CREAM), title, accent)

    stretch = vstack_rgb(
        [
            pair(
                rot_rest,
                rot_def,
                ["ROTATING SQUARES   NU = -0.123", "SAME AXIAL STRETCH  --  PATTERN GETS WIDER"],
                TEAL,
                "REST",
                "STRETCHED  WIDENS",
            ),
            pair(
                ch_rest,
                ch_def,
                ["CAHN-HILLIARD SPINODAL   NU = +0.23", "SAME AXIAL STRETCH  --  PATTERN GETS NARROWER"],
                RUST,
                "REST",
                "STRETCHED  NARROWS",
            ),
        ],
        gap=18,
        gap_color=CREAM,
    )
    write_png(ROOT / "inverse_homogenization_stretch.png", stretch)

    sa = field2d(seed_a)
    sb = field2d(seed_b)
    write_png(
        ROOT / "inverse_homogenization_two_seeds.png",
        hstack_rgb(
            [
                captioned(
                    frame(to_rgb(upsample(tile(sa, 2, 2), 2))),
                    ["SEED A  CONTINUED", "SAME AUXETIC TARGET"],
                    accent=TEAL,
                ),
                captioned(
                    frame(to_rgb(upsample(tile(sb, 2, 2), 2))),
                    ["SEED B  NOISE", "CORR(H) = 0.19"],
                    accent=GOLD,
                ),
            ],
            gap=16,
            gap_color=CREAM,
        ),
    )

    vx, vy, vz, vv = load_h(vol_p)
    mn, mx = min(vv), max(vv)
    span = mx - mn if mx > mn else 1.0
    stretched = [(v - mn) / span for v in vv]
    slices = []
    for k in (0, vz // 4, vz // 2, 3 * vz // 4):
        sl = slice2d(vx, vy, vz, stretched, k)
        sl_bin = [[1.0 if p > 0.5 else 0.0 for p in row] for row in sl]
        slices.append(
            captioned(
                frame(to_rgb(upsample(sl_bin, 3)), 3),
                [f"Z = {k}/{vz}"],
                accent=TEAL,
            )
        )
    write_png(
        ROOT / "inverse_homogenization_3d_slices.png",
        hstack_rgb(slices, gap=8, gap_color=CREAM),
    )

    iso = isometric_surface(vx, vy, vz, stretched, pix=4)
    stack = stacked_slices(vx, vy, vz, stretched, nshow=8, scale=3)
    write_png(
        ROOT / "inverse_homogenization_3d_iso.png",
        hstack_rgb(
            [
                captioned(
                    frame(iso, 8, NAVY),
                    ["64 X 64 X 64 SURFACE VOXELS", "JOB 21964843  THRESHOLD 0.5"],
                    accent=TEAL,
                ),
                captioned(
                    frame(stack, 8, NAVY),
                    ["EIGHT Z-SLICES, STACKED", "NOT AN EXTRUDED 2-D PATTERN"],
                    accent=GOLD,
                ),
            ],
            gap=16,
            gap_color=CREAM,
        ),
    )

    C = [
        [0.241, -0.017, 0.067, 0.0, 0.0, -0.001],
        [-0.017, 0.241, 0.067, 0.0, 0.0, -0.001],
        [0.067, 0.067, 0.658, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.124, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.124, 0.0],
        [-0.001, -0.001, 0.0, 0.0, 0.0, 0.055],
    ]
    stiffness_svg(ROOT / "inverse_homogenization_CH.svg", C, "C_H rotating squares (HIP, job 21967095)")


if __name__ == "__main__":
    main()
