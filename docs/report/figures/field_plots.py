#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render OpenPFC field snapshots as publication-quality figures.

Five entry points, all returning a `matplotlib.figure.Figure` the caller
saves. Three render a 2-D field as an image:

- `render_panel` -- one field, one axes, a colour bar.
- `render_montage` -- a time series as a row of panels sharing one colour
  bar, so morphology evolution is visible in a single figure.
- `render_comparison` -- two fields (typically two runs) side by side on a
  *shared* colour scale, for A-vs-B comparisons.

Two render a **one-dimensional** field as a line plot, for the applications
whose domain is `Lx x 1 x 1` (`apps/kawahara`), where an image of a
one-pixel-tall field would be unreadable:

- `render_line_panel` -- several 1-D profiles overlaid on one axes.
- `render_line_comparison` -- two runs side by side on shared x and y axes.

Colour map policy (see module docstring in `field_io.py` for the data
side): a field is either "sequential" (one-sided, e.g. a film thickness
that is always positive) or "diverging" (signed around a physically
meaningful midpoint, e.g. a composition around its mean or a PFC density
around its baseline). Callers choose via `kind="sequential"` or
`kind="diverging"` and, for diverging fields, a `center` value; a plain
rainbow map (`jet`, `turble`, ...) is never used.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend choice)
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

from field_io import Field2D  # noqa: E402

# Perceptually uniform, colour-blind-safe. `viridis` for one-sided
# quantities (monotonic lightness ramp); `RdBu_r` for signed quantities
# (ColorBrewer-vetted colour-blind-safe diverging scheme, blue-white-red so
# it does not rely on red/green discrimination).
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

TITLE_FS = 10.5
LABEL_FS = 9
TICK_FS = 8


def _norm_for(data, kind: str, center: Optional[float], vmin, vmax):
    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())
    if kind == "diverging":
        if center is None:
            raise ValueError("kind='diverging' requires a physically meaningful `center`")
        # Guard against a (numerically) flat field, which would make
        # TwoSlopeNorm's vmin==vcenter==vmax degenerate.
        if vmin >= center:
            vmin = center - 1e-12
        if vmax <= center:
            vmax = center + 1e-12
        return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    if kind == "sequential":
        return Normalize(vmin=vmin, vmax=vmax)
    raise ValueError(f"kind must be 'sequential' or 'diverging', got {kind!r}")


def _cmap_for(kind: str) -> str:
    return DIVERGING_CMAP if kind == "diverging" else SEQUENTIAL_CMAP


def _style_image_axes(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")


def render_panel(
    field: Field2D,
    *,
    kind: str,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    cbar_label: str = "",
    axis_units: str = "grid units",
    figsize=(4.6, 4.0),
) -> plt.Figure:
    """One scalar field, one axes, a colour bar.

    `title` should be caption-friendly (what the reader is looking at), not
    just a variable name -- e.g. "tungsten seed, mid-depth slice, t=10".
    """
    norm = _norm_for(field.data, kind, center, vmin, vmax)
    cmap = _cmap_for(kind)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(field.data, extent=field.extent, origin="lower", cmap=cmap, norm=norm,
                    interpolation="nearest")
    _style_image_axes(ax, f"x ({axis_units})", f"y ({axis_units})")
    ax.set_title(title, loc="left", fontsize=TITLE_FS)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    fig.tight_layout()
    return fig


def render_montage(
    fields: Sequence[Field2D],
    *,
    kind: str,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    panel_titles: Optional[Sequence[str]] = None,
    suptitle: str = "",
    cbar_label: str = "",
    axis_units: str = "grid units",
    ncols: Optional[int] = None,
    panel_size=(2.6, 2.4),
) -> plt.Figure:
    """A time series as a row (or grid) of panels sharing one colour bar.

    All panels use the *same* colour scale (computed across all `fields`
    unless `vmin`/`vmax` are given) so morphology can be compared across
    time at a glance.
    """
    n = len(fields)
    if n == 0:
        raise ValueError("render_montage needs at least one field")
    ncols = ncols or n
    nrows = -(-n // ncols)  # ceil

    all_data = fields[0].data if n == 1 else __import__("numpy").concatenate(
        [f.data.ravel() for f in fields]
    )
    norm = _norm_for(all_data, kind, center, vmin, vmax)
    cmap = _cmap_for(kind)

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                              squeeze=False)
    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i < n:
            f = fields[i]
            im = ax.imshow(f.data, extent=f.extent, origin="lower", cmap=cmap, norm=norm,
                            interpolation="nearest")
            label = panel_titles[i] if panel_titles else (
                f"t={f.time:g}" if f.time is not None else "")
            ax.set_title(label, fontsize=TITLE_FS)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=TITLE_FS + 0.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 0.92, 0.94 if suptitle else 1.0))
    cbar_ax = fig.add_axes((0.93, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    return fig


def render_comparison(
    field_a: Field2D,
    field_b: Field2D,
    *,
    kind: str,
    label_a: str,
    label_b: str,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    suptitle: str = "",
    cbar_label: str = "",
    axis_units: str = "grid units",
    figsize=(8.4, 4.0),
) -> plt.Figure:
    """Two fields side by side on a shared colour scale (A-vs-B panel).

    This is the "is run A different from run B" figure: e.g. single-mode
    growth vs broadband coarsening from the same physical model, or an
    unstable-vs-stable parameter choice.
    """
    import numpy as np

    both = np.concatenate([field_a.data.ravel(), field_b.data.ravel()])
    norm = _norm_for(both, kind, center, vmin, vmax)
    cmap = _cmap_for(kind)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize)
    for ax, f, label in ((axa, field_a, label_a), (axb, field_b, label_b)):
        ax.imshow(f.data, extent=f.extent, origin="lower", cmap=cmap, norm=norm,
                  interpolation="nearest")
        _style_image_axes(ax, f"x ({axis_units})", f"y ({axis_units})")
        ax.set_title(label, loc="left", fontsize=TITLE_FS)
    axb.set_ylabel("")
    axb.tick_params(labelleft=False)

    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=TITLE_FS + 0.5)
    fig.tight_layout(rect=(0, 0, 0.90, 1.0))
    im = axa.images[0]
    cbar_ax = fig.add_axes((0.91, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    return fig


# --------------------------------------------------------------------------
# 1-D line renderers
# --------------------------------------------------------------------------
#
# Why these exist. `apps/kawahara` is a one-dimensional application: its
# domain is `Lx x 1 x 1` and its `.vti` snapshots are single rows. Rendering
# a 1-pixel-tall image through `render_panel` would be technically correct
# and visually useless -- and worse, it would hide the very thing the
# chapter is about, because the interesting signal there (a shed dispersive
# wave train whose RMS is under a tenth of the pulse height) is an
# *amplitude*, and amplitude is what a colour bar reads worst. A line plot
# puts u on a linear axis, where a 10% ripple is 10% of the frame.
#
# Colour discipline for lines follows `make_figures.py` (which draws the
# scalability curves): a small ordered palette that is colour-blind-safe
# (Okabe-Ito, the standard eight-colour set), ordered so consecutive entries
# also differ in luminance, and paired one-for-one with dash patterns so a
# greyscale print never has to separate two curves on hue alone.

# Okabe & Ito (2008). Ordered so that consecutive entries also step up in
# luminance -- blue L*=46, vermillion L*=54, bluish green L*=58, orange
# L*=71, reddish purple L*=61 -- but note vermillion and bluish green are
# only four L* apart, which greyscale will not separate reliably. That is
# what LINE_STYLES is for: every curve gets a different dash pattern as
# well as a different hue, so the encoding never rests on colour alone.
LINE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")
LINE_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)))


def line_profile(field: Field2D):
    """Extract `(x, u)` from a 1-D `Field2D` (a `Lx x 1 x 1` snapshot).

    `read_vti` squeezes a `Ly == Lz == 1` domain to a `(1, nx)` array, so the
    single row *is* the profile.

    `x` is **node-centred**: sample `i` sits at `xmin + i*dx`, which is
    what the solver itself uses -- `pfc::data::Field::coords` returns
    `origin + (low + i) * spacing`, so the first sample is at the origin and
    not half a cell in from it. This matters because it is exactly the
    convention `Field2D.extent` does *not* use: `extent` spans
    `n * dx` because `imshow` draws pixels as cells, and taking the naive
    `(i + 1/2) * dx` from it would shift every curve by half a cell. That is
    invisible on a single curve and a systematic bias the moment the figure
    is read for where a pulse is.
    """
    import numpy as np

    data = field.data
    if data.ndim != 2 or data.shape[0] != 1:
        raise ValueError(
            f"line_profile expects a 1-D field of shape (1, nx), got {data.shape}"
        )
    u = data[0]
    xmin, xmax = field.extent[0], field.extent[1]
    n = u.size
    dx = (xmax - xmin) / n
    x = xmin + np.arange(n) * dx
    return x, u


def _style_line_axes(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", axis="both", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def _draw_profiles(ax, fields: Sequence[Field2D], labels: Optional[Sequence[str]]):
    for i, f in enumerate(fields):
        x, u = line_profile(f)
        label = (
            labels[i] if labels is not None
            else (f"t={f.time:g}" if f.time is not None else f"#{i}")
        )
        ax.plot(
            x, u,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.4,
            label=label,
        )


def render_line_panel(
    fields: Sequence[Field2D],
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "u",
    ylim=None,
    figsize=(6.4, 3.4),
) -> plt.Figure:
    """Several 1-D profiles of the same field, overlaid on one axes.

    The line analogue of `render_panel`: one run, several times (or several
    parameter values) on a shared pair of axes. `title` should say what the
    reader is looking at, as for the image renderers.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_profiles(ax, fields, labels)
    _style_line_axes(ax, xlabel, ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def render_line_comparison(
    fields_a: Sequence[Field2D],
    fields_b: Sequence[Field2D],
    *,
    label_a: str,
    label_b: str,
    labels: Optional[Sequence[str]] = None,
    suptitle: str = "",
    xlabel: str = "x",
    ylabel: str = "u",
    ylim=None,
    annotate_a: Optional[str] = None,
    annotate_b: Optional[str] = None,
    figsize=(9.0, 3.6),
) -> plt.Figure:
    """Two 1-D runs side by side on **shared x and y axes** (A-vs-B lines).

    The line analogue of `render_comparison`, and the reason it shares axes
    is the same: this figure exists to answer "is run A different from run
    B", and that question is only answerable by eye if a millimetre means
    the same amplitude in both panels. `ylim` defaults to the union of both
    runs' data with a small margin, so neither panel is silently rescaled to
    flatter its own run.

    The same time (or parameter) in both panels gets the same colour *and*
    the same dash pattern, so the reader tracks a curve across the panels
    without consulting two legends -- only the left panel carries one.
    `annotate_a`/`annotate_b` place one short note inside a panel for a
    feature worth pointing at directly (e.g. "no wave train here").
    """
    import numpy as np

    fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    _draw_profiles(axa, fields_a, labels)
    _draw_profiles(axb, fields_b, labels)

    if ylim is None:
        both = np.concatenate(
            [f.data.ravel() for f in list(fields_a) + list(fields_b)]
        )
        lo, hi = float(both.min()), float(both.max())
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        ylim = (lo - pad, hi + pad)
    axa.set_ylim(*ylim)

    for ax, label in ((axa, label_a), (axb, label_b)):
        _style_line_axes(ax, xlabel, ylabel)
        ax.set_title(label, loc="left", fontsize=TITLE_FS)
    axb.set_ylabel("")

    for ax, note in ((axa, annotate_a), (axb, annotate_b)):
        if note:
            ax.annotate(
                note, xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=TICK_FS, color="0.35",
            )

    axa.legend(fontsize=TICK_FS, frameon=False, loc="upper right")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=TITLE_FS + 0.5)
    fig.tight_layout()
    return fig
