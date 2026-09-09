#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render OpenPFC field snapshots as publication-quality figures.

Three entry points, all returning a `matplotlib.figure.Figure` the caller
saves:

- `render_panel` -- one field, one axes, a colour bar.
- `render_montage` -- a time series as a row of panels sharing one colour
  bar, so morphology evolution is visible in a single figure.
- `render_comparison` -- two fields (typically two runs) side by side on a
  *shared* colour scale, for A-vs-B comparisons.

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
