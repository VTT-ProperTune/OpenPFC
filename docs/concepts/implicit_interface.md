<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Implicit interfaces on the Cartesian grid

The geometry layer sits on `Domain`. It does not allocate a mesh and it
does not change field storage. A node `(i, j, k)` is still
`origin + (i, j, k) * spacing`.

## Sign convention

`φ < 0` is inside, `φ > 0` is outside, and `φ = 0` is the interface.
`make_plane`, `make_sphere`, and `make_circle` store signed distances.
`make_ellipsoid` and `make_ellipse` store the algebraic polynomial

```text
(Rᵀ (x − c))ᵢ² / aᵢ² − 1
```

That polynomial is not a distance. `|φ|` must not be read as a length
unless `is_signed_distance` is true. The outward normal is `∇φ / ||∇φ||`.
A vanishing gradient is reported and is not replaced.

`R` is a proper rotation (orthonormal, determinant +1). Its columns are
the local principal axes in physical space. An ellipse rotation preserves
the z axis; a general ellipsoid rotation does not.

## Cells

`layout_from_domain` builds cells from the nodes that already exist. A
periodic axis has one cell per node. The last cell ends on the periodic
image of the origin. A non-periodic axis has one fewer cell than nodes.
An axis with a single node is inactive, so a 2D domain reports area.

φ is not tiled. `wrap_coordinate` is explicit.

## What a cut cell is

Analytic shapes use the exact minimum and maximum of φ on the cell. A
cell is interior only when the maximum is strictly negative, and exterior
only when the minimum is strictly positive. A sphere that sits inside one
cell is positive at every corner and is still cut.

`classify_general` is the contract for an arbitrary implicit function.
A Lipschitz constant can prove that a cell is uncut. Without one, an
unresolved leaf stays cut. Equal corner signs are not that proof.

## Quadrature

`measure_box` partitions a cell into inside and outside volumes. A plane
is the exact clip of the cell. A sphere or ellipsoid uses dyadic
subdivision and, on the smallest cells, the clip of the linear interpolant
of φ. That interpolant is not the curved surface. The two volumes sum to
the cell volume. A positive fraction below `sliver_fraction` is kept and
flagged.

A body that lies entirely inside one cell uses its analytic volume. The
analytic volume of a whole ellipsoid is not the exact volume of an
arbitrary cell that the ellipsoid only partly covers.

Quadratic moments of a completed plane-cut point set are moments of that
polyhedron. For a curved interface they are moments of the polyhedral
approximation and they move as `max_depth` increases.

## Device

`classify_on_device` writes one byte per cell. `evaluate_on_device` then
measures the compact cut list. Both call the host functions, so the
device result is the same classification and the same measure. Regular
cells stay on the structured grid.
