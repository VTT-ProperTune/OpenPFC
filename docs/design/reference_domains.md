<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Reference domain and physical map

A geometry operator takes two objects.

- The reference domain is the chart the series already uses. A periodic
  Cartesian cell remains `Domain` plus `AffineMap`. A filled cylinder,
  disk, or ball remains its own series type. These types are not cases of
  a shape enum, and `Domain` is not extended with one.
- The physical map is `AffineMap`, `x = A ξ + b`. An identity Jacobian
  keeps the embedding operator. Any other constant nonsingular map uses
  that chart's inverse metric.

`sample_reference` is the scalar reading shared by the three filled
reference series. It returns the physical Laplacian at one station, the
integral in the reference weight, and the physical measure. The caller
passes the series it holds. Cylinder, disk, and ball each keep the
transform strategy from their own implementation: smooth powers of `r²`
for the cylinder and the disk, and frozen solid harmonics for the ball.
None of those strategies replaces `1/r` by `1/(r+ε)`.

The periodic Cartesian Laplacian stays `laplacian_symbol`. Identity calls
`k_laplacian_value`. Existing spectral and finite-difference applications
are not routed through the filled-domain series.

Boundary conditions are not part of this layer. The boundary-condition
roadmap remains separate. This header carries no material model.
