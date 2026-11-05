<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Compact periodic grain interface

`build_grain_interface` names every cell of one periodic Cartesian grid
with a single integer grain id. The id is the nearest seed. A tie keeps
the smaller seed index. `assign_periodic_voronoi` searches the 27 periodic
images. `assign_periodic_voronoi_min_image` uses the separated min-image
formula. The two maps are the geometry check.

Cells whose 26-neighbourhood contains another grain form the band. A site
records the owner, the other material, the number of distinct ids, a local
neighbour-direction estimate, and a positive half-neighbour spacing. When the
neighbour votes cancel, the normal estimate is zero rather than an invented
direction. These quantities are **not** an exact interface normal or a true
signed-distance solution. Two ids are a face, three a triple line, and more
than three a higher junction under this local 26-neighbour classification.
There is no signed-distance field per grain, and the band does not move.

`compact_storage_bytes` is the host grain-id field, the band, and the
per-grain totals. `dense_level_set_bytes` is the rejected layout of one
full-domain level set per grain. `resident_device_bytes` is the device
grain-id field, the band, the per-grain counts, and one mismatch word.
`GrainInterfaceCuda::measure` uploads that layout, recounts the cells, and
checks each band site against the resident ids.

The coefficients are code units. The single-rank reference uses `int` band
indices and explicitly rejects grids whose total cell count exceeds
`INT_MAX`; this is **not** a distributed large-domain grain-network solver.
Grain-boundary kinetics are out of scope.
HIP was not run.
