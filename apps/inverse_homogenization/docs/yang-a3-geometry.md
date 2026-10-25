<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Yang 2015 design A3 geometry note (OpenPFC #43)

Primary source: Yang, Harrysson, West & Cormier, *Int. J. Solids
Struct.* 69–70 (2015) 475–490,
DOI [10.1016/j.ijsolstr.2015.05.005](https://doi.org/10.1016/j.ijsolstr.2015.05.005).

Li Yang's dissertation §3.2.1 is used only to confirm unit-cell
topology (alternating 2-D layers, half-cell shift). It is not a
parameter source.

Wang, Li & Ma, *Mater. Des.* 99 (2016) 467–476, eq. (4) is an
independent density oracle for the same topology. It is not used to
change $H$, $L$, $\theta$, or $t$.

## Frozen A3 numbers (Yang Table 1)

Verified from the primary paper before any OpenPFC Poisson evaluation:

| symbol | value | role |
|--------|-------|------|
| $\theta$ | $45^\circ$ | angle between a **vertical** strut and a **re-entrant** strut |
| $H$ | 7.74 mm | vertical strut length |
| $L$ | 3.78 mm | re-entrant (oblique) strut length |
| $H/L$ | 2.05 | Yang reports 2.05; $7.74/3.78 = 2.0476\ldots$ |
| $t$ | 0.80 mm | square-section side |
| $\bar\rho$ | 23.30 % | published relative density |
| $\nu_{zx}$ | $-1.90$ | theoretical sign/mechanism; **not** a voxel fit target |

Constraints used by the same topology (Wang): $2 L \cos\theta < H$
and $t < L \sin\theta$. For A3 both hold:
$2 L \cos 45^\circ = 5.346 < 7.74$, $t = 0.80 < 2.673$.

## Unit-cell box (Yang §4.1.2)

The geometrical bounding box of the unit cell is

$$
2(H - L\cos\theta) \times 2 L \sin\theta \times 2 L \sin\theta.
$$

For A3 that is $L_x = L_y = 5.346\,\mathrm{mm}$,
$L_z = 10.134\,\mathrm{mm}$, $L_z/L_x = 1.896$.

The first implementation used $L_z = H + 2 L\cos\theta = 13.086\,\mathrm{mm}$
(outward convex V's, one story). That box is **not** Yang's.

## Topology

- Build the 3-D network from 2-D re-entrant bowties in the $x z$ and
  $y z$ planes.
- Re-entrant V's fold **inward** (waist between the ends of $H$).
- Adjacent parallel layers are shifted by half a cell in-plane
  ($L_x/2$ or $L_y/2$).
- Each 2-D layer family has two $z$-stories offset by
  $H-2L\cos\theta$, so a layer contains a column of $H$ members
  (the 2-D honeycomb column). That is not the #36 surrogate of
  through-going pillars on every half-lattice point.
- Verticals of adjacent layers are **not** collinear. A through-going
  pillar along the full $L_z$ at one $(x,y)$ is a different
  surrogate (OpenPFC #36).
- Wang's unit-cell drawing counts 4 verticals and 16 obliques. After
  periodic identification of opposite faces the unique skeleton has 2
  $H$ columns (one per story) and 16 $L$ members.

## Cross section

Yang: "thickness of the strut cross section $t$". Wang (same
topology) uses a square of side $t$ on every member. Occupancy is
the square prism of half-width $t/2$ in the local frame of the
segment, **not** the Euclidean tube $\sqrt{d^2} \le t/2$.

## Density oracles (before elasticity)

- Published Table 1: $0.233$.
- Wang eq. (4) overlap-corrected: $\approx 0.165$ for A3.
- Predeclared voxel gate, frozen before Poisson:
  $\varphi \in [0.113, 0.353]$ (half to 1.5 times published).
- Do **not** thicken $t$ to hit 0.233.

The 0.089 solid fraction of job 22159982 is the first skeleton
(wrong $L_z$, one story, cylinders). It is not a faithful
Yang-model failure.

## Resolutions (frozen by $t/L_x$, not by Poisson)

$t/L_x = 0.150$. 32³ gives $\approx 4.8$ voxels across $t$
(too coarse). Pair:

- $64 \times 64 \times 121$ ($\approx 9.6$ voxels across $t$)
- $96 \times 96 \times 182$

$N_z = \mathrm{round}(N_x L_z/L_x)$.
