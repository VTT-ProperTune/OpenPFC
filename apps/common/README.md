<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Shared app helpers (`apps/common`)

Header-only INTERFACE library `openpfc_apps_common`. Per-app `RunConfig` and
usage strings stay in the app; this tree holds duplicated mechanics and
PFC directional-solidification BCs relocated from the kernel.

| Header | Role |
|--------|------|
| `openpfc_apps/cli.hpp` | `parse_or_print_usage`, `--flag` tokens, even FD-order check |
| `openpfc_apps/mpi_report.hpp` | MPI SUM/MAX reduce, timing lines, step-timing report |
| `openpfc_apps/gather.hpp` | pack owned z=0, rank-0 XY gather, ordered field stats |
| `openpfc_apps/fixed_bc.hpp` | sigmoid density band (tungsten / aluminum JSON App) |
| `openpfc_apps/moving_bc.hpp` | front-tracking band (same apps) |
| `openpfc_apps/solidification_bc_json.hpp` | JSON + `register_solidification_bcs()` |
| `openpfc_apps/microelasticity.hpp` | quasi-static eigenstrain elasticity: Fourier Green operator, Hu-Chen / Eyre-Milton polarisation fixed points, `f_el` and `d f_el/d phi` (host, periodic) |

`microelasticity.hpp` is the one header here that carries physics rather than
plumbing, so it has its own Catch2 suite (`tests/test_microelasticity.cpp`,
ctest name `apps-common-microelasticity`, ~5 s single rank). Its oracles are
closed forms — Eshelby's spherical inclusion, an exact dilatation identity,
and a finite difference of the re-converged elastic energy — not stored
baselines. It defaults to the Eyre-Milton accelerated fixed point, which
contracts as `(sqrt(r)-1)/(sqrt(r)+1)` in the solid/liquid stiffness ratio
rather than `(r-1)/(r+1)`; that matters because a liquid supports no shear,
so `r` is realistically 10–100. See the header's `@details` block for the
measured iteration tables, the recommended liquid stiffness, and what the
solver does not do.
