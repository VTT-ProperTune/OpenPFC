<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# `vlasov_maxwell` — 1D2V electromagnetic Vlasov–Maxwell

A collisionless kinetic distribution on a three-dimensional phase-space grid
`(x, v_x, v_y)`, coupled self-consistently to Maxwell. One binary and one
stepper; `--case` selects the validation rung. The electrostatic
Vlasov–Poisson reduction is a runtime specialisation of the same code, not a
second implementation.

This is the kinetic capstone of issue #84. The report chapter
(`docs/report/17_vlasov_maxwell.qmd`) presents the full Vlasov–Maxwell system
before deriving the 1D2V reduction. GPU measurements live in
[`docs/hpc/vlasov_gpu.md`](../../docs/hpc/vlasov_gpu.md).

| | |
|---|---|
| Unknown | \(f_s(x, v_x, v_y, t)\) plus \(E_x, E_y, B_z\) |
| Domain | one OpenPFC 3-D `Domain`; decompose on \(v_y\) only |
| BCs | periodic in \(x\); zero-inflow in velocity |
| Time | Strang split: exact spectral shift in \(x\), high-order semi-Lagrangian in velocity, exact ETD light wave |
| Backends | CPU; HIP via `--device=hip` (parity and cost binaries too) |

## Binaries

```bash
# Stage 2: Landau damping. Unknown keys are rejected.
vlasov_run --case=landau --summary=results/landau.csv

# Stage 4: Weibel. The growth rate is compared to a dispersion root solved
# at run time, not a remembered number.
vlasov_run --case=weibel --summary=results/weibel.csv

# HIP science path (ROCm build)
vlasov_run --case=landau --device=hip
```

| binary | role |
|---|---|
| `vlasov_run` | science driver; `--case=wave\|landau\|twostream\|gyro\|weibel\|filament` |
| `vlasov_hip_parity` | CPU against GPU, both steppers in one process |
| `vlasov_hip_cost` | per-phase timing that decided what to port |

CSV is append-only. Every sample carries particle number, energy parts,
entropy, \(\min f\), Gauss residual (absolute and relative), and
velocity-boundary occupancy.

## What the tests assert

- `test_fields`: vacuum \(\omega = k\) to round-off, charge-conserving Gauss,
  moment deposition, boundary occupancy.
- `test_transport`: exact \(x\)-shift, interpolation order, 1-vs-N rank
  `advect_vy`, halo overflow is an error.
- `test_diagnostics`: estimators on synthetic series with known answers,
  including the biased-window failure modes they exist to catch.

A grid that does not resolve the velocity spacing is refused
(`require_resolved_spacing`); that is not the same check as the tail
occupancy. The transverse Weibel relation used as the Stage-4 oracle is
**derived in this repository**, not quoted — see
`openpfc_apps/plasma_dispersion.hpp`.
