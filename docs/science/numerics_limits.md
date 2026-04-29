<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Science and numerics — practical limits

This page collects **honest** expectations: OpenPFC is a serious HPC framework, but **no** numerical method is universal. For code layering, read [`architecture.md`](../concepts/architecture.md); for spectral vs FD direction, read [`when_not_to_use_openpfc.md`](../when_not_to_use_openpfc.md) and [`adr/0002-gradient-operators-fd-vs-spectral.md`](../adr/0002-gradient-operators-fd-vs-spectral.md).

## Spectral semi-implicit style (typical shipped models)

- **Linear** operators are often treated **implicitly** in Fourier space; **nonlinear** terms are evaluated in **real space** and coupled through FFTs.  
- **Timestep** `dt` must respect the **explicit** pieces of your splitting; blowing up often means `dt` is too large for the nonlinear or stabilized explicit stages—not “MPI is broken.”  
- **Resolution** changes physics: interface widths, defect cores, and nucleation depend on grid spacing; do not compare runs at different resolutions without a convergence mindset.

## Finite differences (kernel helpers and examples)

- FD stencils need **halo exchange**; **in-place** vs **separated** halos interact with FFT layouts ([`halo_exchange.md`](../concepts/halo_exchange.md)).  
- FD and spectral operators can coexist in **principle**; user-facing **unified switching** for gradients/Laplacians is **still maturing** (see ADR 0002).

## What pretty pictures do not prove

- Visualizations (VTK, PNG) are **diagnostics**. They do not replace mesh/time convergence, mass/energy checks where applicable, or comparison to reference solutions.

## See also

- [`science_tungsten_quicklook.md`](tungsten_quicklook.md) — what the tungsten model encodes  
- [`science_cahn_hilliard_vs_allen_cahn.md`](cahn_hilliard_vs_allen_cahn.md) — example entry points  
