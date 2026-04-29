<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0002: Spatial operators — FD vs spectral (direction)

## Status

**Proposed / in progress** (implementation spans multiple releases)

## Context

- **Spectral** methods (FFT via HeFFTe) are the default backbone for shipped PFC-style models: linear operators in *k*-space, nonlinear terms in real space.  
- **Spectral** discretizations can be limiting for some boundary treatments, localized operator control, or formulations where global FFT assumptions are awkward.  
- **Finite-difference** stencils already integrate with the same **decomposition** and **halo** machinery ([`halo_exchange.md`](../halo_exchange.md)); kernel helpers live under `pfc::field::fd` ([`finite_difference.hpp`](../../include/openpfc/kernel/field/finite_difference.hpp)).

## Decision

Advance a **unified abstraction for spatial operators** (gradients, Laplacians, and related terms) so that, where supported, callers can select **spectral** versus **FD** evaluation consistently with domain decomposition and halo policy. Shipped spectral JSON apps may remain FFT-centric until wiring is complete; new and internal models can adopt the abstraction earlier.

## Consequences

- Documentation must describe **both** the current spectral-first reality and the **FD roadmap** ([`when_not_to_use_openpfc.md`](../when_not_to_use_openpfc.md)).  
- Performance and correctness tests must cover **both** paths where exposed.  
- [`refactoring_roadmap.md`](../refactoring_roadmap.md) may track concrete milestones.

## See also

- [`architecture.md`](../architecture.md) — spectral vs FD coexistence  
- [`science_numerics_limits.md`](../science_numerics_limits.md) — stability caveats  
