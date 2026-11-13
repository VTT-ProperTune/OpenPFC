<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0006: Precision policy — template on RealType, instantiate double

## Status

Accepted (2026-07-23).

## Context

The audit (§17.3) noted the codebase is effectively `double`-only today, while
the tungsten GPU model already templates on `RealType` (and a latent
`float`-instantiation bug surfaced in Pre-M0 PA when the sync hooks became
virtual). Retrofitting a precision parameter after the fact is the expensive
direction; introducing it up front is cheap.

## Decision

1. New 0.2 core types — `Field<T, MemorySpace>`, `SimulationState`, steppers,
   spatial operators — are **templated on `RealType`** (the scalar type),
   from their introduction (M1/M2/M6).
2. For 0.2 we **instantiate and test `double` only**. `float` (and mixed
   precision) must *compile* but are not validated; no baseline covers them.
3. Host mirrors and I/O paths must not assume a fixed scalar type: convert
   explicitly between `RealType` and any fixed-type sink (as Pre-M0 PA now does
   for the tungsten host mirror) rather than assuming `double`.

## Consequences

- The template parameter exists everywhere from day one, so a later `float` or
  mixed-precision effort is instantiation + validation work, not a refactor.
- CI and baselines (`tests/baselines/BASELINES.md`) cover `double`; a `float`
  claim requires new baselines and is out of scope for 0.2.
- Explicit-conversion discipline at type boundaries (mirrors, writers,
  checkpoint) is a review checklist item.

## Amendment (2026-11-13, #854 step 1)

The spectral ETD chain is now instantiated and tested in `float` as well:
`SpectralETDSystem` / `SpectralETDSession` take the field scalar from
`Physics::real_type` (`physics_real_t`, default `double`), `IHostFFT` and
`IDeviceFFT` carry both precisions, and `BasicSpectralCPUStack<Real>` /
`GPUSpectralStack<Space, Real>` bundle float fields. The k-space symbols
stay `double` functions narrowed on upload; field modifiers are served
through an explicit double scratch, `ResultsWriter` widens float views in
its base class, and checkpoint bundles pack float fields as float64 bricks
and narrow them on load (point 3 above), so the on-disk formats are
unchanged and float and double sessions restart from each other's bundles.
Validation is by tolerance against the double reference
(`[spectral_etd][float]`, the float cases of `CUDA_SpectralETD` /
`HIP_SpectralETD`), not bitwise. Shipped apps, the JSON precision switch
and recorded tolerance baselines remain open under #854; until they land,
`float` is validated for the library chain and its I/O, not yet for a
shipped session.
