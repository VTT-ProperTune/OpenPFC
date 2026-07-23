<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0007: Decomposition splitter — in-repo, HeFFTe as an FFT-only dependency

## Status

Accepted (2026-07-23). Scheduled for M4.

## Context

The audit (§9, §17.4) found `Decomposition` delegates the domain split to
`heffte::split_world` / `heffte::proc_setup_min_surface`, so the decomposition
translation unit links HeFFTe even for finite-difference-only use, and the
x-fastest rank↔box ordering that `get_neighbor_rank` assumes is an implicit
HeFFTe contract (Pre-M0 PI added a construction-time assertion for it). FD is
meant to be a first-class method that should not require an FFT library.

## Decision

In M4, replace the HeFFTe calls in `src/openpfc/kernel/decomposition/` with a
small **in-repo min-surface brick splitter** behind the same `Decomposition`
API, so:

1. HeFFTe becomes purely an **FFT dependency**; an FD-only build links no HeFFTe.
2. The in-repo splitter enumerates subdomain boxes in the **x-fastest rank
   order** that `get_neighbor_rank` and the halo machinery require, by
   construction. The Pre-M0 PI ordering assertion is retargeted to the new
   splitter as its own invariant.
3. The splitter is validated against recorded `heffte::split_world` output for a
   matrix of (grid, ranks) cases (M4 test), so the FFT decomposition is
   unchanged where HeFFTe is still used.

## Consequences

- FD apps (kobayashi, heat3d, wave2d, allen_cahn) can build and run without a
  GPU/CPU FFT library.
- The spectral path is unaffected: HeFFTe still plans and executes transforms;
  only the ownership-map computation moves in-repo.
- One more piece of "HeFFTe is an implementation detail behind `Box3i`" (the
  audit's include-hygiene strength) is realized.
- Until M4 lands, the Pre-M0 PI assertion guards the HeFFTe ordering invariant.
