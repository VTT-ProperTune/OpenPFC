<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0004: Execution layer for 0.2 — minimal homegrown, no Kokkos

## Status

Accepted (2026-07-23). Supersedes the "ease later Kokkos adoption" rationale in
`kernel/execution/view.hpp`.

## Context

The audit (`OPENPFC_ARCHITECTURE_AUDIT.md` §6, §17.1) found two parallel
execution abstractions:

- a **Kokkos-facsimile** in `include/openpfc/kernel/execution/` (`View`,
  memory/execution spaces, `RangePolicy`/`MDRangePolicy`, `parallel_for`,
  `deep_copy`, `create_mirror`), whose device `parallel_for` was a serial host
  loop (Pre-M0 PB turned that into a compile error) and which **no application
  uses**; and
- the **real device layer** — `DataBuffer<Tag,T>`, hand-written CUDA/HIP
  kernels, device halo exchangers — which every GPU app actually uses.

The facsimile's only stated benefit was making a future switch to Kokkos
mechanical. That benefit is preserved by *deleting* the half-implementation
rather than maintaining it, and the hard part (halo/MPI interplay) is homegrown
either way.

## Decision

1. For the entire **0.2 series**, OpenPFC keeps a **minimal homegrown execution
   layer**: `DataBuffer<MemorySpace,T>` as the single storage primitive plus a
   single-sourced device kernel layer (M3). **Kokkos is not adopted in 0.2.**
2. The Kokkos-facsimile above `DataBuffer` (`View`, `parallel_for`, policies,
   `create_mirror`, layout/memory-trait scaffolding, and the vendor
   `view_*`/`parallel_*`/`execution_space_*` headers) is **removed in M3**.
   `DataBuffer`, the memory-space tags, and `deep_copy` on buffers survive.
3. Kokkos adoption is **deferred to a possible 0.3**, to be reconsidered only if
   kernel diversity grows beyond stencil-shaped loops (reductions, scans,
   irregular/unstructured access) where a mature backend earns its dependency
   cost alongside HeFFTe.

## Consequences

- M3 (single-source GPU runtime) proceeds against `DataBuffer` + a vendor shim,
  not against a portability framework.
- No new dependency is added; build/packaging surface stays as audited.
- Device `parallel_for`/`parallel_reduce`/`View` are **not** part of the public
  0.2 API; models use `DataBuffer` and the runtime kernels.
- If 0.3 adopts Kokkos, `DataBuffer` is the seam to replace with `Kokkos::View`;
  this ADR is revisited then.
