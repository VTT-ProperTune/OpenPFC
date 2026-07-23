<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Architecture Decision Records (ADR)

Short, durable notes for **why** the codebase looks the way it does. They complement narrative docs ([`architecture.md`](../concepts/architecture.md), [`refactoring_roadmap.md`](../development/refactoring_roadmap.md)).

| ADR | Title |
|-----|--------|
| [0001-record-architecture-decisions.md](0001-record-architecture-decisions.md) | Why we keep ADRs |
| [0002-gradient-operators-fd-vs-spectral.md](0002-gradient-operators-fd-vs-spectral.md) | FD vs spectral operators (direction) |
| [0003-time-integrator-interface.md](0003-time-integrator-interface.md) | Time integrator interface contracts |
| [0004-execution-layer.md](0004-execution-layer.md) | Execution layer for 0.2 — minimal homegrown, no Kokkos |
| [0005-fft-interface.md](0005-fft-interface.md) | FFT interface — split host and device transforms |
| [0006-precision-policy.md](0006-precision-policy.md) | Precision policy — template on RealType, instantiate double |
| [0007-decomposition-splitter.md](0007-decomposition-splitter.md) | Decomposition splitter — in-repo, HeFFTe as FFT-only dependency |
| [0008-io-formats.md](0008-io-formats.md) | I/O and checkpoint formats for 0.2 |

New ADRs: copy the template sections from `0001`, use the next number, and add a row here.
