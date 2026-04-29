<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0001: Record architecture decisions

## Status

Accepted

## Context

OpenPFC is a research code with multiple layers (kernel, runtime, frontend) and evolving MPI / HeFFTe / GPU integration. Long-form docs can lag; **decisions** need a lightweight trail.

## Decision

We keep **ADRs** in `docs/adr/` as short markdown files: status, context, decision, consequences, and links to code or docs.

## Consequences

- Contributors can propose decisions in the same PR as code when appropriate.  
- Readers find “why” without spelunking every closed issue.  

## See also

- [`refactoring_roadmap.md`](../refactoring_roadmap.md) — phased refactors  
