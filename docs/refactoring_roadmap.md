<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Refactoring roadmap (architecture)

This document tracks planned and in-progress structural improvements discussed for OpenPFC: clearer layering, less duplication, and alignment with SOLID-style practices. It complements [`architecture.md`](architecture.md).

## Phase A — Communicator consistency (in progress)

Goal: `Model::is_rank0()` and `Simulator` / `SimulationContext` use the same `MPI_Comm` for “rank 0” semantics.

Done:

- Optional `MPI_Comm mpi_comm = MPI_COMM_WORLD` on `Model`; rank-0 via `mpi_comm_rank_is_zero` (same helper as `SimulationContext`); accessor `Model::mpi_comm()`.
- `SpectralSimulationSession` constructs `ConcreteModel(fft, world, m_stack.mpi_comm())` so JSON-driven apps align with the simulator communicator.
- Shipped / example models updated (`Aluminum`, `Tungsten` CPU/CUDA/HIP, `examples/10_ui_register_ic.cpp`).
- CUDA/HIP Tungsten: HeFFTe GPU FFT and rank/size queries use `Model::mpi_comm()` (no hardcoded `MPI_COMM_WORLD` in `set_cuda_fft` / `set_hip_fft` or constructors).

## Phase B — Simulator decomposition

Goal: Reduce responsibilities of `Simulator` (time orchestration vs modifier application vs results scheduling).

Done:

- `apply_field_modifier_list` in `simulator_field_modifiers_dispatch.hpp` — shared apply loop for IC and BC lists; `Simulator` delegates to it.

Planned steps:

- Optionally move writer map ownership behind a narrow interface for tests.

## Phase C — Unified config-driven stack (CPU / GPU)

Goal: One JSON → session pipeline for spectral runs, parameterized by FFT backend instead of CPU-only `SpectralCpuStack`.

Planned steps:

- Introduce a stack factory or templated `SpectralSimulationSession<Model, FftBackend>` (or type-erased FFT handle at the session boundary).
- Align `from_json` FFT backend selection with session construction (see [`app_pipeline.md`](app_pipeline.md)).

## Phase D — CMake library split (optional)

Goal: Enforce kernel vs frontend vs optional GPU objects at link time; faster incremental builds.

Planned steps:

- Split `openpfc` into e.g. `openpfc_core` + `openpfc_frontend` (+ optional CUDA/HIP object libs), with an umbrella `OpenPFC` target for compatibility.

## How to use this doc

Update the Phase A–D bullets when work lands or priorities change. Link relevant PRs or commits in project notes if desired (not required in-repo).
