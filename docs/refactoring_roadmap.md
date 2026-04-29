<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Refactoring roadmap (architecture)

This document tracks planned and in-progress structural improvements discussed for OpenPFC: clearer layering, less duplication, and alignment with SOLID-style practices. It complements [`architecture.md`](architecture.md).

## Phase A — Communicator consistency

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
- `try_push_field_modifier_with_model_check` and message constants in `simulator_modifier_registration.hpp` — IC/BC registration rules and warning strings in one place; `Simulator` uses them instead of a private duplicate loop.
- `simulation_wiring_conditions.hpp`: `detail::wire_field_modifiers_from_json_array` — one implementation for parsing `initial_conditions` / `boundary_conditions` JSON arrays (inject `add_initial_conditions` vs `add_boundary_conditions` via callback).
- Tungsten CUDA/HIP VTK integration tests call `add_initial_conditions_from_json` / `add_boundary_conditions_from_json` instead of duplicating factory loops; removed accidental double `apply_initial_conditions()` before the first step.

Planned steps:

- Optionally move writer map ownership behind a narrow interface for tests.

## Phase C — Unified config-driven stack (CPU / GPU)

Goal: One JSON → session pipeline for spectral runs, parameterized by FFT backend instead of CPU-only `SpectralCpuStack`.

Done (foundation):

- **`spectral_cpu_stack_detail.hpp`**: `cpu_spectral_plan_options_from_json` and `cpu_fft_from_json_and_decomposition` centralize JSON → HeFFTe CPU FFT construction; `SpectralCpuStack` calls these (extension point for a future GPU stack builder using the same JSON surface).

Planned steps:

- Introduce a stack factory or templated `SpectralSimulationSession<Model, FftBackend>` (or type-erased FFT handle at the session boundary).
- Align `from_json` FFT backend selection with session construction (see [`app_pipeline.md`](app_pipeline.md)).

## Phase D — CMake library split

Goal: Enforce kernel vs frontend vs optional GPU objects at link time; faster incremental builds.

Done:

- **`openpfc_kernel_obj`** and **`openpfc_frontend_obj`** are `OBJECT` libraries; **`openpfc`** is built from their objects (same installed `libopenpfc` / `OpenPFC::openpfc` as before). Optional CUDA/HIP FFT `.cpp` files stay in the kernel object list when enabled.

## How to use this doc

Update the Phase A–D bullets when work lands or priorities change. Link relevant PRs or commits in project notes if desired (not required in-repo).
