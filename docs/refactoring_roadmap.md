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
- Tungsten CPU/CUDA/HIP: NaN check macros use `CHECK_AND_ABORT_IF_NANS_MPI(..., mpi_comm())` so `MPI_Abort` and rank reporting match the model’s communicator.
- `CHECK_AND_ABORT_IF_NAN` / `CHECK_AND_ABORT_IF_NANS` use `default_nan_check_mpi_comm()`; `App::main` calls `set_default_nan_check_mpi_comm(m_comm)` so the default matches non-world app communicators. `pfc::mpi::get_rank()` / `get_size()` without a communicator are documented as deprecated (world-only).

## Phase B — Simulator decomposition

Goal: Reduce responsibilities of `Simulator` (time orchestration vs modifier application vs results scheduling) and keep `App` thin versus the spectral JSON run pipeline.

Done:

- `apply_field_modifier_list` in `simulator_field_modifiers_dispatch.hpp` — shared apply loop for IC and BC lists; `Simulator` delegates to it.
- `try_push_field_modifier_with_model_check` and message constants in `simulator_modifier_registration.hpp` — IC/BC registration rules and warning strings in one place; `Simulator` uses them instead of a private duplicate loop.
- `simulation_wiring_conditions.hpp`: `detail::wire_field_modifiers_from_json_array` — one implementation for parsing `initial_conditions` / `boundary_conditions` JSON arrays (inject `add_initial_conditions` vs `add_boundary_conditions` via callback).
- Tungsten CUDA/HIP VTK integration tests call `add_initial_conditions_from_json` / `add_boundary_conditions_from_json` instead of duplicating factory loops; removed accidental double `apply_initial_conditions()` before the first step.
- `from_json.hpp`: `set_from_json_log_rank` / `get_from_json_log_rank` replace static loggers fixed at rank `-1`; `App::main` sets the rank so FFT / HeFFTe parse diagnostics align with other MPI-aware logs. Split into `from_json_*` headers under the same umbrella; GPU stack factory includes `from_json_heffte.hpp` only.
- `ResultsWriterMap` alias in `results_writer.hpp`; `Simulator::results_writers()` const accessor; `write_results_for_registered_fields` takes `ResultsWriterMap` (named type for tests / tooling).
- Deprecated `pfc::get_field(Model&)` and `Simulator::get_field()` / `pfc::get_field(Simulator&)` removed; diffusion fixtures register `"default"` alongside `"density"` and drop `get_field()` overrides. `Model::get_field()` remains deprecated for out-of-tree subclasses.
- Field modifier catalog: header + [`extending_openpfc/README.md`](extending_openpfc/README.md) document singleton vs explicit-catalog DI; `App::set_field_modifier_catalog` forwards an explicit catalog into `wire_simulator_from_settings` ([`app_pipeline.md`](app_pipeline.md)).
- `app_spectral_run.hpp`: `SpectralJsonAppRun` owns the post-settings spectral pipeline (session → wire → integrate); `App` keeps settings I/O and pre-run logs.
- Deprecation hygiene: `DiscreteField` member `interpolate` equivalence test suppresses Clang/MSVC warnings; `Model::get_field()` Doxygen expanded for migration and out-of-tree overrides.

## Phase C — Unified config-driven stack (CPU / GPU)

Goal: One JSON → session pipeline for spectral runs, parameterized by FFT backend instead of CPU-only `SpectralCpuStack`.

Done (foundation):

- **`spectral_cpu_stack_detail.hpp`**: `cpu_spectral_plan_options_from_json` and `cpu_fft_from_json_and_decomposition` centralize JSON → HeFFTe CPU FFT construction; `SpectralCpuStack` calls these (extension point for a future GPU stack builder using the same JSON surface).
- **CPU spectral `backend` alignment:** `cpu_spectral_plan_options_from_json` merges a root-level `"backend"` into the `plan_options` object when the latter omits it; rejects `"cuda"` on this path (always `fft::CpuFft` / FFTW). See [`app_pipeline.md`](app_pipeline.md).
- **`spectral_fft_stack_factory.hpp`:** `merged_spectral_plan_options_json` (shared merge); `cuda_spectral_plan_options_from_json` / `hip_spectral_plan_options_from_json` apply the same HeFFTe JSON overlay as CPU but start from cuFFT / ROCm defaults (GPU integration tests and future GPU `App` paths).

Planned steps:

- Optional: templated `SpectralSimulationSession` or type-erased FFT at the session boundary so `App` can skip constructing a dummy `CpuFft` for GPU-only models.

## Phase D — CMake library split

Goal: Enforce kernel vs frontend vs optional GPU objects at link time; faster incremental builds.

Done:

- **`openpfc_kernel_obj`** and **`openpfc_frontend_obj`** are `OBJECT` libraries; **`openpfc`** is built from their objects (same installed `libopenpfc` / `OpenPFC::openpfc` as before). Optional CUDA/HIP FFT `.cpp` files stay in the kernel object list when enabled.

## How to use this doc

Update the Phase A–D bullets when work lands or priorities change. Link relevant PRs or commits in project notes if desired (not required in-repo).
