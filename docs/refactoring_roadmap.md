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
- `ResultsWriterMap` alias in `results_writer.hpp`; `Simulator::results_writers()` const accessor; `write_results_for_registered_fields` takes `ResultsWriterMap` (named type for tests / tooling). Doxygen on `write_results` / dispatch + [`io_results.md`](io_results.md) call out the narrow test seam.
- Deprecated `pfc::get_field(Model&)` and `Simulator::get_field()` / `pfc::get_field(Simulator&)` removed; diffusion fixtures register `"default"` alongside `"density"` and drop `get_field()` overrides. `Model::get_field()` remains deprecated for out-of-tree subclasses.
- Field modifier catalog: header + [`extending_openpfc/README.md`](extending_openpfc/README.md) document singleton vs explicit-catalog DI; `App::set_field_modifier_catalog` forwards an explicit catalog into `wire_simulator_from_settings` ([`app_pipeline.md`](app_pipeline.md)).
- `app_spectral_run.hpp`: `SpectralJsonAppRun` owns the post-settings spectral pipeline (session → wire → integrate); `App` keeps settings I/O and pre-run logs.
- `simulator_integrator.hpp` / `simulator_queries.hpp`: post-class `Simulator` helpers (scheduled writes, integrator seam, `get_model` / `get_time` / …) split out of `simulator.hpp` for readability (SRP); single include of `simulator.hpp` remains the public entry point.
- `model_free_functions.hpp`: non-member `Model` API (`get_world`, `has_field`, `step`, …) split out of `model.hpp` (same include-once pattern as `simulator.hpp`).
- Deprecation hygiene: `DiscreteField` member `interpolate` equivalence test suppresses Clang/MSVC warnings; `Model::get_field()` Doxygen expanded for migration and out-of-tree overrides.

## Phase C — Unified config-driven stack (CPU / GPU)

Goal: One JSON → session pipeline for spectral runs, parameterized by FFT backend instead of CPU-only `SpectralCpuStack`.

Done (foundation):

- **`spectral_cpu_stack_detail.hpp`**: `cpu_spectral_plan_options_from_json` and `cpu_fft_from_json_and_decomposition` centralize JSON → HeFFTe CPU FFT construction; `SpectralCpuStack` calls these (extension point for a future GPU stack builder using the same JSON surface).
- **CPU spectral `backend` alignment:** `cpu_spectral_plan_options_from_json` merges a root-level `"backend"` into the `plan_options` object when the latter omits it; rejects `"cuda"` on this path (always `fft::CpuFft` / FFTW). See [`app_pipeline.md`](app_pipeline.md).
- **`spectral_fft_stack_factory.hpp`:** `merged_spectral_plan_options_json` (shared merge); `cuda_spectral_plan_options_from_json` / `hip_spectral_plan_options_from_json` apply the same HeFFTe JSON overlay as CPU but start from cuFFT / ROCm defaults (GPU integration tests and future GPU `App` paths).

Planned steps:

- Optional: templated `SpectralSimulationSession` or type-erased FFT at the session boundary so `App` can skip constructing a dummy `CpuFft` for GPU-only models. (Design note in `spectral_cpu_stack.hpp` Doxygen `@note`.)
- Documented interim policy (Doxygen): reuse the one `SpectralCpuStack` `CpuFft` for `Model(fft, world, comm)` when adding GPU drivers; use `spectral_fft_stack_factory.hpp` for cuFFT/ROCm plan JSON only—no second throwaway CPU FFT in app code.

### Phase C spike (time-boxed exploration)

Purpose: validate a **single JSON document** driving either CPU or GPU spectral stacks without committing to a full `App` rewrite.

**Spike scope (1–2 weeks of prototyping, not merge criteria by itself):**

- Build a throwaway or feature-flagged **“spectral session”** type that owns `World`, `Decomposition`, and an FFT handle produced either from `cpu_fft_from_json_and_decomposition` or from the GPU plan builders in `spectral_fft_stack_factory.hpp`, using **`merged_spectral_plan_options_json`** so root `backend` and `plan_options` behave like today’s CPU path.
- Wire **`Time`** and a **minimal `Model` stub** (existing mock or smallest example model) through the same **`wire_simulator_and_runtime_from_json`** entry points to prove IC/BC/result wiring does not depend on `fft::CpuFft` specifically.
- Measure **what must become type-erased** at the session boundary (e.g. `IFFT &` vs concrete `CpuFft`) and list **API breaks** for shipped apps if `SpectralSimulationSession` were templated on FFT type.

**Exit criteria for closing the spike (documentation-only deliverable is OK):**

- Short decision: **templated session** vs **type-erased FFT interface** vs **defer** until a GPU-first `App` is scheduled.
- List of **must-keep JSON keys** and **test gaps** (MPI rank, GPU-aware MPI, VTK writers) before any production merge.

## Phase E — Wiring and driver ergonomics

Goal: Fewer repeated parameters at JSON → `Simulator` boundaries; clearer seams for custom drivers and tests.

**Layering:** Confirmed `include/openpfc/kernel` / `src/openpfc/kernel` do not include `openpfc/frontend` headers (see [`architecture.md`](architecture.md) *Include audit*).

Done:

- `JsonWiringContext` (`simulation_wiring_context.hpp`): bundles `MPI_Comm`, `mpi_rank`, and `rank0` for `add_result_writers_from_json`, `add_initial_conditions_from_json`, `add_boundary_conditions_from_json`, and `wire_simulator_and_runtime_from_json`. Legacy `(comm, rank, rank0)` overloads forward to the context form; `SpectralSimulationSession` uses the context overload.
- `configure_spectral_json_driver_hooks` (`spectral_json_driver_hooks.hpp`): one call sets `from_json` log rank and default NaN-check communicator; `App::main` uses it instead of duplicating globals.
- `write_scheduled_simulator_results(Simulator&)` in `simulator.hpp`: extracted from `Simulator::write_results()` so scheduled writes + counter bump live in one free-function seam ([`io_results.md`](io_results.md)).
- `results_writer_catalog.hpp` + optional `fields[].writer` string: `add_result_writers_from_json` resolves writers through `ResultsWriterCatalog` (default `binary`); inject a custom catalog at the wiring call site for tests and app-specific formats ([`app_pipeline.md`](app_pipeline.md)).
- **`errors.hpp` split:** `errors_config_format.hpp` (JSON field messages + `get_json_value_string`) and `errors_field_modifiers.hpp` (unknown modifier type + `list_valid_field_modifiers`); `errors.hpp` remains an umbrella include. `from_json_world_time.hpp` includes only the format header; `field_modifier_registry.hpp` includes only the modifier header.

## Backlog — larger SOLID-oriented refactors

High impact, not tied to a single PR; pick by maintenance pain.

### Suggested PR-scale moves (free-function & data-centric API)

Aligned with [**laboratory, not fortress**](architecture.md#design-ethos-laboratory-not-fortress) and the [styleguide API shape](styleguide.md#api-shape-free-functions-and-data-centric-types): keep `virtual` boundaries thin; push mechanics to namespaced free functions.

1. **Examples + apps:** mechanical pass replacing `model.get_world()` / `get_fft()` member spellings with `pfc::get_world(model)` / `pfc::get_fft(model)` (and simulator analogs) in touched files — high visibility, low risk.
2. **Shipped models (Tungsten, Aluminum, diffusion fixtures):** extract `initialize` / `step` internals into **`namespace …::`** free functions; leave `Model::step` as a one-line forwarder (easier testing and profiling).
3. **`Time`:** add small free wrappers (`pfc::time::…` or `pfc::` overloads) mirroring hot members (`next`, `done`, …) where it improves consistency with `Model` / `Simulator` free APIs.
4. **`errors.hpp`:** split by concern + prefer free `format_*` / `make_*` helpers so parsers do not pull unrelated types.
5. **GPU runtime (`runtime/cuda` vs `runtime/hip`):** deduplicate with **`runtime/common`** free helpers (plan/layout/device buffer) instead of parallel class hierarchies.
6. **JSON wiring:** extend catalog/factory patterns (already: field modifiers, results writers) for any remaining `if (type == …)` branches in wiring.
7. **`SpectralCpuStack` / session:** optional free `assemble_*` returning plain structs + explicit `wire_*` free functions for drivers that skip `App` (clearer data flow than only member methods).
8. **Integrator loop:** narrow `run_simulator_time_integration_loop` inputs to structs + free functions (less hidden state than callbacks on opaque objects).
9. **Tests:** shared **`tests/fixtures/`** free factories (`make_world`, `make_mock_model`, …) to avoid 40-line setup blocks repeating OO construction patterns.
10. **Include hygiene:** document + optionally CI-check “minimal includes” (`openpfc_minimal.hpp` + domain headers) so new code does not re-expand umbrella dependencies.

- **Gradient / spatial-operator abstraction:** unify spectral (FFT) and finite-difference evaluation of gradients and related operators where supported; track [`adr/0002-gradient-operators-fd-vs-spectral.md`](adr/0002-gradient-operators-fd-vs-spectral.md) and [`when_not_to_use_openpfc.md`](when_not_to_use_openpfc.md).
- **Simulator:** If orchestration grows again, consider named collaborators (e.g. explicit IC/BC pipeline type vs results scheduling) on top of existing `*_dispatch.hpp` helpers.
- **Model:** Narrower test- and tool-facing facades around field registry / world access (interface segregation) without a monolithic `Model` rewrite.
- **Multi-backend apps:** Share physics and parameters across Tungsten (and similar) CPU/CUDA/HIP; keep only execution and FFT device setup separate.
- **Configuration:** One story for `ParameterValidator`, JSON `from_json`, and docs for `model.params` so validation behavior matches reader expectations.

## Phase D — CMake library split

Goal: Enforce kernel vs frontend vs optional GPU objects at link time; faster incremental builds.

Done:

- **`openpfc_kernel_obj`** and **`openpfc_frontend_obj`** are `OBJECT` libraries; **`openpfc`** is built from their objects (same installed `libopenpfc` / `OpenPFC::openpfc` as before). Optional CUDA/HIP FFT `.cpp` files stay in the kernel object list when enabled.

## How to use this doc

Update the Phase A–E bullets and **Backlog** when work lands or priorities change. Link relevant PRs or commits in project notes if desired (not required in-repo).
