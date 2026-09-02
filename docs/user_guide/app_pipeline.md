<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Config-driven application pipeline (`App` → `Simulator`)

This page describes how a JSON or TOML file becomes a running simulation when you use `pfc::ui::App<YourModel>` (the pattern used by `apps/tungsten`, `apps/aluminumNew`, and several `examples/`). It ties together headers under `include/openpfc/frontend/ui/` and `simulation_wiring.hpp`.

For install and MPI setup, see [`INSTALL.md`](../../INSTALL.md). For shared config vocabulary, see [`configuration.md`](configuration.md).

## Big picture

```mermaid
flowchart LR
  subgraph load [Load]
    A[config.json / .toml]
  end
  subgraph stack [SimulationSession SpectralCPUStack]
    W[Domain]
    D[Decomposition]
    F[CPUFFT / HeFFTe]
    T[Time]
  end
  subgraph session [SpectralSimulationSession]
    M[ConcreteModel]
    S[Simulator]
  end
  subgraph wire [Wiring from JSON]
    RW[Results writers]
    IC[Initial conditions]
    BC[Boundary conditions]
    SM[simulator section]
  end
  A --> W --> D --> F --> T
  F --> M
  T --> M
  M --> S
  A --> wire
  wire --> S
```

- `make_simulation_session<SpectralCPUStack>` reads domain, time, `method`/`backend`, and `plan_options` (FFT) from the parsed document and constructs `pfc::sim::stacks::SpectralCPUStack` (Domain → Decomposition → CPUFFT). CPU plan options live in `spectral_fft_stack_factory.hpp` (`cpu_spectral_plan_options_from_json`, `cpu_fft_from_json_and_decomposition`). If `plan_options` omits `backend` but the document has a root-level `backend` string (same key as `from_json<fft::Backend>`), that value is merged into the plan slice for parsing; `backend: "cuda"` is rejected on this CPU-only path.
- `make_simulation_session<GPUSpectralStack<CUDASpace/HIPSpace>>` overlays the same JSON `plan_options` keys onto cuFFT / rocFFT defaults (`cuda_spectral_plan_options_from_json` / `hip_spectral_plan_options_from_json`) and passes them to `create_cuda` / `create_hip`. Tungsten and aluminum GPU ETD sessions use the same overlay. GPU-aware MPI, pencils, and reshape algorithm therefore apply to multi-rank device FFTs, not only the CPU stack.
- `SpectralSimulationSession` owns `SimulationSession<SpectralCPUStack>`, constructs `ConcreteModel(fft, world, comm)` (same `MPI_Comm` as the stack and simulator), then `Simulator(model, time, comm)`. `World` is stored by value so `Model`’s reference does not dangle. Custom models should forward the optional third `MPI_Comm` argument in their constructor (default `MPI_COMM_WORLD` keeps two-argument construction valid).
- `wire_simulator_from_settings` (on the session) calls `wire_simulator_and_runtime_from_json`, which attaches writers, `ICs`, `BCs`, and optional `simulator` subsection keys. The wiring APIs require **explicit** `FieldModifierCatalog` and `ResultsWriterCatalog` references (no default parameters). `App` passes `default_field_modifier_catalog()` when no override is set, and always passes `default_results_writer_catalog()` for the stock binary writers unless you change the call path.

## `App<ConcreteModel>::main()` order of operations

Implementation: `include/openpfc/frontend/ui/app.hpp` (settings load, MPI hints; `configure_json_driver_hooks` for `from_json` rank + NaN-check comm) and `include/openpfc/frontend/ui/app_json_run.hpp` (`JsonAppRun` — session through time loop).

**Optional, outside the library:** after loading the config file and **before** `App::main()`, application code may run `ParameterValidator` on `settings["model"]["params"]` (or your params subtree). OpenPFC does not call `ParameterValidator` automatically; ordering relative to `from_json` is described in [`parameter_validation.md`](parameter_validation.md#validation-vs-app-parsing-order).

| Step | What happens |
|------|----------------|
| 1 | Load settings from `argv[1]` via `load_settings_file` (JSON or TOML). |
| 2 | `SpectralSimulationSession::assemble(settings, comm, rank, nranks)` — builds stack + model + simulator. |
| 3 | If present, `from_json(settings["model"]["params"], model)` — fills model parameters after construction. |
| 4 | Profiling controller reads `[profiling]` / root keys as implemented in `app_profiling.hpp`. |
| 5 | `model.initialize(dt)` from session time. |
| 6 | Memory report (model + FFT allocations). |
| 7 | `wire_simulator_from_settings` — writers, ICs, BCs, `simulator` subsection (explicit modifier + results-writer catalogs; optional `set_field_modifier_catalog` on `App` only affects the modifier catalog passed in). |
| 8 | Time integration loop (`run_simulator_time_integration_loop` + `SimulatorIntegratorLoopEnv` in `app_integrator_loop.hpp`) + profiling finalize/export. |

Custom drivers can replicate subsets: build a `Simulator` yourself, then call `add_result_writers_from_json`, `add_initial_conditions_from_json`, `add_boundary_conditions_from_json`, and `apply_simulator_section_from_json` from `simulation_wiring.hpp` directly. Each helper requires the relevant catalog at the call site (`default_field_modifier_catalog()`, `default_results_writer_catalog()`, or your injected registries). Pass `JsonWiringContext{comm, mpi_rank, rank0}` for MPI metadata. `JsonWiringSession` bundles context with **both** catalogs for `wire_simulator_and_runtime_from_json(sim, time, settings, session)`.

## JSON sections consumed by the default spectral path

Exact keys vary slightly by app and schema version; always treat `apps/tungsten/inputs_json/` and `examples/fft_backend_selection.toml` as ground truth. Typical `top-level` usage:

| Section / keys | Handled by | Role |
|----------------|------------|------|
| `Lx`, `Ly`, `Lz`, `dx`, `dy`, `dz`, `origin`, … | `from_json` for `Domain` / `make_simulation_session` | Grid and physical extent. |
| `t0`, `t1`, `dt`, `saveat` | `Time` | Integration interval and output cadence. |
| `plan_options` | HeFFTe `plan_options` | FFT backend (`backend`), `use_gpu_aware`, `reshape_algorithm`, etc. Root `backend` is merged when `plan_options` omits it (CPU `App` path only; `cuda` is rejected there). |
| `model.name`, `model.params` | Your `ConcreteModel` + `from_json` into params in `JsonAppRun::apply_model_params_` (step 3) | Physics coefficients; optional `ParameterValidator` in your `main` targets the same subtree—see [`parameter_validation.md`](parameter_validation.md). |
| `fields`, `saveat` | `add_result_writers_from_json` | If `saveat > 0`, each `fields[]` entry with `name` and `data` path gets a writer from `ResultsWriterCatalog` (default `writer`: `"binary"` → `BinaryWriter`, MPI-IO). |
| `initial_conditions[]` | `add_initial_conditions_from_json` | Each entry has `type`, optional `target`, type-specific fields. |
| `boundary_conditions[]` | `add_boundary_conditions_from_json` | Same pattern: `type`, `target`, … |
| `simulator` | `apply_simulator_section_from_json` | Gen-1 overlay: optional `result_counter`, `increment`, `integrator.method`. Mutually exclusive with `restart_from`. |
| `checkpoint`, `restart_from` | `checkpoint_config_from_json` / `CheckpointService` | 0.2 restart: `checkpoint.every`, `checkpoint.directory`, `restart_from: <dir>`. See [`checkpoint_publish.md`](../development/checkpoint_publish.md). |
| `method`, `backend`, `fd_order` | `SessionSelection` / `make_simulation_session` | Stack pick (`spectral`/`fd` × `cpu`/`cuda`/`hip`). FD CPU JSON sessions: `json_fd_session.hpp`. |
| `profiling` | `AppProfilingController` | Export paths and regions; see [`performance_profiling.md`](../hpc/performance_profiling.md). |

TOML uses the same logical sections (e.g. `[plan_options]`).

## `SimulationContext` and field modifiers

When `Simulator` applies initial or boundary modifiers, it passes a `SimulationContext` (MPI communicator, rank-0 flag) together with the `Model`. Modifier authors should read `include/openpfc/kernel/simulation/simulation_context.hpp`. This is separate from JSON but central to why IC/BC code can perform rank-aware I/O.

## Registration of JSON `type` strings

Initial/boundary entries use `"type": "<name>"`. Those names are resolved via `FieldModifierCatalog` (`field_modifier_registry.hpp`). Applications call `register_field_modifier<MyModifier>("my_type")` before constructing `App`. See `examples/10_ui_register_ic.cpp` and shipped apps’ `main`.

## See also

| Topic | Where |
|--------|--------|
| Spectral `App` JSON/TOML key reference | [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md) |
| Layered architecture | [`architecture.md`](../concepts/architecture.md) |
| Main types and headers (`Model`, `App`, …) | [`class_tour.md`](../reference/class_tour.md) |
| Minimal out-of-tree `App` + JSON | [`tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md) |
| Validated `model.params` | [`parameter_validation.md`](parameter_validation.md) |
| FFT / `[plan_options]` examples | [`examples/fft_backend_selection.toml`](../../examples/fft_backend_selection.toml) |
| Results formats (binary, VTK, PNG) | [`io_results.md`](io_results.md) |
| CMake options | [`build_options.md`](../reference/build_options.md) |
| Extending models | [`extending_openpfc/README.md`](../extending_openpfc/README.md) |
