<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Spectral `App` configuration reference (JSON / TOML)

This page lists the **JSON / TOML surface** consumed by the default **CPU spectral** stack (`SpectralCpuStack` → `SpectralSimulationSession` → `Simulator`) when you use `pfc::ui::App<Model>` (e.g. `apps/tungsten`, `apps/aluminumNew`). **Apps may add keys** or nest objects; treat shipped inputs as ground truth when in doubt.

**Pipeline order:** [`app_pipeline.md`](../user_guide/app_pipeline.md). **Mental model:** [`configuration.md`](../user_guide/configuration.md).

## World (grid)

Parsed by `from_json<World>(settings)` — supports **flat** keys on the root object **or** nested under `"domain"`.

| Key(s) | Type | Meaning |
|--------|------|---------|
| `Lx`, `Ly`, `Lz` | integer | Grid point counts |
| `dx`, `dy`, `dz` | number | Spacing |
| `origin` (or `origo`) | string | `"center"` or `"corner"` — sets physical origin convention |

**Examples:** top-level keys in some snippets; `tungsten` uses a `"domain"` object with the same logical fields ([`apps/tungsten/inputs_json/`](../../apps/tungsten/inputs_json/README.md)).

## Time stepping

Parsed by `from_json<Time>(settings)` — supports **flat** keys or nested under `"timestepping"`.

| Key | Type | Meaning |
|-----|------|---------|
| `t0`, `t1` | number | Integration window |
| `dt` | number | Step size |
| `saveat` | number | Output cadence (also gates binary writers in JSON wiring; `≤ 0` disables periodic result writes in the default helper) |

## FFT (`plan_options`)

Object key `"plan_options"` (JSON) or `[plan_options]` (TOML). Passed to HeFFTe plan construction on the CPU spectral path. **`backend: "cuda"`** is rejected for `SpectralCpuStack` / `CpuFft` — use CPU `fftw` here or a GPU-specific app driver.

**Annotated reference file:** [`examples/fft_backend_selection.toml`](../../examples/fft_backend_selection.toml). **Tutorial:** [`tutorials/fft_heffte_plan_options.md`](../tutorials/fft_heffte_plan_options.md).

## Model

| Key | Type | Meaning |
|-----|------|---------|
| `model.name` | string | Conventionally matches the app’s model registration |
| `model.params` | object | Model-specific; optional `from_json` into your C++ params type |

Validation is **model-dependent** (see [`parameter_validation.md`](../user_guide/parameter_validation.md)).

## Result writers (binary)

When `saveat > 0` and `fields` is present, `add_result_writers_from_json` registers **`BinaryWriter`** per entry:

| Key | Type | Meaning |
|-----|------|---------|
| `fields[].name` | string | Field identifier known to the simulator / model |
| `fields[].data` | string | Filename template; if it contains `%`, `printf`-style formatting with the simulator’s output **increment** is applied (see [`binary_field_io_spec.md`](binary_field_io_spec.md)) |

There is **no** VTK branch in this helper today.

## Initial / boundary conditions

| Key | Type | Meaning |
|-----|------|---------|
| `initial_conditions` | array | Objects with `type`, optional `target`, type-specific fields |
| `boundary_conditions` | array | Same pattern |

Modifier types must be **registered** in `main` before `App` runs (`register_field_modifier<…>`). See [`app_pipeline.md`](../user_guide/app_pipeline.md) and `examples/10_ui_register_ic.cpp`.

## Optional sections

| Key | Handled by | Notes |
|-----|------------|--------|
| `simulator` | `apply_simulator_section_from_json` | e.g. counter / increment for filenames |
| `profiling` | `AppProfilingController` | [`performance_profiling.md`](../hpc/performance_profiling.md) |

## TOML vs JSON

Same logical keys; TOML uses tables such as `[domain]`, `[timestepping]`, `[plan_options]`, `[model]`, `[model.params]`.

## See also

- [`app_pipeline.md`](../user_guide/app_pipeline.md) — `App::main` order  
- [`binary_field_io_spec.md`](binary_field_io_spec.md) — binary file layout  
- [`io_results.md`](../user_guide/io_results.md) — writers overview  
- [`learning_paths.md`](../learning_paths.md) — guided tracks  
