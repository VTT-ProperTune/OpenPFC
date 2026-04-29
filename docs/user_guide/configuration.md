<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Configuration files (JSON / TOML)

Many programs use `pfc::ui::App<Model>` and accept a single configuration file (JSON or TOML) on the command line. The exact keys depend on the model and how validation is set up, but the same structural ideas recur across apps.

Read this first for the full pipeline: [`app_pipeline.md`](app_pipeline.md) (how settings map to `World`, FFT, `Simulator`, writers, ICs, BCs). For output formats, see [`io_results.md`](io_results.md) and [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md). **Normative key tables** (world, time, `plan_options`, `fields`, …): [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md). To build your own CMake project around `App<Model>`, see [`tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md); for `model.params` validation, [`parameter_validation.md`](parameter_validation.md) and the root `README.md` (Configuration Validation). For a sequenced **run** track that ends with config-driven apps, see [`learning_paths.md`](../learning_paths.md).

## Mental model

1. Domain / grid — sizes, spacing, origin (sometimes nested under `domain`, sometimes top-level keys depending on app and parser version).
2. Model — `name` and `params` (physics coefficients); often validated at startup.
3. Time — start, end, timestep, output cadence (`saveat`, etc.).
4. Fields — named order parameters and optional file paths for I/O.
5. Initial / boundary conditions — declarative or references to registered modifiers.
6. `[plan_options]` (TOML) or equivalent — HeFFTe FFT backend and communication options.

For layering (what parses this vs what runs physics), see [`architecture.md`](../concepts/architecture.md).

## FFT and parallel layout

The annotated example `examples/fft_backend_selection.toml` documents:

- `[plan_options]` — `backend` (`fftw`, `cuda`, …), `reshape_algorithm`, `use_pencils`, `use_gpu_aware`, etc.

Copy patterns from that file into your own TOML or translate key names to JSON as your app expects.

## Full application examples

| App | Sample configs | Notes |
|-----|------------------|--------|
| Tungsten | [`apps/tungsten/inputs_json/`](../../apps/tungsten/inputs_json/README.md), `inputs_toml/` | Large validated parameter sets; JSON and TOML mirrors. |
| AluminumNew | [`apps/aluminumNew/aluminumNew.json`](../../apps/aluminumNew/aluminumNew.json), `aluminumNew.toml` | `App<Aluminum>` + registered modifiers. |

Examples that use JSON for demonstration include `examples/12_cahn_hilliard.cpp` and `examples/10_ui_register_ic.cpp` — inspect the source for the expected file shape.

## Profiling block

Runtime profiling output (JSON/HDF5) is configured in the same files for App-driven runs. See [`performance_profiling.md`](../hpc/performance_profiling.md) and [`profiling_export_schema.md`](../hpc/profiling_export_schema.md).

## Validation

Models can register parameter metadata (ranges, required keys). On failure you get a printed report instead of a silent wrong run — see the root [`README.md`](../../README.md) (“Configuration Validation”).

## See also

- [`quickstart.md`](../quickstart.md) — run an app with a stock input file  
- [`faq.md`](../faq.md) — paths and `find_package` issues  
- [`applications.md`](applications.md) — which binaries read which configs
