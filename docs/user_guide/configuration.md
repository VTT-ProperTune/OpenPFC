<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Configuration files (JSON / TOML)

Many programs use `pfc::ui::App<Model>` and accept a single configuration file (JSON or TOML) on the command line. The exact keys depend on the model and how validation is set up, but the same structural ideas recur across apps.

Use this page when you are trying to read an input file and understand what kind of thing each section controls. It is not the normative key reference; that lives in [the spectral App configuration reference](../reference/spectral_app_config_reference.md). This page is the bridge between a real JSON or TOML file and the runtime objects OpenPFC builds from it.

For the full lifecycle from configuration to `World`, FFT, `Simulator`, writers, initial conditions and boundary conditions, read [the App pipeline guide](app_pipeline.md). For output formats, continue with [the results I/O guide](io_results.md) and [the binary field layout reference](../reference/binary_field_io_spec.md). If you are building your own `App<Model>`, the practical tutorial is [the minimal custom App walkthrough](../tutorials/custom_app_minimal.md), with [parameter validation](parameter_validation.md) as the companion for `model.params`.

## Mental model

Most configuration files describe the same broad concerns. The domain or grid section defines sizes, spacing and origin. The model section names the physics and supplies parameters. Time controls start, end, timestep and output cadence. Fields describe order parameters and, often, paths for I/O. Initial and boundary conditions either appear directly or refer to registered modifiers. FFT settings appear as `plan_options` in TOML, or as equivalent keys in JSON, when the application exposes HeFFTe backend and communication choices.

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
