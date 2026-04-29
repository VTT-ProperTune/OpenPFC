<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Extending OpenPFC

OpenPFC is meant to be extended without editing the core library: you add models, initial/boundary behavior, writers, and (optionally) coordinate systems in your translation units and link against `OpenPFC`.

Read [`../architecture.md`](../architecture.md) first so you know where code belongs. If you use `App<Model>` and JSON, follow [`../app_pipeline.md`](../app_pipeline.md) for wiring order and section names. For an ordered **extend** track (and links to examples), see [`../learning_paths.md`](../learning_paths.md) → *Extend physics and declarative configs*.

- Kernel — backend-agnostic data, decomposition, simulation abstractions (`Model`, `Simulator`, `FieldModifier`, …).
- Runtime — CPU / CUDA / HIP execution and FFT implementations.
- Frontend — optional JSON/TOML `App`, I/O helpers, UI-oriented pieces.

Most extension work is new types in your app or examples that plug into `Model`, `FieldModifier`, `ResultsWriter`, or the `App` registration APIs.

## Minimum file set for a config-driven binary

If you ship an executable that uses `pfc::ui::App<YourModel>` (JSON/TOML on disk), you typically need:

| Piece | Purpose |
|-------|---------|
| `your_model.hpp` / `.cpp` | `YourModel : public pfc::Model` with `initialize`, `step`; optional `void from_json(const pfc::ui::json &, YourModel &)` for `model.params`. |
| `main.cpp` | `register_field_modifier<…>(…)` for any custom IC/BC types; `pfc::ui::App<YourModel> app(argc, argv); return app.main();` |
| CMake | `find_package(OpenPFC)`, `find_package(nlohmann_json)`, `target_link_libraries(… OpenPFC nlohmann_json::nlohmann_json)` |
| Config file | Path as `argv[1]` — see [`../app_pipeline.md`](../app_pipeline.md) for section names. |

End-to-end walkthrough: [`../tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md). Type map: [`../class_tour.md`](../class_tour.md).

## Extension points (typical)

| Goal | Mechanism | Starting points |
|------|-----------|-----------------|
| New physics / PDE model | Subclass `Model`, wire FFT and fields | `examples/04_diffusion_model.cpp`, `examples/12_cahn_hilliard.cpp` |
| Initial or boundary behavior | `FieldModifier` or functional `field::apply` | [`../getting_started/functional_field_ops.md`](../getting_started/functional_field_ops.md), `examples/14_custom_field_initializer.cpp` |
| Declarative runs (JSON/TOML) | `pfc::ui::App<YourModel>` + registration | `examples/10_ui_register_ic.cpp`, shipped apps under `apps/` |
| Custom coordinate / spatial setup | World and field helpers | `examples/17_custom_coordinate_system.cpp` |
| Output formats | Implement `ResultsWriter` or use existing writers | `examples/11_write_results.cpp` |

## Worked examples in `examples/`

| Source | Focus |
|--------|--------|
| `14_custom_field_initializer.cpp` | Custom initializer pattern |
| `17_custom_coordinate_system.cpp` | Non-trivial spatial setup |
| `10_ui_register_ic.cpp` | Registering pieces with the UI / config path |
| `12_cahn_hilliard.cpp` | End-to-end spectral model with simulator stack |

Full index: [`../examples_catalog.md`](../examples_catalog.md).

## Configuration validation

Models can expose validated parameters (ranges, required keys). See [`../parameter_validation.md`](../parameter_validation.md), the Configuration Validation section in the root [`README.md`](../../README.md), and `apps/tungsten/include/tungsten/common/tungsten_input.hpp` for a large metadata-driven example.

## Style and API conventions

Follow [`../styleguide.md`](../styleguide.md) (naming, headers, SPDX, free-function style where appropriate).

## Applications as references

Production-style programs under `apps/` (tungsten, aluminum, Allen–Cahn) show how a full binary ties JSON/TOML, model parameters, and MPI together. See [`../applications.md`](../applications.md).
