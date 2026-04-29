<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Parameter validation (custom models)

OpenPFC encourages fail-fast configuration: invalid or missing parameters should be caught at startup, not hours into a run. The machinery lives in the frontend UI layer and is optional—your `Model` can ignore it or adopt it fully.

## Building blocks

| Piece | Header | Role |
|-------|--------|------|
| `ParameterMetadata<T>` | `openpfc/frontend/ui/parameter_metadata.hpp` | Describes one parameter: name, range, units, required, typical value. |
| `ParameterValidator` | `openpfc/frontend/ui/parameter_validator.hpp` | Aggregates metadata, `validate(json)`, formats errors and summaries. |
| `from_json` into your model | Your translation unit | Used by the default spectral `App` path to apply `model.params` after the session is built (see below). |

Validation is typically invoked from application `main` or a thin wrapper before the expensive `App::main()` path, or integrated inside your app’s settings loader if you have one.

## Validation vs. App parsing order

`ParameterValidator` and the default spectral pipeline both read the **`model.params` JSON object**, but they are separate layers:

| Layer | When | Responsibility |
|-------|------|------------------|
| **Optional validation** | In *your* code, after loading the config and **before** `pfc::ui::App<Model>::main()` | Fail fast on missing keys, bad types, or out-of-range values using metadata you register on `ParameterValidator`. |
| **Library `from_json`** | Inside `SpectralJsonAppRun::execute` (`app_spectral_run.hpp`), **after** `SpectralSimulationSession::assemble` and **before** `model.initialize(dt)` | Copies JSON fields into your model’s parameter struct via your `from_json` overload—same subtree as step 3 in [`app_pipeline.md`](app_pipeline.md#appconcretemodelmain-order-of-operations). |
| **Wiring** | After initialization | `wire_simulator_from_settings` consumes `fields`, `initial_conditions`, `boundary_conditions`, etc.—not `model.params` for physics scalars. |

The framework **never** calls `ParameterValidator` for you. If you validate in `main` and then call `app.main()`, validation runs first; the library still applies `from_json` so your model receives the parsed values. Keep validator metadata and `from_json` field names in sync to avoid rejecting configs that would parse, or accepting configs that `from_json` would mis-handle.

## Pattern

1. Declare metadata for each scalar (or structured) parameter your model reads from `model.params`.
2. Call `validator.validate(config["model"]["params"])` (or the `json` subtree you store parameters in).
3. If `!result.is_valid()`, print `result.format_errors()` and exit.
4. Optionally print `result.format_summary()` for reproducibility (see root `README.md` — Configuration Validation).

Minimal sketch (matches the root `README.md` snippet; headers live under `openpfc/frontend/ui/`):

```cpp
#include <cstdlib>
#include <iostream>
#include <openpfc/frontend/ui/parameter_metadata.hpp>
#include <openpfc/frontend/ui/parameter_validator.hpp>

void validate_my_params(const pfc::ui::json &root) {
  pfc::ui::ParameterValidator validator;
  validator.add_metadata(
      pfc::ui::ParameterMetadata<double>::builder()
          .name("temperature")
          .description("Effective temperature")
          .required(true)
          .range(0.0, 10000.0)
          .typical(3300.0)
          .units("K")
          .build());

  const pfc::ui::json &params = root["model"]["params"];
  auto result = validator.validate(params);
  if (!result.is_valid()) {
    std::cerr << result.format_errors() << '\n';
    std::exit(1);
  }
  if (/* rank 0 */) {
    std::cout << result.format_summary() << '\n';
  }
}
```

Call this from `main` after loading the config file and before `App::main()` if you want validation outside the library; many apps instead fold validation into the same code path that parses `model.params`.

## Reference implementation

`apps/tungsten/include/tungsten/common/tungsten_input.hpp` (and related) registers many parameters with ranges and descriptions—use it as the full example.

Smaller programs may only validate 3–5 critical scalars; you can still use the same `ParameterMetadata<double>::builder()` pattern as in the root `README.md` snippet.

## Documentation elsewhere

- Root [`README.md`](../README.md) — user-facing description of validation output and benefits.
- [`app_pipeline.md`](app_pipeline.md) — when `model.params` is applied relative to `App::main()`.

## See also

- [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) — minimal `App` tutorial
- [`styleguide.md`](styleguide.md) — API and header conventions
