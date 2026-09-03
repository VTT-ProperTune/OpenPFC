<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Parameter validation (custom models)

OpenPFC encourages fail-fast configuration: invalid or missing parameters should be caught at startup, not hours into a run. The machinery lives in the frontend UI layer and is optional—physics and custom drivers can ignore it or adopt it fully.

## Building blocks

| Piece | Header | Role |
|-------|--------|------|
| `ParameterMetadata<T>` | `openpfc/frontend/ui/parameter_metadata.hpp` | Describes one parameter: name, range, units, required, typical value. |
| `ParameterValidator` | `openpfc/frontend/ui/parameter_validator.hpp` | Aggregates metadata, `validate(json)`, formats errors and summaries. |
| `Physics::from_json` | Your physics header | Used by `SpectralETDSession` to apply `model.params` while the session is constructed (see below). |

Value-returning helpers (`ParameterValidator::validate`, `ValidationResult::{is_valid,format_errors,format_summary}`, `ParameterMetadata::validate` / `format_info`, and `ParameterMetadata::Builder::build`) are marked `[[nodiscard]]` so ignoring validation output is typically a compile-time error.

Validation is typically invoked from application `main` right after `load_settings_file`, before the session is constructed, or integrated inside your app’s settings loader if you have one.

## Validation vs. session parsing order

`ParameterValidator` and the default spectral pipeline both read the **`model.params` JSON object**, but they are separate layers:

| Layer | When | Responsibility |
|-------|------|------------------|
| **Optional validation** | In *your* code, after loading the config and **before** constructing the session | Fail fast on missing keys, bad types, or out-of-range values using metadata you register on `ParameterValidator`. |
| **Library `from_json`** | Inside `SpectralETDSession` setup (`json_spectral_etd_session.hpp`), when `Physics::from_json(params, domain, inbox)` runs | Copies JSON fields into your physics via `from_json`—same subtree as step 3 in [`app_pipeline.md`](app_pipeline.md#driver-order-of-operations). |
| **Wiring** | After physics construction | Catalog ICs/BCs/writers and `CheckpointService` consume `initial_conditions`, `boundary_conditions`, `fields`, `checkpoint`—not `model.params` for physics scalars. |

The framework **never** calls `ParameterValidator` for you. If you validate in `main` and then construct the session, validation runs first; the library still applies `Physics::from_json` so the physics receives the parsed values. Keep validator metadata and `from_json` field names in sync to avoid rejecting configs that would parse, or accepting configs that `from_json` would mis-handle.

## Pattern

1. Declare metadata for each scalar (or structured) parameter your physics reads from `model.params`.
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

Call this from `main` after loading the config file and before constructing the session if you want validation outside the library; many apps instead fold validation into `Physics::from_json`.

## Reference implementation

`apps/tungsten/include/tungsten/common/tungsten_input.hpp` (and related) registers many parameters with ranges and descriptions—use it as the full example.

Smaller programs may only validate 3–5 critical scalars; you can still use the same `ParameterMetadata<double>::builder()` pattern as in the root `README.md` snippet.

## Documentation elsewhere

- Root [`README.md`](../../README.md) — user-facing description of validation output and benefits.
- [`app_pipeline.md`](app_pipeline.md) — when `model.params` is applied in the session pipeline.

## See also

- [`tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md) — minimal config-driven application
- [`styleguide.md`](../development/styleguide.md) — API and header conventions
