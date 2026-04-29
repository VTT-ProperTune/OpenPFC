<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Parameter validation (custom models)

OpenPFC encourages **fail-fast** configuration: invalid or missing parameters should be caught **at startup**, not hours into a run. The machinery lives in the **frontend UI** layer and is **optional**—your **`Model`** can ignore it or adopt it fully.

## Building blocks

| Piece | Header | Role |
|-------|--------|------|
| **`ParameterMetadata<T>`** | `openpfc/frontend/ui/parameter_metadata.hpp` | Describes one parameter: name, range, units, **required**, typical value. |
| **`ParameterValidator`** | `openpfc/frontend/ui/parameter_validator.hpp` | Aggregates metadata, **`validate(json)`**, formats errors and summaries. |
| **`from_json` into your model** | Your translation unit | Still used by **`App`** to apply **`model.params`** after construction. |

Validation is typically invoked from application **`main`** or a thin wrapper **before** the expensive **`App::main()`** path, or integrated inside your app’s settings loader if you have one.

## Pattern

1. **Declare metadata** for each scalar (or structured) parameter your model reads from **`model.params`**.
2. **Call** **`validator.validate(config["model"]["params"])`** (or the root JSON you use).
3. If **`!result.is_valid()`**, print **`result.format_errors()`** and exit.
4. Optionally print **`result.format_summary()`** for reproducibility (see root **`README.md`** — Configuration Validation).

## Reference implementation

**`apps/tungsten/include/tungsten/common/tungsten_input.hpp`** (and related) registers **many** parameters with ranges and descriptions—use it as the **full** example.

Smaller programs may only validate **3–5** critical scalars; you can still use the same **`ParameterMetadata<double>::builder()`** pattern as in the root **`README.md`** snippet.

## Documentation elsewhere

- Root **[`README.md`](../README.md)** — user-facing description of validation output and benefits.
- **[`app_pipeline.md`](app_pipeline.md)** — when **`model.params`** is applied relative to **`App::main()`**.

## See also

- **[`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md)** — minimal **`App`** tutorial
- **[`styleguide.md`](styleguide.md)** — API and header conventions
