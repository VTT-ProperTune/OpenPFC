<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Getting started

**New to OpenPFC?** Start with **[`../quickstart.md`](../quickstart.md)** (install → run examples or an app → or link the library in your own CMake project).

## Tutorials (in-repo)

| Topic | Document |
|--------|-----------|
| Quick path: three tracks + “next steps” | [`../quickstart.md`](../quickstart.md) |
| World, decomposition, FFT, `find_package(OpenPFC)` | [`01-basics/README.md`](01-basics/README.md) |
| Functional field ops (IC/BC without nested loops) | [`functional_field_ops.md`](functional_field_ops.md) |

## Reference tables

| Topic | Document |
|--------|-----------|
| Runnable `examples/` executables | [`../examples_catalog.md`](../examples_catalog.md) |
| Shipped `apps/` binaries and inputs | [`../applications.md`](../applications.md) |
| **`App`** config pipeline (JSON → `Simulator`) | [`../app_pipeline.md`](../app_pipeline.md) |
| Results writers (binary / VTK / PNG) | [`../io_results.md`](../io_results.md) |
| CMake options | [`../build_options.md`](../build_options.md) |
| Extending models and the UI pipeline | [`../extending_openpfc/README.md`](../extending_openpfc/README.md) |

## See also

- **[`../README.md`](../README.md)** — full documentation index (architecture, profiling, LUMI, …)
- **[`../faq.md`](../faq.md)** — common questions (MPI, CMake, missing examples/apps)
- **[`../troubleshooting.md`](../troubleshooting.md)** — configure/run fixes
- **[`../configuration.md`](../configuration.md)** — JSON/TOML and `plan_options`
- **[`../glossary.md`](../glossary.md)** — terminology
- **[`../../examples/README.md`](../../examples/README.md)** — building and running examples
- **[`INSTALL.md`](../../INSTALL.md)** — supported build and dependencies
