<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Getting started

New to OpenPFC? **Shortest path:** [`../start_here_15_minutes.md`](../start_here_15_minutes.md). Then [`../quickstart.md`](../quickstart.md) (install → run examples or an app → or link the library in your own CMake project). For a **sequenced** path by role, use [`../learning_paths.md`](../learning_paths.md). **Short persona pages:** [`../personas.md`](../personas.md). **How-to recipes:** [`../recipes/README.md`](../recipes/README.md).

## Tutorials (in-repo)

| Topic | Document |
|--------|-----------|
| ~15 min first run (clone → build → `mpirun`) | [`../start_here_15_minutes.md`](../start_here_15_minutes.md) |
| Spectral stack mental model | [`../spectral_stack.md`](../spectral_stack.md) |
| Named how-to recipes | [`../recipes/README.md`](../recipes/README.md) |
| Learning paths by role (run / extend / integrate) | [`../learning_paths.md`](../learning_paths.md) |
| Showcase (figures → apps / examples) | [`../showcase.md`](../showcase.md) |
| Tutorials hub (`docs/tutorials/`) | [`../tutorials/README.md`](../tutorials/README.md) |
| VTK / ParaView workflow | [`../tutorials/vtk_paraview_workflow.md`](../tutorials/vtk_paraview_workflow.md) |
| HeFFTe `plan_options` tutorial | [`../tutorials/fft_heffte_plan_options.md`](../tutorials/fft_heffte_plan_options.md) |
| Spectral examples sequence (04 → 05 → 12) | [`../tutorials/spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md) |
| End-to-end: build → PNG or binary outputs | [`../tutorials/end_to_end_visualization.md`](../tutorials/end_to_end_visualization.md) |
| Quick path: three tracks + “next steps” | [`../quickstart.md`](../quickstart.md) |
| World, decomposition, FFT, `find_package(OpenPFC)` | [`01-basics/README.md`](01-basics/README.md) |
| Functional field ops (IC/BC without nested loops) | [`functional_field_ops.md`](functional_field_ops.md) |
| Tour of main classes and headers | [`../class_tour.md`](../class_tour.md) |
| Minimal custom `App` project (CMake + JSON + MPI) | [`../tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md) |
| Parameter validation for `model.params` | [`../parameter_validation.md`](../parameter_validation.md) |

## Reference tables

| Topic | Document |
|--------|-----------|
| Runnable `examples/` executables | [`../examples_catalog.md`](../examples_catalog.md) |
| Doxygen `api/examples` reading order | [`../api_examples_walkthrough.md`](../api_examples_walkthrough.md) |
| Shipped `apps/` binaries and inputs | [`../applications.md`](../applications.md) |
| `App` config pipeline (JSON → `Simulator`) | [`../app_pipeline.md`](../app_pipeline.md) |
| `ctest` / unit tests | [`../testing.md`](../testing.md) |
| GPU-enabled shipped apps | [`../tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) |
| Example terminal output (reference) | [`../example_run_output.md`](../example_run_output.md) |
| Results writers (binary / VTK / PNG) | [`../io_results.md`](../io_results.md) |
| CMake options | [`../build_options.md`](../build_options.md) |
| Extending models and the UI pipeline | [`../extending_openpfc/README.md`](../extending_openpfc/README.md) |

## See also

- [`../README.md`](../README.md) — full documentation index (architecture, profiling, LUMI, …)
- [`../faq.md`](../faq.md) — common questions (MPI, CMake, missing examples/apps)
- [`../troubleshooting.md`](../troubleshooting.md) — configure/run fixes
- [`../configuration.md`](../configuration.md) — JSON/TOML and `plan_options`
- [`../glossary.md`](../glossary.md) — terminology
- [`../../examples/README.md`](../../examples/README.md) — building and running examples
- [`INSTALL.md`](../../INSTALL.md) — supported build and dependencies
- [`../contributing-docs.md`](../contributing-docs.md) — link checks and doc PR habits
- [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) — contributing overview; [`../../CHANGELOG.md`](../../CHANGELOG.md) — release history
