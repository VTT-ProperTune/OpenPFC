<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC documentation (repository)

This directory holds guides and assets. Build and install from the repository root: [`INSTALL.md`](../INSTALL.md) (toolchains, HeFFTe 2.4.1, MPI, CUDA/HIP).

## Where to go first

| If you want to… | Open |
|-----------------|------|
| Pick a guided track (run apps, extend models, or integrate the library) | [`learning_paths.md`](learning_paths.md) |
| Figures and runnable entry points | [`showcase.md`](showcase.md) |
| Get running in one pass (examples, app, or `find_package`) | [`quickstart.md`](quickstart.md) |
| Tutorials and the examples hub | [`getting_started/README.md`](getting_started/README.md) |
| Fix configure/MPI/HeFFTe issues | [`troubleshooting.md`](troubleshooting.md) |
| Short Q&A | [`faq.md`](faq.md) |
| Understand JSON/TOML → `Simulator` | [`app_pipeline.md`](app_pipeline.md) |
| Planned structural refactors (roadmap) | [`refactoring_roadmap.md`](refactoring_roadmap.md) |
| Tour of main types (`Model`, `App`, …) | [`class_tour.md`](class_tour.md) |
| Minimal custom `App` project (CMake + JSON) | [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) |
| Parameter validation for custom models | [`parameter_validation.md`](parameter_validation.md) |
| Run `ctest` / Catch2 | [`testing.md`](testing.md) |
| GPU (CUDA/HIP) build + `tungsten_cuda` / config backend | [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) |
| Compare logs to a reference shape | [`example_run_output.md`](example_run_output.md) |
| Edit or add markdown in this tree | [`contributing-docs.md`](contributing-docs.md) |
| Contribute code, tests, or changelog entries | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| See what changed between releases | [`../CHANGELOG.md`](../CHANGELOG.md) |

## Guides by topic

### Configuration and applications

| Topic | Document |
|--------|-----------|
| JSON/TOML sections, `plan_options` | [`configuration.md`](configuration.md) |
| Validated `model.params` (custom apps) | [`parameter_validation.md`](parameter_validation.md) |
| Results writers (binary / VTK / PNG) | [`io_results.md`](io_results.md) |
| Shipped `apps/` programs | [`applications.md`](applications.md) |
| Runnable `examples/` (catalog + folder README) | [`examples_catalog.md`](examples_catalog.md), [`../examples/README.md`](../examples/README.md) |
| Extend models and `App` | [`extending_openpfc/README.md`](extending_openpfc/README.md), [`class_tour.md`](class_tour.md) |
| Terminology | [`glossary.md`](glossary.md) |

### Build and tooling

| Topic | Document |
|--------|-----------|
| CMake options | [`build_options.md`](build_options.md) |
| CPU vs GPU build trees | [`build_cpu_gpu.md`](build_cpu_gpu.md) |
| Code style / API shape | [`styleguide.md`](styleguide.md) |

### Architecture and numerics

| Topic | Document |
|--------|-----------|
| Kernel / runtime / frontend | [`architecture.md`](architecture.md) |
| Halo exchange (FD vs FFT-safe) | [`halo_exchange.md`](halo_exchange.md) |
| Debugging, NaN checks | [`debugging.md`](debugging.md) |

### Profiling and HPC

| Topic | Document |
|--------|-----------|
| Runtime profiling | [`performance_profiling.md`](performance_profiling.md) |
| Profiling export schema | [`profiling_export_schema.md`](profiling_export_schema.md) |
| LUMI-G (ROCm / Cray) | [`INSTALL.LUMI.md`](INSTALL.LUMI.md) |
| LUMI Slurm / tungsten jobs | [`lumi_slurm/README.md`](lumi_slurm/README.md) |

## Tutorials (in-repo)

| Section | Document |
|---------|-----------|
| End-to-end run → PNG or binary artifacts | [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md) |
| World, decomposition, FFT, CMake “hello” | [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md) |
| Functional IC/BC (`field::apply`, …) | [`getting_started/functional_field_ops.md`](getting_started/functional_field_ops.md) |
| Tour of main types and headers | [`class_tour.md`](class_tour.md) |
| Minimal out-of-tree `App` + JSON | [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) |
| Parameter validation for `model.params` | [`parameter_validation.md`](parameter_validation.md) |
| GPU-enabled apps (CUDA/HIP, HeFFTe, JSON backend) | [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) |
| `ctest`, `openpfc-tests`, MPI test suites | [`testing.md`](testing.md) |
| What successful runs print | [`example_run_output.md`](example_run_output.md) |

## API examples (Doxygen)

C++ snippets under [`api/examples/`](api/examples/) are included in the Doxygen build (see [`CMakeLists.txt`](CMakeLists.txt)). Reading order and optional `BUILD_API_EXAMPLES` binaries: [`api_examples_walkthrough.md`](api_examples_walkthrough.md).

## Other

- Changelog / release history: [`CHANGELOG.md`](../CHANGELOG.md) (user-facing and developer-facing changes by version).
- Contributing (overview): [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- Scalability analysis plan: [`scalability_analysis_plan.md`](scalability_analysis_plan.md) (stub that points to [`experiments/scalability/docs/scalability_analysis_plan.md`](../experiments/scalability/docs/scalability_analysis_plan.md) when you use the experiment submodule/tree).
- Image / branding notes: [`image-prompts.md`](image-prompts.md) (prompts for project artwork; not required for simulation).

## Generated HTML (Doxygen)

With `OpenPFC_BUILD_DOCUMENTATION=ON`, configure and build the `docs` target; HTML output is under the build tree (see root [`README.md`](../README.md) and [`CMakeLists.txt`](CMakeLists.txt)).
