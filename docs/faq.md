<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Frequently asked questions

Short answers; deeper detail lives in [`INSTALL.md`](../INSTALL.md), [`architecture.md`](concepts/architecture.md), [`quickstart.md`](quickstart.md), and [`troubleshooting.md`](troubleshooting.md).

## Getting started

Where do I begin after cloning the repo?  
Follow [`quickstart.md`](quickstart.md) (configure → run an example or an app → or link OpenPFC from your own CMake project).

Is OpenPFC header-only?  
No. You link the compiled `openpfc` library and include headers from `include/openpfc/`. See [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md).

Do I need MPI?  
Yes for the documented workflows: the distributed FFT stack and examples assume an MPI-enabled build. There is no supported serial-only configuration.

## Build and CMake

`find_package(OpenPFC)` cannot find OpenPFC  
Install OpenPFC (or point at the build tree if your workflow exports the package), then set `CMAKE_PREFIX_PATH` to the installation prefix, or `-DOpenPFC_DIR=/path/to/lib/cmake/OpenPFC`. See the CMake error walkthrough in [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md).

Examples or apps are missing from my build directory  
Ensure `OpenPFC_BUILD_EXAMPLES=ON` and `OpenPFC_BUILD_APPS=ON` (both default ON). If you previously configured with OFF, clear the option or delete the build directory and reconfigure.

CUDA vs CPU build  
Use separate build trees when toggling GPU options so CMake does not mix flags; see [`build_cpu_gpu.md`](hpc/build_cpu_gpu.md).

## Running

Where are the example executables?  
Under `<build>/examples/` when examples are enabled. Names match the source basename (e.g. `05_simulator`). Full list: [`examples_catalog.md`](reference/examples_catalog.md).

Tungsten / app cannot find my JSON  
Pass an absolute path, or a path relative to your current working directory (often the `build/` folder). Stock samples live under `apps/tungsten/inputs_json/` in the source tree.

How do I know an example or tungsten run succeeded?  
Expect `mpirun` exit code 0 and rank-0 INFO logs. Examples do not all print the same banner; apps may write result files when configured. Short checklist: [`quickstart.md`](quickstart.md) (sections 2A / 2B).

## Extending the framework

How do I add a custom model or IC?  
See [`extending_openpfc/README.md`](extending_openpfc/README.md), [`class_tour.md`](reference/class_tour.md) (where types live), [`app_pipeline.md`](user_guide/app_pipeline.md) (JSON sections), and examples `14_custom_field_initializer.cpp`, `17_custom_coordinate_system.cpp`, `10_ui_register_ic.cpp`. For an out-of-tree binary with `App<Model>` and a config file, follow [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md). Optional startup validation of `model.params`: [`parameter_validation.md`](user_guide/parameter_validation.md).

## Documentation map

| Need | Document |
|------|-----------|
| Index of all guides | [`README.md`](README.md) |
| Learning paths by role | [`learning_paths.md`](learning_paths.md) |
| Showcase (figures → runs) | [`showcase.md`](user_guide/showcase.md) |
| End-to-end artifacts (PNG / binary) | [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md) |
| Onboarding | [`quickstart.md`](quickstart.md), [`start_here_15_minutes.md`](start_here_15_minutes.md) |
| How-to recipes | [`recipes/README.md`](recipes/README.md) |
| Spectral stack (mental model) | [`spectral_stack.md`](concepts/spectral_stack.md) |
| GPU vs CPU choice | [`gpu_path_decision.md`](hpc/gpu_path_decision.md) |
| HPC runbook index | [`hpc_operator_guide.md`](hpc/operator_guide.md) |
| When OpenPFC is (not) right; FD vs spectral direction | [`when_not_to_use_openpfc.md`](when_not_to_use_openpfc.md) |
| Doc vs release versioning | [`documentation_versioning.md`](development/documentation_versioning.md) |
| Paper → repository map | [`from_paper_to_run.md`](development/from_paper_to_run.md) |
| Workshop curriculum | [`workshop/README.md`](workshop/README.md) |
| ADRs (architecture decisions) | [`adr/README.md`](adr/README.md) |
| Operator playbooks (symptom → fix) | [`operator_playbooks.md`](reference/operator_playbooks.md) |
| Numerics / limits | [`science_numerics_limits.md`](science/numerics_limits.md) |
| Printable handbook | [`handbook_build.md`](development/handbook_build.md) |
| Examples catalog + curriculum | [`examples_catalog.md`](reference/examples_catalog.md) |
| API examples reading order | [`api_examples_walkthrough.md`](reference/api_examples_walkthrough.md) |
| Tutorials hub | [`tutorials/README.md`](tutorials/README.md) |
| Personas (by role) | [`personas.md`](development/personas.md) |
| Add a Catch2 test | [`tutorials/add_catch2_test.md`](tutorials/add_catch2_test.md) |
| Binary field format | [`binary_field_io_spec.md`](reference/binary_field_io_spec.md) |
| Post-processing raw `.bin` fields | [`postprocess_binary_fields.md`](user_guide/postprocess_binary_fields.md) |
| Toolchain / deps matrix | [`dependency_matrix.md`](reference/dependency_matrix.md) |
| Spectral `App` config keys | [`spectral_app_config_reference.md`](reference/spectral_app_config_reference.md) |
| Slurm day one | [`tutorials/hpc_slurm_day_one.md`](tutorials/hpc_slurm_day_one.md) |
| MPI / I/O checklist | [`mpi_io_layout_checklist.md`](hpc/mpi_io_layout_checklist.md) |
| Troubleshooting | [`troubleshooting.md`](troubleshooting.md) |
| Config files | [`configuration.md`](user_guide/configuration.md) |
| Terminology | [`glossary.md`](reference/glossary.md) |
| `App` + JSON pipeline | [`app_pipeline.md`](user_guide/app_pipeline.md) |
| Main types / headers map | [`class_tour.md`](reference/class_tour.md) |
| Minimal custom `App` + CMake | [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) |
| Parameter validation | [`parameter_validation.md`](user_guide/parameter_validation.md) |
| `ctest` / tests | [`testing.md`](development/testing.md) |
| GPU (CUDA/HIP) apps | [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md) |
| Example log output (shape) | [`example_run_output.md`](reference/example_run_output.md) |
| CMake options | [`build_options.md`](reference/build_options.md) |
| Editing documentation | [`contributing-docs.md`](development/contributing-docs.md) |
| Contributing (code, tests, changelog) | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Release history / upgrades | [`../CHANGELOG.md`](../CHANGELOG.md) |
| Examples folder | [`../examples/README.md`](../examples/README.md) |
| Published HTML API | [GitHub Pages dev docs](https://vtt-propertune.github.io/OpenPFC/dev/) |

---

## Future documentation improvements (ideas)

Smaller polish items that may still help:

1. Richer transcripts — [`example_run_output.md`](reference/example_run_output.md) documents log *shape*; optional verbatim captures from CI or release machines can be added as collapsible sections if maintainers want byte-level fixtures.
2. CHANGELOG user notes — Keep meaningful entries under `[Unreleased]` in [`CHANGELOG.md`](../CHANGELOG.md); add “upgrading from X” blurbs when CMake or config keys change.
3. Published site — The doc index opens with **Published API reference vs prose in `docs/`**; keep GitHub Pages builds fresh so class reference matches releases.

Link checks: run `python3 scripts/check_doc_links.py` before merging doc changes (see [`contributing-docs.md`](development/contributing-docs.md)).
