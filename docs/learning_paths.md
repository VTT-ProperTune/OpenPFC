<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Learning paths

Pick a track that matches what you need today. Each path is a **sequence**—follow it in order the first time through. Cross-links point at the same guides indexed in [`README.md`](README.md).

## Run simulations (HPC user)

Goal: build OpenPFC, run a shipped application with a sample input, and know where outputs go.

| Step | What to read / do |
|------|-------------------|
| 1 | Install and toolchain: [`INSTALL.md`](../INSTALL.md) |
| 2 | Fast path from clone to `mpirun`: [`quickstart.md`](quickstart.md) §1–2B |
| 3 | Which binary and config: [`applications.md`](applications.md) (**App chooser** table) |
| 4 | JSON/TOML vocabulary: [`configuration.md`](configuration.md), [`app_pipeline.md`](app_pipeline.md) |
| 5 | Result files (binary / VTK / PNG): [`io_results.md`](io_results.md) |
| 6 | End-to-end “files on disk” (PNG + Tungsten binary): [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md) |
| 7 | More walkthroughs (VTK/ParaView, HeFFTe `plan_options`, spectral `examples/` sequence): [`tutorials/README.md`](tutorials/README.md) |
| 8 | When things break: [`troubleshooting.md`](troubleshooting.md), [`faq.md`](faq.md) |
| 9 | Slurm / batch basics: [`tutorials/hpc_slurm_day_one.md`](tutorials/hpc_slurm_day_one.md), [`mpi_io_layout_checklist.md`](mpi_io_layout_checklist.md) |
| 10 | GPU / LUMI-style jobs: [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md), [`INSTALL.LUMI.md`](INSTALL.LUMI.md), [`lumi_slurm/README.md`](lumi_slurm/README.md) |

## Extend physics and declarative configs (researcher / developer)

Goal: subclass `Model`, optionally drive runs with `App<Model>` and JSON, and validate parameters.

| Step | What to read / do |
|------|-------------------|
| 1 | Layering (kernel / runtime / frontend): [`architecture.md`](architecture.md) |
| 2 | Types and headers map: [`class_tour.md`](class_tour.md) |
| 3 | Narrative tutorial (world → FFT → CMake): [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md) |
| 4 | Functional IC/BC patterns: [`getting_started/functional_field_ops.md`](getting_started/functional_field_ops.md) |
| 5 | How JSON becomes `Simulator`: [`app_pipeline.md`](app_pipeline.md) |
| 6 | Extension checklist: [`extending_openpfc/README.md`](extending_openpfc/README.md) |
| 7 | Out-of-tree `App` + CMake: [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md) |
| 8 | Validated `model.params`: [`parameter_validation.md`](parameter_validation.md) |
| 9 | Example ladder: [`examples_catalog.md`](examples_catalog.md) (tiers), runnable [`../examples/README.md`](../examples/README.md) |
| 10 | Spectral sequence + VTK outputs: [`tutorials/spectral_examples_sequence.md`](tutorials/spectral_examples_sequence.md), [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md); FFT tuning: [`tutorials/fft_heffte_plan_options.md`](tutorials/fft_heffte_plan_options.md) |
| 11 | Science context (tungsten, CH vs Allen–Cahn): [`science_tungsten_quicklook.md`](science_tungsten_quicklook.md), [`science_cahn_hilliard_vs_allen_cahn.md`](science_cahn_hilliard_vs_allen_cahn.md) |

## Library and API reference (integrator)

Goal: link OpenPFC from your CMake project, run small `examples/`, and use the HTML API docs.

| Step | What to read / do |
|------|-------------------|
| 1 | `find_package` and first build: [`quickstart.md`](quickstart.md) §2C, [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md) |
| 2 | Catalog of `examples/` targets: [`examples_catalog.md`](examples_catalog.md) |
| 2b | Guided spectral runs + VTK: [`tutorials/spectral_examples_sequence.md`](tutorials/spectral_examples_sequence.md), [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md) |
| 3 | Doxygen API snippets in order: [`api_examples_walkthrough.md`](api_examples_walkthrough.md) |
| 4 | Published HTML reference: [OpenPFC dev docs](https://vtt-propertune.github.io/OpenPFC/dev/) (build locally with `OpenPFC_BUILD_DOCUMENTATION=ON`) |
| 5 | CMake options and install layout: [`build_options.md`](build_options.md), [`INSTALL.md`](../INSTALL.md) |

## See also

- [`personas.md`](personas.md) — same tracks as short “by role” pages  
- [`tutorials/README.md`](tutorials/README.md) — all walkthroughs in one place
- [`showcase.md`](showcase.md) — figures and which app or example they map to
- [`README.md`](README.md) — full documentation index
