<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Learning paths

OpenPFC is easier to learn when you follow one story at a time. The same pages appear in several places in the documentation, but you do not need to read everything. Choose the route that matches what you are trying to do now, follow it until you can run or modify something real, and only then branch into reference pages.

If you have not yet built the project, start with [`start_here_15_minutes.md`](start_here_15_minutes.md). It gives you one clean success path: configure, build, and run a small MPI example. The routes below assume that first run works.

## I want to run simulations

This route is for someone who wants an existing OpenPFC application to run on a workstation or cluster, produce files, and behave reproducibly.

Begin with the repository-root [`INSTALL.md`](../INSTALL.md). It explains the dependency alignment that matters most in practice: the compiler, MPI and HeFFTe must belong to the same stack. After that, [`quickstart.md`](quickstart.md) shows the two basic execution shapes: a small executable under `examples/`, and a shipped application under `apps/` that reads a JSON or TOML file.

Once the mechanics work, read [`concepts/spectral_stack.md`](concepts/spectral_stack.md). That page explains what the code is doing when it says “spectral”: real-space fields, FFTs, a model step, a simulator, and writers. Then move to [`user_guide/applications.md`](user_guide/applications.md) to choose a shipped binary. Tungsten is the production-style PFC application; Allen–Cahn is a smaller visual demo; Heat3D is a benchmark-like comparison of finite differences and the spectral heat-equation path.

Configuration is the next layer. [`user_guide/configuration.md`](user_guide/configuration.md) introduces the vocabulary, [`user_guide/app_pipeline.md`](user_guide/app_pipeline.md) explains how JSON or TOML becomes a `Simulator`, and [`reference/spectral_app_config_reference.md`](reference/spectral_app_config_reference.md) is where you look up exact keys. When a run writes files, [`user_guide/io_results.md`](user_guide/io_results.md) explains the writers and [`reference/binary_field_io_spec.md`](reference/binary_field_io_spec.md) documents the raw binary format.

For a complete “run and inspect output” walkthrough, read [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md). If your next move is a batch system, continue with [`hpc/operator_guide.md`](hpc/operator_guide.md), then [`tutorials/hpc_slurm_day_one.md`](tutorials/hpc_slurm_day_one.md). GPU decisions belong in [`hpc/gpu_path_decision.md`](hpc/gpu_path_decision.md), not in the middle of your first CPU run.

## I want to extend the physics

This route is for someone who wants to write or modify a model, add parameters, or build a custom config-driven application.

Start with [`concepts/architecture.md`](concepts/architecture.md), because it explains where the kernel, runtime and frontend responsibilities begin and end. Then read [`reference/class_tour.md`](reference/class_tour.md), which connects the important names — `World`, `Model`, `Simulator`, `App`, fields and writers — to the headers you will actually open.

After that, use the long-form getting-started tutorial at [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md). It is slower than `quickstart.md`, but it gives you the mental model for a small out-of-tree CMake project. [`getting_started/functional_field_ops.md`](getting_started/functional_field_ops.md) is the companion for initial and boundary conditions without hand-written nested loops.

When you are ready to make something application-shaped, read [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md). It deliberately teaches the wiring rather than new physics: CMake, MPI, `pfc::ui::App<YourModel>`, JSON on disk and where `Model::step` belongs. The parameter-validation story is in [`user_guide/parameter_validation.md`](user_guide/parameter_validation.md), and the broader extension checklist lives in [`extending_openpfc/README.md`](extending_openpfc/README.md).

The example ladder is useful once you have the concepts. [`tutorials/spectral_examples_sequence.md`](tutorials/spectral_examples_sequence.md) walks through the spectral examples in order, [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md) shows the VTK path, and [`tutorials/fft_heffte_plan_options.md`](tutorials/fft_heffte_plan_options.md) explains the FFT tuning knobs. For the scientific context of shipped models, read [`science/tungsten_quicklook.md`](science/tungsten_quicklook.md) and [`science/cahn_hilliard_vs_allen_cahn.md`](science/cahn_hilliard_vs_allen_cahn.md).

## I want to integrate the library

This route is for someone embedding OpenPFC into another CMake project rather than primarily running the shipped applications.

The shortest mechanical example is in [`quickstart.md`](quickstart.md), in the section about `find_package(OpenPFC)`. The longer version is again [`getting_started/01-basics/README.md`](getting_started/01-basics/README.md), because it shows what a small linked program actually does after CMake resolves the package. Keep [`reference/build_options.md`](reference/build_options.md) nearby when you care about install layout, exported targets or optional components.

For API orientation, use [`reference/examples_catalog.md`](reference/examples_catalog.md) to find runnable examples and [`reference/api_examples_walkthrough.md`](reference/api_examples_walkthrough.md) for the curated Doxygen snippets. The published HTML reference at [vtt-propertune.github.io/OpenPFC/dev](https://vtt-propertune.github.io/OpenPFC/dev/) is where class and function signatures belong; this prose tree is where the surrounding story lives.

## If you are still unsure

If the routes above all sound close but not quite right, read [`when_not_to_use_openpfc.md`](when_not_to_use_openpfc.md). It is intentionally candid about where OpenPFC is a good fit, where the spectral stack has limits today, and when a smaller tool may be faster.
