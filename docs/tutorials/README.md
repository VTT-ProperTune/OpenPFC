<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorials

The tutorials are for moments when you want to do one concrete thing with OpenPFC and understand what just happened. They are not meant to be read alphabetically. If you have not yet built the project, start with [`../start_here_15_minutes.md`](../start_here_15_minutes.md); if you are still deciding which route fits you, use [`../learning_paths.md`](../learning_paths.md).

The most practical first tutorial is [`end_to_end_visualization.md`](end_to_end_visualization.md). It takes you from a built project to files on disk, using either Allen–Cahn for a quick visual run or tungsten for the more realistic application path. If your goal is ParaView, continue with [`vtk_paraview_workflow.md`](vtk_paraview_workflow.md), which uses built-in examples that already write VTK. If your output is raw binary instead, [`../user_guide/postprocess_binary_fields.md`](../user_guide/postprocess_binary_fields.md) explains how to reason about metadata, Fortran order and NumPy-side analysis; the exact binary contract lives in [`../reference/binary_field_io_spec.md`](../reference/binary_field_io_spec.md).

If you are learning the spectral stack as a developer, read [`spectral_examples_sequence.md`](spectral_examples_sequence.md). It walks through the examples in the order that makes conceptual sense rather than the order they happen to appear in the directory. Once that sequence feels familiar, [`fft_heffte_plan_options.md`](fft_heffte_plan_options.md) explains the `plan_options` vocabulary and how FFT backend choices enter JSON or TOML-driven runs.

If you want to build your own config-driven application, go to [`custom_app_minimal.md`](custom_app_minimal.md). It is intentionally about wiring: out-of-tree CMake, MPI setup, `pfc::ui::App<YourModel>`, JSON on disk and the boundary between framework plumbing and your physics. The parameter-validation details are in [`../user_guide/parameter_validation.md`](../user_guide/parameter_validation.md), and the JSON-to-`Simulator` lifecycle is described in [`../user_guide/app_pipeline.md`](../user_guide/app_pipeline.md).

Cluster and GPU work should come after a successful CPU run. [`hpc_slurm_day_one.md`](hpc_slurm_day_one.md) gives you the smallest useful Slurm job shape, and [`gpu_app_quickstart.md`](gpu_app_quickstart.md) explains CUDA/HIP builds and GPU-enabled application binaries. For production-style runs, the broader operator path starts in [`../hpc/operator_guide.md`](../hpc/operator_guide.md).

There are a few tutorials that are closer to reference material. [`add_catch2_test.md`](add_catch2_test.md) is the shortest route to adding a unit test and running it through `ctest`. The science pages [`../science/tungsten_quicklook.md`](../science/tungsten_quicklook.md) and [`../science/cahn_hilliard_vs_allen_cahn.md`](../science/cahn_hilliard_vs_allen_cahn.md) help you choose the right model or example before you spend time on a run.

For a slower conceptual introduction, [`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md) tells the “world to FFT to CMake” story. For a lookup-oriented catalog, use [`../reference/examples_catalog.md`](../reference/examples_catalog.md) and [`../reference/api_examples_walkthrough.md`](../reference/api_examples_walkthrough.md).
