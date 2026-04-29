<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC

<div align="center">

![Visualization of a rapidly solidifying tungsten block — an example phase-field crystal workflow shipped with the project.](img/simulation.png)

*Example result image; parameters, domain, and post-processing will differ in your own runs.*

</div>

## What OpenPFC is

OpenPFC is an open-source C++ framework for phase-field crystal (PFC) and related spectral phase-field models on structured grids. It is built around three ideas: that microstructure-scale simulations of solidification, defects and elastic fields should be tractable at length scales where atomistic molecular dynamics is too costly; that the physics community already has a working language for those problems in the form of PFC and spectral phase-field models; and that the software underneath should be honest about the cost — distributed memory, FFT-based operators, and a build that you can reproduce on a cluster.

In practice this means OpenPFC is MPI-parallel, uses HeFFTe and friends for the spectral work, and ships in two complementary shapes. There is a library, with public headers under `include/openpfc/`, that you can link from your own CMake project when you want to write a custom model or `App`. And there is a small set of ready-made applications under `apps/` that read JSON or TOML and run reproducibly on the cluster you trust. Most users start with the second shape and graduate to the first when they have a research question of their own.

## Who this documentation is for

If you came here from a project landing page and you're wondering whether OpenPFC is worth your week, start with [the fit guide](when_not_to_use_openpfc.md). It explains what kinds of simulations the spectral stack is good at today and where the finite-difference machinery is still being completed.

Beyond that, three reader profiles end up here. The first is someone who needs to run a published phase-field crystal simulation, perhaps on an HPC system, and wants the shortest reproducible path from a clone of the repository to a `mpirun` line. The second is a researcher or graduate student who already understands the physics and wants to add their own model, change boundary conditions, or wire up validated parameters. The third is a software engineer integrating OpenPFC into a larger simulation stack, who cares about CMake, `find_package(OpenPFC)`, and the ABI of the library. The next section points each of those readers somewhere useful.

## Where to start

The fastest possible introduction is [Start here](start_here_15_minutes.md). It walks from a clean clone through configure, build and one MPI example, and stops there on purpose. If you can finish that page, the rest of the documentation will make a lot more sense.

If you want to understand what OpenPFC is going to feel like before you commit to installing anything, read [the spectral stack story](concepts/spectral_stack.md) and [the architecture overview](concepts/architecture.md). Both are short, prose-first pages that treat the abstractions as actual concepts rather than as type lists.

When you're ready to do real work, [Learning paths](learning_paths.md) lays out three sequenced tracks: running existing apps, extending physics, and integrating the library. The [tutorials](tutorials/README.md) are the hands-on walkthroughs those tracks point into, and the small [recipes](recipes/README.md) collection contains copy-paste solutions for questions people usually ask after their first run.

For installation, the canonical reference is the repository-root [INSTALL guide](../INSTALL.md). It covers compilers, MPI, HeFFTe 2.4.1, and the optional CUDA and HIP paths. If your build is misbehaving, [Troubleshooting](troubleshooting.md) collects the failures we see most often and how to fix them; if you have a quick question, the [FAQ](faq.md) has short answers.

## Published API reference and the prose docs

OpenPFC has two complementary documentation surfaces. The public [HTML class reference](https://vtt-propertune.github.io/OpenPFC/dev/) is generated from headers and the snippets under [`api/examples/`](api/examples); it is the right place to look up a class, a function signature, or a Doxygen group. You can also build it locally by configuring with `OpenPFC_BUILD_DOCUMENTATION=ON` and building the `docs` target.

This `docs/` tree is the other surface. It is the prose home: the install guide, the tutorials, the configuration vocabulary, the HPC playbooks, the architecture story. It deliberately does not duplicate the API reference, and the API site deliberately does not host tutorials. Pair them: when you find a class in the API site that you don't recognise, search this tree for its name in [`class_tour.md`](reference/class_tour.md); when you finish a tutorial here that mentions a method, jump to the API site for its full signature.

## How the documentation is organised

The shape of this directory follows the journey, not the source tree. Conceptual material — what OpenPFC is, how the spectral stack flows, why halos matter, where finite differences fit — lives mostly under `concepts/` and `science/`. Operational material — running `examples/`, configuring an `App`, debugging a build — lives in the top-level start pages and under `user_guide/`, `tutorials/`, and `hpc/`.

When the topic is large enough to need its own narrative, it gets a folder. The hands-on tutorials live under [`tutorials/`](tutorials/README.md), the question-shaped recipes under [`recipes/`](recipes/README.md), the multi-day workshop curriculum under [`workshop/`](workshop/README.md), the LUMI-G runbook under [`lumi_slurm/`](lumi_slurm/README.md), the architectural decision records under [`adr/`](adr/README.md), and the framework-extension guides under [`extending_openpfc/`](extending_openpfc/README.md). When you're inside one of those folders, its own `README.md` is the local table of contents; you don't need to keep coming back here.

For the things you actually look up rather than read — JSON keys, the binary file layout, the dependency matrix, CMake options, the glossary — use the reference pages for [spectral app configuration](reference/spectral_app_config_reference.md), [binary field files](reference/binary_field_io_spec.md), [dependencies](reference/dependency_matrix.md), [CMake options](reference/build_options.md), and [terminology](reference/glossary.md).

## Running on clusters

If you are heading for a cluster, the entry point is [the HPC operator guide](hpc/operator_guide.md). It points at Slurm patterns, MPI-IO checklists, runtime profiling, and the LUMI-G specifics under [the LUMI install notes](hpc/INSTALL.LUMI.md) and [the LUMI Slurm examples](lumi_slurm/README.md). The decision of whether to enable CUDA or HIP at all is covered in [the GPU path guide](hpc/gpu_path_decision.md). The first cluster-side run, with Slurm, is walked end-to-end in [the Slurm day-one tutorial](tutorials/hpc_slurm_day_one.md).

## Extending OpenPFC

If you intend to write a model, an `App`, or a custom writer, start with [Extending OpenPFC](extending_openpfc/README.md). The shortest path from “I have an idea for a custom App” to “I have a built binary that consumes a JSON config” is [the minimal custom App tutorial](tutorials/custom_app_minimal.md); it is deliberately about wiring rather than physics, so the changes you make in your own copy will be physics rather than plumbing. For the parameter-validation story behind `model.params`, read [the parameter validation guide](user_guide/parameter_validation.md).

## Contributing and changelog

Code contributions are described in the repository-root [contributing guide](../CONTRIBUTING.md), and the user-visible release history is in the [changelog](../CHANGELOG.md). For doc-only contributions — fixing a sentence, adding a recipe, refreshing a screenshot — see [Contributing to docs](development/contributing-docs.md), which also describes how to preview the site locally with MkDocs and `uv`.

## How to use these docs

Use the top navigation as a reading path. `Welcome` helps you decide whether OpenPFC is the right tool. `Start` gets a build and first run working. `Learn` explains the concepts. `Run` covers applications, configuration and cluster operation. `Extend` is for new models and integrations. `Reference` is where lookup-heavy material lives.
