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

If you came here from a project landing page and you're wondering whether OpenPFC is worth your week, the most honest answer lives in [`when_not_to_use_openpfc.md`](when_not_to_use_openpfc.md). Read that first; it will tell you what kinds of simulations the spectral stack is good at today and where the finite-difference machinery is still being completed.

Beyond that, three reader profiles end up here. The first is someone who needs to run a published phase-field crystal simulation, perhaps on an HPC system, and wants the shortest reproducible path from a clone of the repository to a `mpirun` line. The second is a researcher or graduate student who already understands the physics and wants to add their own model, change boundary conditions, or wire up validated parameters. The third is a software engineer integrating OpenPFC into a larger simulation stack, who cares about CMake, `find_package(OpenPFC)`, and the ABI of the library. The next section points each of those readers somewhere useful.

## Where to start

The fastest possible introduction is [`start_here_15_minutes.md`](start_here_15_minutes.md). It walks from a clean clone through configure, build and one MPI example, and stops there on purpose. If you can finish that page, the rest of the documentation will make a lot more sense.

If you want to understand what OpenPFC is going to feel like before you commit to installing anything, read [`spectral_stack.md`](concepts/spectral_stack.md) for the data-flow story and [`architecture.md`](concepts/architecture.md) for how the kernel, runtime and frontend are split. Both are short, prose-first pages that treat the abstractions as actual concepts rather than as type lists.

When you're ready to do real work, [`learning_paths.md`](learning_paths.md) lays out three sequenced tracks — running existing apps, extending physics, and integrating the library — written for the three reader profiles described above. The [`tutorials/`](tutorials/README.md) directory holds the hands-on walkthroughs that those tracks point into, and the small [`recipes/`](recipes/README.md) collection contains copy-paste solutions for the questions people actually ask after their first run.

For installation, the canonical reference is the repository-root [`INSTALL.md`](../INSTALL.md). It covers compilers, MPI, HeFFTe 2.4.1, and the optional CUDA and HIP paths. If your build is misbehaving, [`troubleshooting.md`](troubleshooting.md) collects the failures we see most often and how to fix them; if you have a quick question, [`faq.md`](faq.md) has short answers.

## Published API reference and the prose docs

OpenPFC has two complementary documentation surfaces. The public [HTML class reference](https://vtt-propertune.github.io/OpenPFC/dev/) is generated from headers and the snippets under [`api/examples/`](api/examples); it is the right place to look up a class, a function signature, or a Doxygen group. You can also build it locally by configuring with `OpenPFC_BUILD_DOCUMENTATION=ON` and building the `docs` target.

This `docs/` tree is the other surface. It is the prose home: the install guide, the tutorials, the configuration vocabulary, the HPC playbooks, the architecture story. It deliberately does not duplicate the API reference, and the API site deliberately does not host tutorials. Pair them: when you find a class in the API site that you don't recognise, search this tree for its name in [`class_tour.md`](reference/class_tour.md); when you finish a tutorial here that mentions a method, jump to the API site for its full signature.

## How the documentation is organised

The shape of this directory follows the journey, not the source tree. Conceptual material — what OpenPFC is, how the spectral stack flows, why halos matter, where finite differences fit — lives at the top of the directory in pages such as [`architecture.md`](concepts/architecture.md), [`spectral_stack.md`](concepts/spectral_stack.md), [`halo_exchange.md`](concepts/halo_exchange.md), and [`science_numerics_limits.md`](science/numerics_limits.md). Operational material — running `examples/`, configuring an `App`, debugging a build — sits next to it in [`quickstart.md`](quickstart.md), [`configuration.md`](user_guide/configuration.md), [`troubleshooting.md`](troubleshooting.md), and the tutorials hub.

When the topic is large enough to need its own narrative, it gets a folder. The hands-on tutorials live under [`tutorials/`](tutorials/README.md), the question-shaped recipes under [`recipes/`](recipes/README.md), the multi-day workshop curriculum under [`workshop/`](workshop/README.md), the LUMI-G runbook under [`lumi_slurm/`](lumi_slurm/README.md), the architectural decision records under [`adr/`](adr/README.md), and the framework-extension guides under [`extending_openpfc/`](extending_openpfc/README.md). When you're inside one of those folders, its own `README.md` is the local table of contents; you don't need to keep coming back here.

For the things you actually look up rather than read — JSON keys, the binary file layout, the dependency matrix, CMake options, the glossary — see the reference pages: [`spectral_app_config_reference.md`](reference/spectral_app_config_reference.md), [`binary_field_io_spec.md`](reference/binary_field_io_spec.md), [`dependency_matrix.md`](reference/dependency_matrix.md), [`build_options.md`](reference/build_options.md), and [`glossary.md`](reference/glossary.md).

## Running on clusters

If you are heading for a cluster, the entry point is [`hpc_operator_guide.md`](hpc/operator_guide.md). It is a thin runbook index that points at Slurm patterns, MPI-IO checklists, runtime profiling, and the LUMI-G specifics under [`INSTALL.LUMI.md`](hpc/INSTALL.LUMI.md) and [`lumi_slurm/`](lumi_slurm/README.md). The decision of whether to enable CUDA or HIP at all is its own page: [`gpu_path_decision.md`](hpc/gpu_path_decision.md). The first cluster-side run, with Slurm, is walked end-to-end in [`tutorials/hpc_slurm_day_one.md`](tutorials/hpc_slurm_day_one.md).

## Extending OpenPFC

If you intend to write a model, an `App`, or a custom writer, the entry point is [`extending_openpfc/README.md`](extending_openpfc/README.md). The shortest path from "I have an idea for a custom App" to "I have a built binary that consumes a JSON config" is [`tutorials/custom_app_minimal.md`](tutorials/custom_app_minimal.md); it is deliberately about wiring rather than physics, so the changes you make in your own copy will be physics rather than plumbing. For the parameter-validation story behind `model.params`, read [`parameter_validation.md`](user_guide/parameter_validation.md).

## Contributing and changelog

Code contributions are described in the repository-root [`CONTRIBUTING.md`](../CONTRIBUTING.md), and the user-visible release history is in [`CHANGELOG.md`](../CHANGELOG.md). For doc-only contributions — fixing a sentence, adding a recipe, refreshing a screenshot — see [`contributing-docs.md`](development/contributing-docs.md), which also describes how to preview the site locally with MkDocs and `uv` ([`mkdocs_preview.md`](development/mkdocs_preview.md)).

## A small navigation note

This page is intentionally short and narrative; the long, exhaustive cross-reference table that used to live here was useful for the framework's authors and almost no one else. If you genuinely want every markdown page in one list, the MkDocs site search box is the better tool, and the directory listing on GitHub is the most exhaustive of all. Otherwise, follow the prose: each page below points at a small number of next destinations rather than dumping the whole map at you.
