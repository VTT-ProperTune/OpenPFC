<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Documentation by role

This page used to duplicate the main role-based documentation paths. To keep the site easier to navigate, those paths now live in one place: [`../learning_paths.md`](../learning_paths.md).

If you run tungsten or another shipped application on a cluster, follow the “I want to run simulations” route there. It starts from installation and a first `mpirun`, then moves through applications, configuration, output files, Slurm and GPU decisions.

If you extend physics in C++, follow the “I want to extend the physics” route. It introduces the architecture, the main types, the getting-started CMake tutorial, custom `App` wiring and parameter validation in the order that usually makes sense for a developer.

If you integrate OpenPFC into another CMake project, follow the “I want to integrate the library” route. It points at the `find_package(OpenPFC)` pattern, the examples catalog, the Doxygen examples and the published API reference.

For a one-command first run, use [`../start_here_15_minutes.md`](../start_here_15_minutes.md). For the public landing page and overall documentation map, return to [`../README.md`](../README.md).
