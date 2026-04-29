<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Doxygen API examples walkthrough

Sources live under [`api/examples/`](api/examples/). They are **included in the HTML API reference** when you build documentation (`OpenPFC_BUILD_DOCUMENTATION=ON`). Optionally, you can compile them as standalone programs with `-DBUILD_API_EXAMPLES=ON` (see [`api/examples/CMakeLists.txt`](api/examples/CMakeLists.txt)); binaries are emitted under `<build>/docs/api/examples/`.

Read them **in order** the first time: each file builds on the same stack (`World` → decomposition / FFT → `Simulator` → I/O and modifiers).

| # | Source | Focus |
|---|--------|--------|
| 01 | [`01_world_basics.cpp`](api/examples/01_world_basics.cpp) | Domain geometry and coordinate systems |
| 02 | [`02_fft_transforms.cpp`](api/examples/02_fft_transforms.cpp) | Spectral transforms and k-space operations |
| 03 | [`03_simulator_workflow.cpp`](api/examples/03_simulator_workflow.cpp) | Orchestrating a simulation |
| 04 | [`04_time_stepping.cpp`](api/examples/04_time_stepping.cpp) | Time stepping and output scheduling |
| 05 | [`05_decomposition_parallel.cpp`](api/examples/05_decomposition_parallel.cpp) | MPI domain decomposition |
| 06 | [`06_results_writers.cpp`](api/examples/06_results_writers.cpp) | Parallel result writers |
| 07 | [`07_field_modifiers.cpp`](api/examples/07_field_modifiers.cpp) | Custom initial and boundary conditions |
| 08 | [`08_discrete_field.cpp`](api/examples/08_discrete_field.cpp) | Discrete fields and coordinate mapping |
| 09 | [`09_initial_conditions.cpp`](api/examples/09_initial_conditions.cpp) | Built-in initial-condition patterns |
| 10 | [`10_complete_pfc_model.cpp`](api/examples/10_complete_pfc_model.cpp) | Full model integrating the above |

## How this relates to `examples/`

The tree under [`examples/`](../examples/) is the **tutorial** catalog (many small executables). The `docs/api/examples/` set is curated for **Doxygen** and reads as a single ascending narrative. Cross-reference: [`examples_catalog.md`](examples_catalog.md) (tiers), [`class_tour.md`](class_tour.md).

## See also

- [Published API docs](https://vtt-propertune.github.io/OpenPFC/dev/) — HTML from headers + these snippets
- [`learning_paths.md`](learning_paths.md) — **Library and API reference** track
