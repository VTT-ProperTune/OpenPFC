<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Examples

Small programs that demonstrate the OpenPFC library. They are built when `OpenPFC_BUILD_EXAMPLES=ON` (CMake default).

## Build

From the repository root (after satisfying [`INSTALL.md`](../INSTALL.md)):

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Executables appear under `build/examples/` (or `build/Release/examples/` with some multi-config generators).

## Suggested first runs

Run with MPI from the build directory (adjust rank count):

```bash
cd build
mpirun -n 4 ./examples/02_domain_decomposition
mpirun -n 4 ./examples/03_parallel_fft
mpirun -n 4 ./examples/05_simulator
mpirun -n 4 ./examples/12_cahn_hilliard
```

These cover decomposition, distributed FFT, the simulator stack, and a richer spectral model.

## Full catalog and curriculum

Every target registered in `CMakeLists.txt` is listed in [`../docs/examples_catalog.md`](../docs/examples_catalog.md), including **Tier 1–3** suggested order and a flowchart.

## See also

- [`../docs/learning_paths.md`](../docs/learning_paths.md) — ordered tracks by role
- [`../docs/quickstart.md`](../docs/quickstart.md) — onboarding
- [`../docs/tutorials/end_to_end_visualization.md`](../docs/tutorials/end_to_end_visualization.md) — run once, inspect outputs
- [`../docs/api_examples_walkthrough.md`](../docs/api_examples_walkthrough.md) — Doxygen `docs/api/examples` reading order
- [`../docs/showcase.md`](../docs/showcase.md) — figures → runnable entry points
- [`../docs/class_tour.md`](../docs/class_tour.md) — how examples map to `World`, `Model`, `Simulator`, `App`
- [`../docs/extending_openpfc/README.md`](../docs/extending_openpfc/README.md) — turning patterns into your own model or `App`
- `fft_backend_selection.toml` — commented TOML for `[plan_options]` (FFT backend and HeFFTe knobs)
- [`../docs/configuration.md`](../docs/configuration.md) — how config files map to the framework
- [`../docs/app_pipeline.md`](../docs/app_pipeline.md) — declarative runs with JSON/TOML
