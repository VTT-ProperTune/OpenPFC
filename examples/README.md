<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Examples

Small programs that demonstrate the OpenPFC library. They are built when **`OpenPFC_BUILD_EXAMPLES=ON`** (CMake default).

## Build

From the **repository root** (after satisfying **[`INSTALL.md`](../INSTALL.md)**):

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build -j"$(nproc)"
```

Executables appear under **`build/examples/`** (or `build/Release/examples/` with some multi-config generators).

## Suggested first runs

Run with **MPI** from the **build** directory (adjust rank count):

```bash
cd build
mpirun -n 4 ./examples/02_domain_decomposition
mpirun -n 4 ./examples/03_parallel_fft
mpirun -n 4 ./examples/05_simulator
mpirun -n 4 ./examples/12_cahn_hilliard
```

These cover **decomposition**, **distributed FFT**, the **simulator** stack, and a richer **spectral** model.

## Full catalog

Every target registered in **`CMakeLists.txt`** is listed with a one-line description in **[`../docs/examples_catalog.md`](../docs/examples_catalog.md)**.

## See also

- **[`../docs/quickstart.md`](../docs/quickstart.md)** — onboarding  
- **`fft_backend_selection.toml`** — commented TOML for **`[plan_options]`** (FFT backend and HeFFTe knobs)  
- **[`../docs/configuration.md`](../docs/configuration.md)** — how config files map to the framework
