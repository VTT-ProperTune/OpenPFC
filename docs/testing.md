<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Building and running tests

OpenPFC ships a **Catch2**-based suite under **`tests/`**, built as the **`openpfc-tests`** executable. This page complements the one-line **`OpenPFC_BUILD_TESTS`** entry in **[`build_options.md`](build_options.md)**.

## Enable / disable

| CMake option | Default | Effect |
|--------------|---------|--------|
| **`OpenPFC_BUILD_TESTS`** | ON | **`enable_testing()`**, **`find_package(Catch2)`**, **`add_subdirectory(tests)`**, install **`openpfc-tests`** to **`bin`** on **`cmake --install`**. |
| **`OpenPFC_BUILD_BENCHMARKS`** | OFF | Adds slow benchmarks from **`tests/benchmarks/`** into the same **`openpfc-tests`** binary when ON. |

Defined in **`cmake/BuildOptions.cmake`**.

## Quick commands

From your **build directory** after a successful configure:

```bash
cmake --build . -j"$(nproc)"
ctest --output-on-failure
```

To run the Catch2 executable directly (faster iteration):

```bash
./openpfc-tests          # or build/openpfc-tests depending on generator
./openpfc-tests '~[MPI]' # same filter as the default CTest target below
```

## What CTest registers

| CTest name | What it runs |
|------------|----------------|
| **`openpfc-all-tests`** | **`openpfc-tests "~[MPI]"`** — all **non-MPI** Catch2 tests in **one** process (fast). |
| **`mpi_2procs_all`** | Only if **`OpenPFC_RUN_MPI_SUITES`** is **ON**: MPI tests on **2** ranks (Catch tags **`[MPI]`** excluding some sub-suites). |
| **`mpi_3procs_ring`** | If **`OpenPFC_RUN_MPI_SUITES`** and the configure host has enough logical CPUs (or you force registration — see below). |
| **`mpi_4procs_grid_multiple`** | Same, for **4** ranks. |

MPI CTest targets are **optional** because many laptops and CI runners cannot launch multi-rank jobs reliably. Options in **`tests/CMakeLists.txt`**:

- **`OpenPFC_RUN_MPI_SUITES`** — master switch for **`mpi_*`** tests.  
- **`OpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS`** — register 3- and 4-rank tests even when the configure host reports few CPUs (e.g. configure on a login node, run tests on a compute node).  
- **`OpenPFC_MPI_TEST_MAX_WORLD_SIZE`** — cap registered MPI world sizes (useful on CI).

VTK writer tests and other targets may **`add_test`** separately under **`tests/unit/frontend/io/`** — use **`ctest -N`** to list everything after configure.

## Application tests

When **`OpenPFC_BUILD_APPS=ON`**, some apps register extra CTest entries (e.g. tungsten / aluminum checks). Those live next to each app’s **`CMakeLists.txt`**.

## See also

- **[`CONTRIBUTING.md`](../CONTRIBUTING.md)** — changelog and review expectations  
- **[`build_options.md`](build_options.md)** — all CMake switches  
- **[`INSTALL.md`](../INSTALL.md)** — Catch2 / MPI / toolchain prerequisites  
- **[`troubleshooting.md`](troubleshooting.md)** — configure and link failures  
