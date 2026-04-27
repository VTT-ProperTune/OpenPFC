<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenPFC test suite

This directory holds automated tests for the library and related workflows. The layout separates **fast unit checks**, **multi-component integration scenarios**, optional **benchmarks**, and shared **fixtures**.

## Testing philosophy

We aim for:

- **High coverage** of production code (Codecov and local reports target on the order of **90%** line coverage; see [`.github/workflows/coverage.yml`](../.github/workflows/coverage.yml)).
- **Fast default feedback**: most development runs **CTest** without benchmarks and without spinning up one OS process per Catch2 case (see below).
- **Layered confidence**: small **unit** tests near components, **integration** tests for cross-cutting behavior (FFT, decomposition, I/O, short simulations).
- **Explicit MPI and GPU paths**: multi-rank and CUDA checks are **tagged** and/or **optional CMake targets** so laptops and minimal CI hosts stay usable.

Further detail by area: [`unit/README.md`](unit/README.md), [`integration/README.md`](integration/README.md), [`benchmarks/README.md`](benchmarks/README.md), [`fixtures/README.md`](fixtures/README.md).

## Strategies: one main executable vs several

### Primary suite: `openpfc-tests`

Most tests compile into a **single** Catch2 executable, `openpfc-tests`:

- **Sources** are wired in via `target_sources(openpfc-tests PRIVATE …)` from `tests/unit/**`, `tests/integration/**`, and optionally `tests/benchmarks/`.
- **Custom `main`** (`tests/runtests.cpp`) constructs `pfc::MPI_Worker` once so **MPI is initialized a single time** for the whole run. That avoids large `MPI_Init` / `MPI_Finalize` overhead per test case.
- **CTest** registers **`openpfc-all-tests`**, which runs:

  ```text
  openpfc-tests "~[MPI]"
  ```

  i.e. **all Catch2 cases except those tagged `[MPI]`**, in **one process invocation**. That is much faster than registering each Catch2 test as its own CTest (which would restart the binary repeatedly).

**When adding a typical test:** new `.cpp` file under `unit/` or `integration/`, list it in the nearest `CMakeLists.txt`, use Catch2 `TEST_CASE` names and **tags** (`[unit]`, `[integration]`, `[MPI]`, …). No new executable is required.

### Additional executables (on purpose)

Smaller binaries are used when **linking**, **launch**, or **isolation** differs from the main suite:

| Target (examples) | Rationale |
|-------------------|-----------|
| `test_gpu_device`, `test_gpu_vector`, `test_gpu_kernels`, `test_gpu_fft` | Built only with CUDA; link CUDA / kernel objects without pulling them into every default build. See `tests/unit/runtime/gpu/CMakeLists.txt`. |
| `test_vtk_writer` | CTest runs **serial and 2-rank MPI** invocations via `mpiexec`. See `tests/unit/frontend/io/CMakeLists.txt`. |
| `test_logging` | Serial Catch2 with `Catch2WithMain` (no shared MPI runner). |
| Tests under `apps/` (e.g. Tungsten) | Application-scoped correctness, not the core library target. |

**Rule of thumb:** use **`openpfc-tests`** unless you need a different `main`, different mandatory libraries, or a dedicated `mpiexec` line in CTest.

### MPI tagging and CTest MPI suites

- Tests that **require** multi-rank MPI behavior should be tagged **`[MPI]`** (and narrower tags like `[ring]`, `[grid]` where `tests/CMakeLists.txt` filters on them).
- The default **`openpfc-all-tests`** run **excludes** `[MPI]` so serial/local runs stay simple.
- If **`OpenPFC_RUN_MPI_SUITES=ON`** at configure time, extra CTest entries (e.g. `mpi_2procs_all`) run **`openpfc-tests`** under **`mpiexec`** with tag filters. Those tests use the environment variable **`OPENPFC_TEST_MPI_INIT=1`** so the runner can align with the pre-initialized MPI world (see comments in `tests/CMakeLists.txt`).
- On hosts with **few logical CPUs**, CMake may **skip** registering 3- and 4-rank suites unless **`OpenPFC_MPI_TEST_REGISTER_HIGH_RANK_ALWAYS=ON`**. CI sets **`OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2`** so only the 2-rank suite is registered on GitHub-hosted runners.

## Directory structure

```text
tests/
├── runtests.cpp            # Catch2 runner; single MPI_Worker for whole suite
├── CMakeLists.txt          # openpfc-tests, CTest, optional benchmarks
├── unit/                   # Unit tests (mirrors kernel / runtime / frontend / operators)
├── integration/            # Multi-component scenarios
├── benchmarks/             # Optional; gated by OpenPFC_BUILD_BENCHMARKS
└── fixtures/               # Shared headers (e.g. mocks); on include path as "fixtures/..."
```

## Building and running

### Configure and build

Enable tests (see root `CMakeLists.txt` / options; typical CI uses `-DOpenPFC_BUILD_TESTS=ON`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DOpenPFC_BUILD_TESTS=ON
cmake --build build --target openpfc-tests
```

Optional: **benchmarks** inside the same binary (default **OFF**):

```bash
cmake -B build -DOpenPFC_BUILD_BENCHMARKS=ON
cmake --build build --target openpfc-tests
```

Optional: **MPI CTest suites** (for multi-rank jobs):

```bash
cmake -B build -DOpenPFC_RUN_MPI_SUITES=ON
```

### CTest (aligned with CI)

From the build directory:

```bash
cd build
ctest --output-on-failure -j2 \
  --exclude-regex "benchmark" \
  --timeout 300
```

This runs **`openpfc-all-tests`**, the standalone targets (**`test_vtk_writer`**, **`test_logging`**, GPU tests if CUDA was enabled), and—when configured—**MPI** suites. CI uses the same **`ctest`** line in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) with **`OpenPFC_RUN_MPI_SUITES=ON`** and **`OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2`**.

**Note:** The **coverage** workflow runs the same **`ctest`** invocation and MPI suite cap as the main CI workflow, with coverage flags passed at configure time. See [`.github/workflows/coverage.yml`](../.github/workflows/coverage.yml).

### Run `openpfc-tests` directly

```bash
cd build
./tests/openpfc-tests '~[MPI]'    # same filter as openpfc-all-tests
./tests/openpfc-tests --list-tests
```

Running **`openpfc-tests` with no arguments** runs **including** `[MPI]` cases, which may fail or hang if not launched with the right rank count—prefer explicit tags or **`ctest`**.

### Catch2 reporters and filtering

```bash
./tests/openpfc-tests '~[MPI]' --reporter compact
./tests/openpfc-tests '~[MPI]' --reporter junit --out results.xml
./tests/openpfc-tests '[unit]'
./tests/openpfc-tests '[fft]'
./tests/openpfc-tests 'World*'
```

### Local coverage (optional)

If you configure with **`OpenPFC_ENABLE_CODE_COVERAGE`** (see `tests/CMakeLists.txt`) or with the same **`--coverage`** flags as the coverage workflow, run **`ctest`** as above, then **`lcov`/`genhtml`** (the coverage job uses **`gcov-11`** to match GCC 11—see the workflow file for the exact **`lcov`** filters).

## Test framework

**Catch2 v3** (`REQUIRE`, `CHECK`, `Approx`, tags, `BENCHMARK` in benchmarks). Reference: [Catch2 docs](https://github.com/catchorg/Catch2/tree/devel/docs).

## Shared fixtures

Headers under [`fixtures/`](fixtures/) (e.g. `mock_model.hpp`) are included as `#include "fixtures/..."` because `tests/` is on the include path for `openpfc-tests`. Details: [`fixtures/README.md`](fixtures/README.md).

## Writing new tests

1. Decide **unit** vs **integration** (and whether the case needs **`[MPI]`** or CUDA).
2. Place the file under the matching subtree and **register** it in that directory’s **`CMakeLists.txt`** (`target_sources(openpfc-tests PRIVATE …)`), unless you are intentionally adding a **separate** executable.
3. Name files **`test_<component>.cpp`** where that matches local convention.
4. Use clear **`TEST_CASE` titles** and **tags** (`[unit]`, `[integration]`, module tags, `[MPI]` when appropriate).

## Continuous integration (summary)

- **[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)** — **Ubuntu 24.04**, matrix **GCC 11** and **GCC 13**, **Debug** and **Release**, HeFFTe 2.4.1 CPU build, **`OpenPFC_RUN_MPI_SUITES=ON`**, **`OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2`**, **`ctest`** with benchmark exclusion. **clang-format** (advisory) and **REUSE** run in a separate job; they do not use the Clang compiler for the build matrix.
- **[`.github/workflows/coverage.yml`](../.github/workflows/coverage.yml)** — **GCC 11**, coverage flags, **`OpenPFC_RUN_MPI_SUITES=ON`**, **`OpenPFC_MPI_TEST_MAX_WORLD_SIZE=2`**, **`ctest`** (benchmarks excluded), Codecov upload.
- **Clang-Tidy** — separate workflow ([`.github/workflows/clang-tidy.yml`](../.github/workflows/clang-tidy.yml)); see [`.github/workflows/README.md`](../.github/workflows/README.md).

## Maintenance

- Prefer **independent** test cases; avoid relying on order or hidden global mutation.
- Keep **unit** cases **fast**; push expensive work to **integration** or **benchmarks**.
- **Benchmarks** are excluded from default **`ctest`** in CI; enable **`OpenPFC_BUILD_BENCHMARKS`** locally when tuning performance.

## Discovering what is registered

From the **build** directory:

- **CTest:** `ctest -N`
- **Catch2:** `./tests/openpfc-tests --list-tests` (and other Catch2 CLI options)

---

**Last updated:** 2026-03-31
