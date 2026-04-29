<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: add a Catch2 unit test

OpenPFC uses **Catch2** and **CTest** for automated tests. This page is a **minimal** pattern for a new test target next to existing ones.

## 1. Read the project conventions

- [`../testing.md`](../development/testing.md) — `OpenPFC_BUILD_TESTS`, MPI test layout, how to run `ctest`.  
- [`../styleguide.md`](../development/styleguide.md) — naming, includes, SPDX headers on new files.

## 2. Pick a home for the test

- **Library / kernel tests:** under `tests/unit/…` mirroring the include layout (see sibling `CMakeLists.txt` files).  
- **Frontend / UI tests:** `tests/unit/frontend/…`.  
- **MPI tests:** some suites use multiple ranks — copy the pattern from an existing `CMakeLists.txt` that declares `MPIEXEC` / `MPIEXEC_NUMPROC`.

## 3. Minimal Catch2 file shape

```cpp
// SPDX-FileCopyrightText: 2026 Your Name or VTT …
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

TEST_CASE("example invariant", "[example]") {
  REQUIRE(1 + 1 == 2);
}
```

Register the source in the appropriate `CMakeLists.txt`: add an executable (or extend `openpfc-tests`), link **`Catch2::Catch2`** or **`Catch2::Catch2WithMain`** and **`OpenPFC`** / **`MPI::MPI_CXX`** as needed — **copy an adjacent directory** (e.g. `tests/unit/frontend/io/CMakeLists.txt`) as a template.

## 4. Build and run

```bash
cmake -S . -B build -DOpenPFC_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"
cd build && ctest -R your_test_pattern --output-on-failure
```

## See also

- [`../contributing-docs.md`](../development/contributing-docs.md) — link checks before you open a PR  
- [`../CONTRIBUTING.md`](../../CONTRIBUTING.md) — changelog and review expectations  
