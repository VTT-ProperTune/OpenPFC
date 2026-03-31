<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Unit Tests

This directory contains **unit tests** for OpenPFC components. Unit tests verify individual components in isolation, providing fast feedback during development.

## Purpose

- **Fast execution**: Each test runs in milliseconds
- **Isolation**: Components tested independently using mocks/fixtures
- **Early bug detection**: Catch issues before integration
- **API documentation**: Tests demonstrate intended usage
- **Refactoring confidence**: Safe to change implementation

## Organization

Tests mirror the library layout (kernel, runtime, frontend) plus operators. Under `unit/`:

- **`kernel/data/`** - World, Field, Box3D, MultiIndex, ArrayND, etc.
- **`kernel/decomposition/`** - Domain decomposition
- **`kernel/fft/`** - FFT functionality and settings
- **`kernel/simulation/`** - Model, Simulator, Time
- **`kernel/field/`** - Field operations
- **`frontend/field_modifiers/`** - Initial conditions and boundary conditions
- **`frontend/io/`**, **`frontend/ui/`** - I/O and UI components
- **`runtime/gpu/`** - GPU runtime tests
- **`operators/`** — Sparse vector tests linked into **`openpfc-tests`** (`test_sparsevector.cpp`). **`test_diffop.cpp`** is intentionally not listed in CMake until the diffop API exists.

## Running Unit Tests

From your **CMake build** directory (same layout as **`tests/integration/README.md`**):

```bash
cd build

# All unit tests
./tests/openpfc-tests "[unit]"

# Specific component
./tests/openpfc-tests "[world]"
./tests/openpfc-tests "[fft]"
```

## Writing Unit Tests

1. Place tests in the appropriate component directory
2. Use `test_<component>.cpp` naming convention
3. Tag tests with `[unit]` and component-specific tags
4. Use mocks from `tests/fixtures/` for dependencies
5. Keep tests fast (<10ms) and independent

Prefer unit tests that avoid **heavy** I/O, **long** runs, and **external** services. The **`openpfc-tests`** binary always initializes MPI once via **`runtests.cpp`**; many tests are still purely serial. Tests that **require multi-rank MPI** should be tagged **`[MPI]`** (they are excluded from the default **`openpfc-all-tests`** / **`~[MPI]`** CTest run—see [`tests/README.md`](../README.md)).

For multi-component workflows and short end-to-end checks, use **`tests/integration/`** instead.
