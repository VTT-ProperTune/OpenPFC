<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
- **`operators/`** - Sparse vector and operator tests. **test_diffop** is intentionally excluded until a diffop API exists and is tracked in the backlog.

## Running Unit Tests

```bash
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

Unit tests should not involve:

- File I/O
- MPI communication (unless testing MPI-specific component)
- Long-running computations
- External dependencies

For tests involving multiple components working together, use `tests/integration/` instead.
