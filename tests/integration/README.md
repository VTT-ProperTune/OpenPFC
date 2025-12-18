<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Integration Test Suite

This directory contains integration tests for OpenPFC, focused on validating end-to-end workflows across domains (FFT, MPI decomposition, models, I/O).

## Categories
- **complete_simulation**: End-to-end diffusion runs, invariants and mass conservation.
- **parallel_scaling**: Domain decomposition and per-rank layout properties.
- **io_workflows**: VTK writer output and basic roundtrip checks.
- **field_operations**: FieldModifier integrations and IC/BC application.
- **gpu_validation**: CUDA-enabled tests (conditional on CUDA):
	- CUDA DataBuffer roundtrip (forward/backward, float/double)
	- CPU vs CUDA Laplacian equivalence (single rank)
	- CPU vs CUDA Laplacian equivalence (multi-rank MPI)
- **convergence_studies**: Heuristic convergence checks for temporal resolution.

## Tags
All tests are tagged with `[integration]` and a category-specific tag (e.g., `[complete]`, `[io]`, `[mpi]`, `[gpu]`, `[convergence]`). CI may skip tests with `[MPI]` or backend-specific tags.

## Running
Build and run the test suite:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build
cd build && ctest --output-on-failure
```

Run only integration tests:

```bash
cd build && ./tests/openpfc-tests -r console "[integration]"
```

Run a single scenario:

```bash
cd build && ./tests/openpfc-tests -r console "[integration][io]"
```

### CUDA-specific
- Configure with CUDA enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenPFC_ENABLE_CUDA=ON
cmake --build build -j
```

- Run only GPU integration tests:

```bash
cd build
./tests/openpfc-tests -r console "[integration][gpu]"
```

### MPI-specific
- Many tests run in single-rank mode; multi-rank tests are tagged `[mpi]` and work with `mpirun`:

```bash
cd build
mpirun -np 2 ./tests/openpfc-tests -r console "[integration][gpu][mpi]"
```

### Notes
- MPI-dependent tests should remain robust in single-rank runs; where multi-rank behavior is required, tag with `[mpi]` and guard accordingly.
- CUDA tests compile only when `OpenPFC_ENABLE_CUDA` is defined; otherwise they will be skipped.
- Tests follow the OpenPFC philosophy: transparent structs and free functions; assertions focus on invariants and measurable quantities.
