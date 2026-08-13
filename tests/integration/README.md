<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Integration Test Suite

This directory contains integration tests for OpenPFC, focused on validating end-to-end workflows across domains (FFT, MPI decomposition, models, I/O).

## Categories
- **complete_simulation**: End-to-end diffusion runs, invariants and mass conservation.
- **parallel_scaling**: Domain decomposition and per-rank layout properties.
- **io_workflows**: VTK writer output and basic roundtrip checks.
- **field_operations**: FieldModifier integrations and IC/BC application.
- **gpu_validation**: CUDA/HIP tests (cases compile to skip stubs when the vendor spectral backend is off):
	- CUDA / HIP DataBuffer roundtrip (forward/backward, float/double)
	- CUDA / HIP vs CPU diffusion smoke (`create_cuda` / `create_hip` construct; model remains host FFT)
	- CPU vs CUDA / HIP Laplacian equivalence (single rank)
	- CPU vs CUDA / HIP Laplacian equivalence (multi-rank MPI)
	- HIP multi-field `for_each_interior_device` and composite-gradient POD layout (HIP twins of the CUDA device TUs)
- **convergence_studies**: Heuristic convergence checks for temporal resolution.

## Tags
All tests are tagged with `[integration]` and a category-specific tag (e.g., `[complete]`, `[io]`, `[mpi]`, `[gpu]`, `[convergence]`). Multi-rank MPI suites are registered as separate CTest targets when **`OpenPFC_RUN_MPI_SUITES=ON`** at configure time (as in **`.github/workflows/ci.yml`**). GPU/CUDA cases require **`OpenPFC_ENABLE_CUDA=ON`** (and **`OpenPFC_ENABLE_CUDA_SPECTRAL`** for FFT roundtrip and Laplacian). HIP FFT roundtrip and Laplacian require **`OpenPFC_ENABLE_HIP_SPECTRAL`**.

## Running
Build and run the test suite:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build
cd build && ctest --output-on-failure -j2 \
  --exclude-regex "benchmark" \
  --timeout 300
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
- CUDA tests compile to skip stubs when `OpenPFC_ENABLE_CUDA` / `OpenPFC_ENABLE_CUDA_SPECTRAL` is off; HIP FFT roundtrip and Laplacian do the same when `OpenPFC_ENABLE_HIP_SPECTRAL` is off.
- Tests follow the OpenPFC philosophy: transparent structs and free functions; assertions focus on invariants and measurable quantities.
