<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Recipe: run `05_simulator` (MPI)

**Goal:** verify OpenPFC + HeFFTe + MPI with a small, standard example (`examples/05_simulator`).

## Prerequisites

- Built OpenPFC with `OpenPFC_BUILD_EXAMPLES=ON` (default). Binary path: `<build>/examples/05_simulator`.
- Same MPI for configure and run as in [`INSTALL.md`](../../INSTALL.md).

## Steps

1. Configure and build from the repo root (see [`start_here_15_minutes.md`](../start_here_15_minutes.md) or [`quickstart.md`](../quickstart.md) §1).

2. Run from the **build directory**:

   ```bash
   cd build
   mpirun -n 4 ./examples/05_simulator
   ```

   Adjust `-n` to your machine; use at least 1 rank.

## Expected result

- Process exits with code **0**.  
- Rank 0 typically emits INFO logs (world size, simulation stepping).  
- No output files are required for “success”; this recipe is a **stack check**.

## Next steps

- Spectral tutorial ladder: [`tutorials/spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md)  
- Concept tour: [`spectral_stack.md`](../spectral_stack.md)  
