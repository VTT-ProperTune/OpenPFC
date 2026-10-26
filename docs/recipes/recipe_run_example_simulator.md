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

1. Configure and build from the repo root (see [`start_here_15_minutes.md`](../start_here_15_minutes.md) or [`quickstart.md`](../quickstart.md)).

2. Run from the checkout root, using one rank for the first check:

   ```bash
   mpirun -n 1 ./builds/release/examples/05_simulator
   ```

   Adjust `-n` to your machine; use at least 1 rank.

## Expected result

- Process exits with code **0**.  
- Rank 0 typically emits INFO logs (world size, simulation stepping).  
- The driver-hooks example also writes its event ledger and field samples under `results/`. Use the [driver-hooks lesson](https://vtt-propertune.github.io/OpenPFC/learn/heat/driver-hooks/) to inspect the saved states; exit zero alone is only a launch check.

## Next steps

- Spectral tutorial ladder: [`tutorials/spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md)  
- Concept tour: [`spectral_stack.md`](../concepts/spectral_stack.md)  
