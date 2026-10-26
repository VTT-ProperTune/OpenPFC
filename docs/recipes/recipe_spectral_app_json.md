<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Recipe: run tungsten with JSON (spectral session)

**Goal:** run the shipped **tungsten** application with a **JSON** config from the repository (production-style spectral PFC path).

## Prerequisites

- Built with `OpenPFC_BUILD_APPS=ON` (default). CPU binary: `<build>/apps/tungsten/tungsten`.  
- HeFFTe and MPI as in [`INSTALL.md`](../../INSTALL.md).  
- Inputs documented under [`apps/tungsten/inputs_json/README.md`](../../apps/tungsten/inputs_json/README.md).

## Steps

1. Build OpenPFC (Release recommended).

2. Follow [the first tungsten run](../quickstart.md#run-an-application) to copy
   the small single-seed input into `results/first-tungsten/` and replace its
   site-specific output paths. From the checkout root, run:

   ```bash
   mpirun -n 1 ./builds/release/apps/tungsten/tungsten results/first-tungsten/input.json
   ```

   Increase `-n` only within the available allocation. On a cluster use its
   supported launcher; see [the HPC operator guide](../hpc/operator_guide.md).

3. Inspect the config if you change ranks or paths — writers and `fields` entries must stay consistent ([`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md), [`io_results.md`](../user_guide/io_results.md)).

## Expected result

- Clean exit (0) if the run completes.  
- Binary field output if configured under `fields` / `data` in the JSON (see [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

## Next steps

- Other sample inputs: `tungsten_fixed_bc.json`, `tungsten_moving_bc.json`, `tungsten_performance.json` in the same folder ([`applications.md`](../user_guide/applications.md)).  
- GPU variants: [`gpu_path_decision.md`](../hpc/gpu_path_decision.md), [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md).  
