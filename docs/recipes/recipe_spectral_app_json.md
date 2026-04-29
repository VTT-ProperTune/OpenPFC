<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Recipe: run tungsten with JSON (spectral `App`)

**Goal:** run the shipped **tungsten** application with a **JSON** config from the repository (production-style spectral PFC path).

## Prerequisites

- Built with `OpenPFC_BUILD_APPS=ON` (default). CPU binary: `<build>/apps/tungsten/tungsten`.  
- HeFFTe and MPI as in [`INSTALL.md`](../../INSTALL.md).  
- Inputs documented under [`apps/tungsten/inputs_json/README.md`](../../apps/tungsten/inputs_json/README.md).

## Steps

1. Build OpenPFC (Release recommended).

2. From the **build directory**, run with a sample JSON (paths relative to `build/`):

   ```bash
   cd build
   mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
   ```

   Increase `-n` for larger jobs; match what your site expects (`mpirun` vs `srun` on Slurm: see [`hpc_operator_guide.md`](../hpc/operator_guide.md)).

3. Inspect the config if you change ranks or paths — writers and `fields` entries must stay consistent ([`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md), [`io_results.md`](../user_guide/io_results.md)).

## Expected result

- Clean exit (0) if the run completes.  
- Binary field output if configured under `fields` / `data` in the JSON (see [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md)).

## Next steps

- Other sample inputs: `tungsten_fixed_bc.json`, `tungsten_moving_bc.json`, `tungsten_performance.json` in the same folder ([`applications.md`](../user_guide/applications.md)).  
- GPU variants: [`gpu_path_decision.md`](../hpc/gpu_path_decision.md), [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md).  
