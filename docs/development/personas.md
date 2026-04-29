<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Documentation by role (personas)

Short entry points. Full index: [`README.md`](../README.md). Sequenced tracks: [`learning_paths.md`](../learning_paths.md).

## I run tungsten (or another `App`) on a cluster

**Fastest lane (clone → build → one example):** [`start_here_15_minutes.md`](../start_here_15_minutes.md). **Cluster runbook index (Slurm, MPI-IO, profiling):** [`hpc_operator_guide.md`](../hpc/operator_guide.md).

1. [`INSTALL.md`](../../INSTALL.md) — modules, MPI, HeFFTe. Optional one-page matrix: [`dependency_matrix.md`](../reference/dependency_matrix.md).  
2. [`quickstart.md`](../quickstart.md) §2B — first `mpirun`.  
3. [`tutorials/hpc_slurm_day_one.md`](../tutorials/hpc_slurm_day_one.md) — generic Slurm skeleton (`#SBATCH`, launcher).  
4. [`mpi_io_layout_checklist.md`](../hpc/mpi_io_layout_checklist.md) — output paths, MPI-IO collectives, restart sanity.  
5. [`postprocess_binary_fields.md`](../user_guide/postprocess_binary_fields.md) — optional: interpret raw `.bin` files outside OpenPFC.  
6. [`spectral_app_config_reference.md`](../reference/spectral_app_config_reference.md) + [`apps/tungsten/inputs_json/README.md`](../../apps/tungsten/inputs_json/README.md).  
7. LUMI-specific: [`INSTALL.LUMI.md`](../hpc/INSTALL.LUMI.md), [`lumi_slurm/README.md`](../lumi_slurm/README.md).

## I extend physics in C++ (`Model`, modifiers)

1. [`architecture.md`](../concepts/architecture.md) — layers.  
1b. [`spectral_stack.md`](../concepts/spectral_stack.md) — spectral path (FFT → model → simulator).  
2. [`class_tour.md`](../reference/class_tour.md) — types and headers.  
3. [`tutorials/spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md) — `04` → `05` → `12`.  
4. [`getting_started/functional_field_ops.md`](../getting_started/functional_field_ops.md) — IC/BC without nested loops.  
5. [`extending_openpfc/README.md`](../extending_openpfc/README.md) — checklist.  
6. [`tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md) — optional JSON `App`.  
7. Tests: [`tutorials/add_catch2_test.md`](../tutorials/add_catch2_test.md), [`testing.md`](testing.md).

## I integrate OpenPFC into another CMake project

1. [`quickstart.md`](../quickstart.md) §2C — `find_package(OpenPFC)`.  
2. [`getting_started/01-basics/README.md`](../getting_started/01-basics/README.md) — linking walkthrough.  
3. [`examples_catalog.md`](../reference/examples_catalog.md) + [`api_examples_walkthrough.md`](../reference/api_examples_walkthrough.md).  
4. Published API: [OpenPFC dev docs](https://vtt-propertune.github.io/OpenPFC/dev/).

## See also

- [`science_tungsten_quicklook.md`](../science/tungsten_quicklook.md) — what tungsten simulates  
- [`science_cahn_hilliard_vs_allen_cahn.md`](../science/cahn_hilliard_vs_allen_cahn.md) — example vs app choice  
