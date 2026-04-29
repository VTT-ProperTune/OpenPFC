<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorials (`docs/tutorials/`)

Step-by-step and workflow guides. Install and MPI setup: [`INSTALL.md`](../../INSTALL.md). Full documentation index: [`../README.md`](../README.md). **Role-based paths:** [`../learning_paths.md`](../learning_paths.md). **Shortest first run:** [`../start_here_15_minutes.md`](../start_here_15_minutes.md). **Named recipes:** [`../recipes/README.md`](../recipes/README.md). **Spectral mental model:** [`../spectral_stack.md`](../concepts/spectral_stack.md). **GPU vs CPU:** [`../gpu_path_decision.md`](../hpc/gpu_path_decision.md).

| Tutorial | What you will do |
|----------|------------------|
| [**End-to-end visualization**](end_to_end_visualization.md) | Build, run Allen–Cahn or Tungsten, get PNG or binary files on disk |
| [**VTK / ParaView workflow**](vtk_paraview_workflow.md) | Run `11_write_results` / `12_cahn_hilliard`, open `.vti` in ParaView |
| [**HeFFTe `plan_options`**](fft_heffte_plan_options.md) | Read and tune `[plan_options]` using `examples/fft_backend_selection.toml` |
| [**Spectral `App` config keys**](../reference/spectral_app_config_reference.md) | Normative JSON/TOML tables for `SpectralCpuStack` + writers |
| [**Binary MPI-IO field format**](../reference/binary_field_io_spec.md) | On-disk layout for `BinaryWriter` / `BinaryReader` |
| [**Post-process binary fields**](../user_guide/postprocess_binary_fields.md) | Metadata, Fortran order, NumPy sketch outside OpenPFC |
| [**Spectral examples sequence**](spectral_examples_sequence.md) | Ordered path through `04_diffusion_model` → `05_simulator` → `12_cahn_hilliard` |
| [**Minimal custom `App` + JSON**](custom_app_minimal.md) | **Why / what / outcome:** out-of-tree binary + config-driven spectral pipeline; physics lives in `Model::step` (see also app pipeline + spectral examples) |
| [**GPU apps quickstart**](gpu_app_quickstart.md) | CUDA/HIP builds and shipped GPU binaries |
| [**Slurm / batch day one**](hpc_slurm_day_one.md) | Minimal `#SBATCH` job + `mpirun` / `srun` |
| [**MPI / I/O checklist**](../hpc/mpi_io_layout_checklist.md) | Paths, collectives, cluster sanity |
| [**Tungsten PFC (science)**](../science/tungsten_quicklook.md) | What the tungsten app is for |
| [**CH vs Allen–Cahn**](../science/cahn_hilliard_vs_allen_cahn.md) | Which entry point matches your goal |
| [**Add a Catch2 test**](add_catch2_test.md) | Minimal unit-test pattern and `ctest` |

## See also

- [`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md) — long-form “world → FFT” narrative  
- [`../examples_catalog.md`](../reference/examples_catalog.md) — every `examples/` target + curriculum tiers  
- [`../api_examples_walkthrough.md`](../reference/api_examples_walkthrough.md) — Doxygen `docs/api/examples/` reading order  
