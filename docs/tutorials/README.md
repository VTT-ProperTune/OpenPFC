<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorials (`docs/tutorials/`)

Step-by-step and workflow guides. Install and MPI setup: [`INSTALL.md`](../../INSTALL.md). Full documentation index: [`../README.md`](../README.md). **Role-based paths:** [`../learning_paths.md`](../learning_paths.md).

| Tutorial | What you will do |
|----------|------------------|
| [**End-to-end visualization**](end_to_end_visualization.md) | Build, run Allen–Cahn or Tungsten, get PNG or binary files on disk |
| [**VTK / ParaView workflow**](vtk_paraview_workflow.md) | Run `11_write_results` / `12_cahn_hilliard`, open `.vti` in ParaView |
| [**HeFFTe `plan_options`**](fft_heffte_plan_options.md) | Read and tune `[plan_options]` using `examples/fft_backend_selection.toml` |
| [**Spectral `App` config keys**](../spectral_app_config_reference.md) | Normative JSON/TOML tables for `SpectralCpuStack` + writers |
| [**Binary MPI-IO field format**](../binary_field_io_spec.md) | On-disk layout for `BinaryWriter` / `BinaryReader` |
| [**Spectral examples sequence**](spectral_examples_sequence.md) | Ordered path through `04_diffusion_model` → `05_simulator` → `12_cahn_hilliard` |
| [**Minimal custom `App` + JSON**](custom_app_minimal.md) | Out-of-tree CMake project with `App<Model>` |
| [**GPU apps quickstart**](gpu_app_quickstart.md) | CUDA/HIP builds and shipped GPU binaries |

## See also

- [`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md) — long-form “world → FFT” narrative  
- [`../examples_catalog.md`](../examples_catalog.md) — every `examples/` target + curriculum tiers  
- [`../api_examples_walkthrough.md`](../api_examples_walkthrough.md) — Doxygen `docs/api/examples/` reading order  
