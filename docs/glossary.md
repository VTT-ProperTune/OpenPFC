<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Glossary

Short definitions for OpenPFC and PFC simulation vocabulary. Deeper design detail is in [`architecture.md`](architecture.md) and [`halo_exchange.md`](halo_exchange.md). Use this page as a **hub**: other guides link here for jargon; follow links out to tutorials and ADRs.

| Term | Meaning |
|------|---------|
| Phase field crystal (PFC) | Semi-atomistic modeling: periodic order parameters approximate crystal density; used for microstructure, defects, and large-scale evolution. |
| World | Global description of the grid: dimensions, spacing, origin, periodicity (`kernel/data`). |
| Decomposition | Partition of the world across MPI ranks; each rank owns an inbox of local samples. |
| Inbox / outbox | Real-space samples owned by a rank (inbox) and complex outbox layout after real-to-complex FFT (sizes differ due to symmetry). |
| Halo (ghost cells) | Overlapping boundary layers exchanged between neighbors for finite differences or stencils. |
| In-place vs separated halos | In-place: ghosts stored in the same array as interior (good for FD-only). Separated: ghosts in face buffers so the interior stays FFT-safe — see [`halo_exchange.md`](halo_exchange.md). |
| Spectral method | Spatial operators applied in Fourier space (FFT forward → multiply → inverse FFT); primary path in OpenPFC. |
| Finite difference (FD) | Spatial derivatives on the grid with halos; coexists with spectral workflows when layouts match the docs. |
| Abstract spatial operators (direction) | Roadmap: choose **spectral vs FD** for gradients/Laplacians behind a unified interface where supported — see [`adr/0002-gradient-operators-fd-vs-spectral.md`](adr/0002-gradient-operators-fd-vs-spectral.md). |
| HeFFTe | Library for distributed FFT; OpenPFC’s FFT backend selection goes through HeFFTe (FFTW / CUDA / ROCm). |
| Kernel / runtime / frontend | Kernel: portable abstractions. Runtime: CPU/CUDA/HIP execution and FFT. Frontend: optional JSON/TOML `App`, I/O helpers — see [`architecture.md`](architecture.md). |
| Model | Your physics: fields, time step, spectral or FD operators (`kernel/simulation`). |
| Simulator | Orchestrates Model, Time, FieldModifier (IC/BC), ResultsWriter. |
| FieldModifier | Applies initial or boundary updates each step or at startup. |
| SimulationContext | Small bundle (MPI comm, rank-0) passed with `Model` to modifiers for rank-aware I/O; see `simulation_context.hpp` and [`app_pipeline.md`](app_pipeline.md). |
| App&lt;Model&gt; | Frontend entry point that loads JSON/TOML and builds a session (`pfc::ui::App`). |
| `plan_options` | HeFFTe FFT planner settings (backend, reshape algorithm, pencils, GPU-aware MPI, …), usually a JSON object or TOML `[plan_options]` — see [`tutorials/fft_heffte_plan_options.md`](tutorials/fft_heffte_plan_options.md). |
| `saveat` | Time between periodic result writes in the spectral `Time` object; `≤ 0` disables the default JSON binary-writer path ([`spectral_app_config_reference.md`](spectral_app_config_reference.md)). |
| `BinaryWriter` / binary field files | MPI-IO raw output per timestep; no file header — see [`binary_field_io_spec.md`](binary_field_io_spec.md). |
| GPU-aware MPI | MPI implementation that can pass device buffers; required for some CUDA/HIP + HeFFTe paths — see [`INSTALL.LUMI.md`](INSTALL.LUMI.md), [`tutorials/gpu_app_quickstart.md`](tutorials/gpu_app_quickstart.md). |
| VTK (`.vti`) | XML image data for visualization; produced by `VTKWriter` in code — see [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md). |

## See also

- [`README.md`](README.md) — full documentation index (this directory)
- [`class_tour.md`](class_tour.md) — main types, headers, and runnable references
- [`configuration.md`](configuration.md) — config file sections
- [`app_pipeline.md`](app_pipeline.md) — JSON/TOML → `Simulator`
- [`faq.md`](faq.md) — practical Q&A
