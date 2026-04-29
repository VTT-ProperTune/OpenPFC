<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Glossary

Short definitions for OpenPFC and PFC simulation vocabulary. Deeper design detail is in **[`architecture.md`](architecture.md)** and **[`halo_exchange.md`](halo_exchange.md)**.

| Term | Meaning |
|------|---------|
| **Phase field crystal (PFC)** | Semi-atomistic modeling: periodic order parameters approximate crystal density; used for microstructure, defects, and large-scale evolution. |
| **World** | Global description of the grid: dimensions, spacing, origin, periodicity (`kernel/data`). |
| **Decomposition** | Partition of the world across MPI ranks; each rank owns an **inbox** of local samples. |
| **Inbox / outbox** | Real-space samples owned by a rank (**inbox**) and complex **outbox** layout after real-to-complex FFT (sizes differ due to symmetry). |
| **Halo (ghost cells)** | Overlapping boundary layers exchanged between neighbors for finite differences or stencils. |
| **In-place vs separated halos** | **In-place:** ghosts stored in the same array as interior (good for FD-only). **Separated:** ghosts in face buffers so the interior stays **FFT-safe** — see [`halo_exchange.md`](halo_exchange.md). |
| **Spectral method** | Spatial operators applied in Fourier space (FFT forward → multiply → inverse FFT); primary path in OpenPFC. |
| **Finite difference (FD)** | Spatial derivatives on the grid with halos; coexists with spectral workflows when layouts match the docs. |
| **HeFFTe** | Library for **distributed FFT**; OpenPFC’s FFT backend selection goes through HeFFTe (FFTW / CUDA / ROCm). |
| **Kernel / runtime / frontend** | **Kernel:** portable abstractions. **Runtime:** CPU/CUDA/HIP execution and FFT. **Frontend:** optional JSON/TOML **`App`**, I/O helpers — see [`architecture.md`](architecture.md). |
| **Model** | Your physics: fields, time step, spectral or FD operators (`kernel/simulation`). |
| **Simulator** | Orchestrates **Model**, **Time**, **FieldModifier** (IC/BC), **ResultsWriter**. |
| **FieldModifier** | Applies initial or boundary updates each step or at startup. |
| **SimulationContext** | Small bundle (MPI comm, rank-0) passed with **`Model`** to modifiers for rank-aware I/O; see **`simulation_context.hpp`** and **[`app_pipeline.md`](app_pipeline.md)**. |
| **App&lt;Model&gt;** | Frontend entry point that loads JSON/TOML and builds a session (`pfc::ui::App`). |

## See also

- **[`configuration.md`](configuration.md)** — config file sections  
- **[`faq.md`](faq.md)** — practical Q&A
