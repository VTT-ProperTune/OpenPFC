<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: HeFFTe `plan_options` (FFT backend and communication)

Config-driven apps read FFT settings from your JSON/TOML (often under `[plan_options]` in TOML). The **annotated** file in the tree is the best reference for key names and comments.

## 1. Open the reference file

Path: [`examples/fft_backend_selection.toml`](../../examples/fft_backend_selection.toml)

It is organized in four blocks you will see in real configs:

| Block | Role |
|-------|------|
| `[domain]` | Grid size, spacing, origin — same information as JSON `domain` / top-level world keys in apps (see [`../configuration.md`](../user_guide/configuration.md)). |
| `[plan_options]` | HeFFTe backend (`fftw`, `cuda`, …), reshape algorithm, pencils, GPU-aware MPI, etc. |
| `[timestepping]` | `t0`, `t1`, `dt`, `saveat` — mirrors the `Time` object in the spectral stack. |
| `[model]` | Placeholder name/params — real apps (e.g. tungsten) use richer `model.params`. |

## 2. Choose a backend

| `backend` | When |
|-----------|------|
| `fftw` | Default CPU FFT; always available in typical CPU builds. |
| `cuda` | Requires OpenPFC built with CUDA and a HeFFTe CUDA build; see [`gpu_app_quickstart.md`](gpu_app_quickstart.md). |

The TOML file contains **commented alternative blocks** (e.g. CUDA + `use_gpu_aware = true`) you can copy into your own config.

## 3. Tune communication (`reshape_algorithm`, `use_pencils`)

HeFFTe redistributes data between MPI ranks for the FFT. Short guide (details in the TOML comments):

| Option | Meaning |
|--------|---------|
| `reshape_algorithm` | `alltoall` (default), `alltoallv`, `p2p`, `p2p_plined` — trade latency vs message size; try `p2p_plined` at very large scale. |
| `use_pencils` | Pencil vs slab decomposition; can help beyond ~O(1000) ranks at the cost of more steps. |
| `use_gpu_aware` | Requires GPU-aware MPI; avoids staging through host when using CUDA backend. |

Start from defaults (`fftw`, `alltoall`, `use_pencils = false`) and change **one knob at a time** when profiling.

## 4. Copy into your project

1. Copy `[plan_options]` from the example into your TOML, **or** translate keys to JSON (same logical names; see [`../app_pipeline.md`](../user_guide/app_pipeline.md)).  
2. Keep **one** coherent toolchain: the HeFFTe variant (CPU/CUDA/ROCm) must match how OpenPFC was built ([`../build_cpu_gpu.md`](../hpc/build_cpu_gpu.md)).  
3. Validate at run time: wrong backend strings usually fail fast at FFT creation.

## See also

- [`../configuration.md`](../user_guide/configuration.md) — mental model for JSON/TOML  
- [`../app_pipeline.md`](../user_guide/app_pipeline.md) — where `SpectralCpuStack` consumes `plan_options`  
- [`../performance_profiling.md`](../hpc/performance_profiling.md) — measuring the effect of changes  
- [`gpu_app_quickstart.md`](gpu_app_quickstart.md) — GPU binaries and CMake flags  
