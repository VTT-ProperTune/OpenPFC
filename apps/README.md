<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications (`apps/`)

Full programs built when **`OpenPFC_BUILD_APPS=ON`** (default). They install under `<prefix>/bin` with `cmake --install`. Unlike [`examples/`](../examples/), each app here is meant as a **model-specific executable** with realistic usage (MPI, FD or spectral stacks, optional GPU).

User-facing overview and example commands: [`docs/user_guide/applications.md`](../docs/user_guide/applications.md).

CMake wires subdirectories from [`apps/CMakeLists.txt`](CMakeLists.txt) in this order: `tungsten`, `aluminumNew`, `cahn_hilliard`, `thin_film`, `surface_diffusion`, `kawahara`, `ehd_film`, `gradient_elasticity`, `allen_cahn`, `heat3d`, `wave2d`, `kobayashi`.

For the seven JSON-session apps, `./scripts/openpfc apps` lists presets and
supported backends. Create a case with `openpfc init CASE --app=NAME`; see the
[CLI guide](../docs/user_guide/cli.md). Teaching drivers with positional arguments
retain their app-specific commands.

## Catalog

| Directory | Role | Config | Typical stack |
|-----------|------|--------|----------------|
| [**`tungsten/`**](tungsten/README.md) | Production-style **3D phase-field crystal** (tungsten); JSON/TOML, validated params, field I/O | JSON / TOML (`inputs_json/`, `inputs_toml/`) | `SpectralETDSession<TungstenPhysics, …>`, spectral / HeFFTe, CPU + optional CUDA/HIP |
| [**`aluminumNew/`**](aluminumNew/README.md) | Compact aluminum PFC sample (`seed_grid_fcc`) | JSON / TOML | `SpectralETDSession<AluminumPhysics, …>`, moving-frame mean-field ETD |
| [**`cahn_hilliard/`**](cahn_hilliard/README.md) | **Fe–Cr-like Cahn–Hilliard** spinodal (\(k^4\) spectral ETD) | JSON (`inputs_json/`) | `SpectralETDSession<CahnHilliardPhysics, …>`; CPU + optional HIP |
| [**`thin_film/`**](thin_film/README.md) | **Lubrication dewetting / coating** (\(k^4\) capillary + disjoining) | JSON (`inputs_json/`) | `SpectralETDSession<ThinFilmPhysics, …>`; CPU + optional HIP |
| [**`surface_diffusion/`**](surface_diffusion/README.md) | **Mullins surface diffusion** (exact \(k^4\) annealing) | JSON (`inputs_json/`) | `SpectralETDSession<SurfaceDiffusionPhysics, …>`; CPU + optional HIP |
| [**`kawahara/`**](kawahara/README.md) | **Kawahara capillary–gravity waves** (odd-order \(ik^3\)/\(ik^5\)) | JSON (`inputs_json/`) | `SpectralETDSession<KawaharaPhysics, …>`; CPU + optional HIP |
| [**`ehd_film/`**](ehd_film/README.md) | **EHD film under a flexible plate** (\(k^6\) bending lubrication) | JSON (`inputs_json/`) | `SpectralETDSession<EhdFilmPhysics, …>`; CPU + optional HIP |
| [**`gradient_elasticity/`**](gradient_elasticity/README.md) | **Strain-gradient elasticity** (Helmholtz–Navier, 4th/6th order) | JSON (`inputs_json/`) | one-shot spectral \(2\times 2\) invert (not ETD); CPU + optional HIP |
| [**`allen_cahn/`**](allen_cahn/README.md) | **2D Allen–Cahn** demo; quick visual check | CLI only (no `App` JSON) | FD, separated halos, optional PNG; CPU + optional CUDA/HIP |
| [**`heat3d/`**](heat3d/README.md) | **3D heat equation** \(\partial_t u = D\Delta u\); five drivers from scratch → spectral implicit | CLI per binary | FD (orders 2–20), spectral pointwise RHS, spectral implicit Euler; OpenMP where enabled |
| [**`wave2d/`**](wave2d/README.md) | **2D acoustic wave** as **coupled first-order** system; mixed periodic / physical **y** boundaries | CLI (+ optional `--vtk` on all variants) | FD (manual 2nd order or orders 2–20); CPU + optional CUDA/HIP |
| [**`kobayashi/`**](kobayashi/README.md) | **Kobayashi** dendritic **phase field + temperature** (periodic **x,y**); Julia `kobayashi_v1`-style FD | CLI; PNG snapshots of \(\phi\) | **`kobayashi_fd_manual`** (MPI halos); **`kobayashi_fd_openmp`** (torus wrap + OpenMP); CPU |

Shared JSON vocabulary for spectral apps is summarized near the repo root in [`schema.json`](schema.json) (see docs for normative references).

## `wave2d` — coupled fields (reference for similar apps)

If you are adding another **multi-field** FD app (several prognostic variables updated together each step), **`wave2d`** is the smallest shipped example that does this cleanly.

**Equations:** the second-order wave equation \(u_{tt} = c^2 \Delta u\) is written as

\[
\partial_t u = v, \qquad \partial_t v = c^2 \Delta u
\]

on an **`nz = 1`** slab, **periodic in \(x\)** (MPI halos), with **homogeneous Dirichlet or Neumann** conditions on **\(y\)** (ghost correction after the periodic exchange).

**Code layout (high level):**

| Piece | Location | Notes |
|-------|----------|--------|
| Physics only (per-point RHS, \(c\), metric scaling) | [`wave2d/include/wave2d/wave_model.hpp`](wave2d/include/wave2d/wave_model.hpp) | `WaveLaplacian`, `WaveIncrements`, `WaveModel::rhs` — same idea as `heat3d`’s `HeatModel` + grads struct, but for \((u,v)\) increments |
| Binaries | [`wave2d/CMakeLists.txt`](wave2d/CMakeLists.txt) | `wave2d_fd_manual`, `wave2d_fd`; optional `wave2d_cuda`, `wave2d_hip` |
| BC / Laplacian wiring | `wave_boundary.hpp`, `wave_step_separated.hpp`, `device_step.hpp`, drivers under `src/cpu/`, `src/cuda/`, `src/hip/` | Mixed boundaries + VTK hooks |

**Binaries:** `wave2d_fd_manual` (fixed second-order stencil), `wave2d_fd` (even orders 2–20), plus GPU twins when CUDA/HIP are enabled. Optional ParaView output: `--vtk` / `--vtk-every`.

**Tests:** `ctest -R wave2d` when tests are enabled (`test_wave2d`, and CPU–GPU parity tests if built).

For extending the framework in general: [`docs/extending_openpfc/README.md`](../docs/extending_openpfc/README.md), [`docs/tutorials/custom_app_minimal.md`](../docs/tutorials/custom_app_minimal.md), [`docs/user_guide/app_pipeline.md`](../docs/user_guide/app_pipeline.md).
