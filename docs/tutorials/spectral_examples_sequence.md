<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Spectral examples sequence

This page gives a reading and running order through three programs that add one
concept at a time: a spectral model, simulator orchestration, and a conserved nonlinear
model.

## Sequence

| Step | Executable | New idea |
|------|------------|----------|
| 1 | `04_diffusion_model` | Apply a backward-Euler multiplier to a distributed Fourier field |
| 2 | `05_simulator` | Add `pfc::sim::run`, `Time`, host ICs, and lifecycle orchestration |
| 3 | `12_cahn_hilliard` | Declare conserved nonlinear physics and advance it with native ETD1 |

Read the Quarto lessons for [implicit heat](https://vtt-propertune.github.io/OpenPFC/learn/heat/implicit-euler/),
[driver hooks](https://vtt-propertune.github.io/OpenPFC/learn/heat/driver-hooks/), and
[Cahn–Hilliard](https://vtt-propertune.github.io/OpenPFC/learn/phase-fields/cahn-hilliard/).
They develop the equations, show source excerpts and give independent checks.
The [snapshot-series lesson](https://vtt-propertune.github.io/OpenPFC/learn/output/vtk-series/) covers
visualization output separately.

## Build and run

From the repository root:

```bash
./scripts/build.sh

mpirun -n 1 ./builds/release/examples/04_diffusion_model
mpirun -n 1 ./builds/release/examples/05_simulator
mpirun -n 1 ./builds/release/examples/12_cahn_hilliard
```

Begin with one rank. After all three programs exit with status zero, repeat with
a rank count supported by your launcher allocation and the example dimensions.
The programs do not share one mandatory success message; see
[Example run output](../reference/example_run_output.md).

## Read the sources

| File | Look for |
|------|----------|
| [`examples/heat/implicit-euler/model.cpp`](../../examples/heat/implicit-euler/model.cpp) | Field allocation, Fourier-space operator construction, and forward/backward transforms |
| [`examples/heat/driver-hooks/model.cpp`](../../examples/heat/driver-hooks/model.cpp) | `pfc::sim::run`, host IC callable, time control |
| [`examples/phase-fields/cahn-hilliard/model.cpp`](../../examples/phase-fields/cahn-hilliard/model.cpp) | Linear/nonlinear symbols, pointwise cubic, state declaration and diagnostics |

Treat these compiled examples as the code-level source of truth. This page
explains their order rather than copying their implementations.

## Continue by goal

| Goal | Next document |
|------|---------------|
| Understand `Domain`, decomposition, and FFT first | [Library basics](../getting_started/01-basics/README.md) |
| Build a JSON-driven application | [Minimal custom application](custom_app_minimal.md) |
| Understand JSON-to-simulator wiring | [Application pipeline](../user_guide/app_pipeline.md) |
| Inspect output in ParaView | [VTK and ParaView workflow](vtk_paraview_workflow.md) |
| Find every runnable example | [Examples catalog](../reference/examples_catalog.md) |
| Look up API-focused snippets | [API examples walkthrough](../reference/api_examples_walkthrough.md) |
| Run linear \(k^4\) surface diffusion as a JSON app | [`apps/surface_diffusion`](../../apps/surface_diffusion/README.md) (`#79`) |
| Run odd-order \(ik^3\)/\(ik^5\) Kawahara waves | [`apps/kawahara`](../../apps/kawahara/README.md) (`#80`) |
| Run sixth-order EHD film under a flexible plate | [`apps/ehd_film`](../../apps/ehd_film/README.md) (`#81`) |
| Run Helmholtz–Navier strain-gradient elasticity | [`apps/gradient_elasticity`](../../apps/gradient_elasticity/README.md) (`#82`) |
