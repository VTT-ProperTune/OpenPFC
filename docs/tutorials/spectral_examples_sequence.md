<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Spectral examples sequence

This page gives a reading and running order through three programs that add one
concept at a time: a spectral model, simulator orchestration, and a nonlinear
model with output.

## Sequence

| Step | Executable | New idea |
|------|------------|----------|
| 1 | `04_diffusion_model` | Subclass `Model`, implement `initialize` and `step`, and apply a Fourier-space operator |
| 2 | `05_simulator` | Add `Simulator`, `Time`, field modifiers, and lifecycle orchestration |
| 3 | `12_cahn_hilliard` | Add a nonlinear operator split, initial data, and VTK output |

Together they mirror how a research application usually grows: first verify the
physics step, then add framework-managed lifecycle and output.

## Build and run

From the repository root:

```bash
cmake --build build --target \
  04_diffusion_model \
  05_simulator \
  12_cahn_hilliard

mpirun -n 1 ./build/examples/04_diffusion_model
mpirun -n 1 ./build/examples/05_simulator
mpirun -n 1 ./build/examples/12_cahn_hilliard
```

Begin with one rank. After all three programs exit with status zero, repeat with
a rank count supported by your launcher allocation and the example dimensions.
The programs do not share one mandatory success message; see
[Example run output](../reference/example_run_output.md).

## Read the sources

| File | Look for |
|------|----------|
| [`examples/04_diffusion_model.cpp`](../../examples/04_diffusion_model.cpp) | Field allocation, Fourier-space operator construction, and forward/backward transforms |
| [`examples/05_simulator.cpp`](../../examples/05_simulator.cpp) | Simulator construction, modifiers, time control, and ownership |
| [`examples/12_cahn_hilliard.cpp`](../../examples/12_cahn_hilliard.cpp) | Nonlinear split, initial state, and VTK write cadence |

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
