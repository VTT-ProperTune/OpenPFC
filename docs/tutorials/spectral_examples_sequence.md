<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: spectral `examples/` sequence (diffusion → simulator → Cahn–Hilliard)

This page is a **reading and running order** through three `examples/` programs that stack concepts: a minimal spectral `Model`, the full `Simulator` loop, then a richer model with VTK output.

## Why this order?

| Step | Executable | New idea |
|------|------------|----------|
| 1 | `04_diffusion_model` | Subclass `Model`, `initialize` / `step`, spectral Laplacian on one field. |
| 2 | `05_simulator` | `Simulator`, `Time`, `FieldModifier`, orchestration without a JSON `App`. |
| 3 | `12_cahn_hilliard` | Nonlinear Cahn–Hilliard-style splitting, random IC, **VTK** time series. |

Together they mirror how research code grows: physics in `Model`, then lifecycle and I/O in `Simulator`, then outputs for visualization.

## Run commands

From your build directory (after [`INSTALL.md`](../../INSTALL.md) + successful `cmake --build`):

```bash
cd build
mpirun -n 4 ./examples/04_diffusion_model
mpirun -n 4 ./examples/05_simulator
mpirun -n 4 ./examples/12_cahn_hilliard
```

Adjust `-n` to your machine. Expect exit code **0**; each program prints different INFO — see [`../example_run_output.md`](../reference/example_run_output.md) for log *shape*.

## What to read in the sources

| File | Skim for |
|------|-----------|
| [`examples/04_diffusion_model.cpp`](../../examples/04_diffusion_model.cpp) | `opL` in Fourier space, `fft.forward` / `backward`, field registration. |
| [`examples/05_simulator.cpp`](../../examples/05_simulator.cpp) | How `Simulator` is constructed, modifiers attached, time loop. |
| [`examples/12_cahn_hilliard.cpp`](../../examples/12_cahn_hilliard.cpp) | Nonlinear operator split, `VtkWriter` cadence, `MPI_Worker`. |

## After this sequence

| Goal | Next document |
|------|----------------|
| JSON-driven `App` | [`custom_app_minimal.md`](custom_app_minimal.md), [`../app_pipeline.md`](../user_guide/app_pipeline.md) |
| VTK in more detail | [`vtk_paraview_workflow.md`](vtk_paraview_workflow.md) |
| Full example list + tiers | [`../examples_catalog.md`](../reference/examples_catalog.md) |
| Doxygen snippet curriculum | [`../api_examples_walkthrough.md`](../reference/api_examples_walkthrough.md) |

## See also

- [`../getting_started/01-basics/README.md`](../getting_started/01-basics/README.md) — textbook-style FFT + derivative walkthrough  
- [`../quickstart.md`](../quickstart.md) — first-run checklist  
