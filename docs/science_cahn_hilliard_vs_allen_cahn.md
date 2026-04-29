<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Science note: Cahn–Hilliard (`examples/`) vs Allen–Cahn (`apps/`)

Both names describe **interface-driven** continuum models, but the **shipped OpenPFC entry points** differ in dimensionality, numerics, and how you run them.

## `12_cahn_hilliard` (example)

| | |
|---|---|
| **What** | Spectral **Cahn–Hilliard–style** split operator on a structured grid; nonlinear + biharmonic-type stiffening via operators in Fourier space. |
| **Dimension** | Often **2D slab** (`Lz = 1`) in the source; grid sizes set in C++. |
| **Driver** | Custom `main` + `MPI_Worker` — **not** JSON `App`. |
| **Output** | **VTK** time series (`.vti`) — good for ParaView ([`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md)). |
| **Use** | Learn **spectral `Model` + VTK** together; see [`spectral_examples_sequence.md`](tutorials/spectral_examples_sequence.md). |

## `allen_cahn` (application)

| | |
|---|---|
| **What** | **2D Allen–Cahn**-type **reaction–diffusion** with a double-well potential; **finite differences** with halos on a 2D grid. |
| **Driver** | **CLI arguments** (no JSON `App`). |
| **Output** | Optional **PNG** snapshots (gather to rank 0) for quick visuals ([`io_results.md`](io_results.md)). |
| **Use** | Lightweight **visual sanity check**, FD + halo patterns ([`halo_exchange.md`](halo_exchange.md)), GPU variants when built. |

## Choosing one

- **Spectral + VTK + MPI FFT stack:** start with **`04` → `05` → `12`** ([`tutorials/spectral_examples_sequence.md`](tutorials/spectral_examples_sequence.md)).  
- **2D PNG + explicit FD:** use **`allen_cahn`** ([`apps/allen_cahn/README.md`](../apps/allen_cahn/README.md), [`applications.md`](applications.md)).  
- **3D PFC production + JSON:** use **tungsten** ([`science_tungsten_quicklook.md`](science_tungsten_quicklook.md)).

## See also

- [`showcase.md`](showcase.md) — figures mapped to examples/apps  
- [`examples_catalog.md`](examples_catalog.md) — full target list  
