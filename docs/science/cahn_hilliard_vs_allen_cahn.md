<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Science note: Cahn–Hilliard vs Allen–Cahn

Both names describe **interface-driven** continuum models, but the **shipped OpenPFC entry points** differ in dimensionality, numerics, and how you run them.

## `cahn_hilliard` (application, `#77`)

| | |
|---|---|
| **What** | Conserved Cahn–Hilliard with a Fe–Cr-like regular-solution free energy; spectral ETD, \(L(k)\propto k^4\). |
| **Dimension** | 2D periodic slab in the shipped input (`Lz = 1`); 3D works if you set `Lz`. |
| **Driver** | JSON/TOML `SpectralETDSession` (same path as tungsten). |
| **Output** | VTK of composition `c` (`results/cahn_hilliard/c_%04d.vti`). |
| **Use** | Materials-science spinodal (475 °C embrittlement story); Catch2 checks mass, \(\lambda(k)\), and growth in the spinodal band. See [`apps/cahn_hilliard/README.md`](../../apps/cahn_hilliard/README.md). |

## `12_cahn_hilliard` (example)

| | |
|---|---|
| **What** | Spectral **Cahn–Hilliard–style** split operator on a structured grid; nonlinear + biharmonic-type stiffening via operators in Fourier space. |
| **Dimension** | Often **2D slab** (`Lz = 1`) in the source; grid sizes set in C++. |
| **Driver** | Custom `main` + `MPI_Worker` — **not** JSON `App`. |
| **Output** | **VTK** time series (`.vti`) — good for ParaView ([`tutorials/vtk_paraview_workflow.md`](../tutorials/vtk_paraview_workflow.md)). |
| **Use** | Learn **spectral `Model` + VTK** together; see [`spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md). |

## `allen_cahn` (application)

| | |
|---|---|
| **What** | **2D Allen–Cahn**-type **reaction–diffusion** with a double-well potential; **finite differences** with halos on a 2D grid. |
| **Driver** | **CLI arguments** (no JSON `App`). |
| **Output** | Optional **PNG** snapshots (gather to rank 0) for quick visuals ([`io_results.md`](../user_guide/io_results.md)). |
| **Checks** | The shipped app can **verify** seed growth (see [`apps/allen_cahn/README.md`](../../apps/allen_cahn/README.md)); exit code may be non-zero if thresholds are not met. |
| **Use** | Lightweight **visual sanity check**, FD + halo patterns ([`halo_exchange.md`](../concepts/halo_exchange.md)), GPU variants when built. |

## Choosing one

- **Spectral Cahn–Hilliard as a materials app:** **`apps/cahn_hilliard`**.  
- **Spectral + VTK teaching sequence:** start with **`04` → `05` → `12`** ([`tutorials/spectral_examples_sequence.md`](../tutorials/spectral_examples_sequence.md)).  
- **2D PNG + explicit FD:** use **`allen_cahn`** ([`apps/allen_cahn/README.md`](../../apps/allen_cahn/README.md), [`applications.md`](../user_guide/applications.md)).  
- **3D PFC production + JSON:** use **tungsten** ([`science_tungsten_quicklook.md`](tungsten_quicklook.md)).

## See also

- [`showcase.md`](../user_guide/showcase.md) — figures mapped to examples/apps  
- [`examples_catalog.md`](../reference/examples_catalog.md) — full target list  
