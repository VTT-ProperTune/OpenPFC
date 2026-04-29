<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tutorial: VTK output and ParaView

OpenPFC can write **VTK Image Data** (`.vti` per rank, or combined pieces you open as a time series). The JSON `App` path registers **binary** writers by default; **VTK is usually attached in C++** (see [`../io_results.md`](../user_guide/io_results.md)). This tutorial uses the built-in `examples/` targets that already call `VTKWriter`.

## Prerequisites

- OpenPFC built with examples (`OpenPFC_BUILD_EXAMPLES=ON`, default).  
- MPI launcher from the same stack you used at configure time.  
- [ParaView](https://www.paraview.org/) (or another VTK-capable viewer) installed locally.

## Example 1: single `results.vti` (`11_write_results`)

This program builds a small `DiscreteField`, fills it with a simple pattern, and writes one file.

```bash
cd build
mpirun -n 2 ./examples/11_write_results
```

**Artifact:** `results.vti` in the **current working directory** (usually `build/`).

**ParaView:** *File → Open* → select `results.vti` → *Apply*. Choose the point array named `density` (see source: `examples/11_write_results.cpp`).

**Source:** [`examples/11_write_results.cpp`](../../examples/11_write_results.cpp) — shows `set_uri`, `set_field_name`, `set_domain`, `set_origin`, `set_spacing`, `initialize`, `write`.

## Example 2: time series (`12_cahn_hilliard`)

Cahn–Hilliard-style stepping with VTK snapshots every 10 steps:

```bash
cd build
mpirun -n 4 ./examples/12_cahn_hilliard
```

**Artifacts:** `cahn_hilliard_0000.vti`, `cahn_hilliard_0001.vti`, … in the working directory.

**ParaView:** *File → Open* → select **all** `cahn_hilliard_*.vti` → ParaView offers to group them as a time sequence; use the playback controls.

**Source:** [`examples/12_cahn_hilliard.cpp`](../../examples/12_cahn_hilliard.cpp) — `VtkWriter` updates `set_uri` each output.

## Tips

- **Paths:** Run from `build/` so outputs land where you expect; or use a job script that `cd`s to a known directory before `mpirun`.  
- **MPI ranks:** Decomposition affects how data are split across ranks; the sample sources show a simple pattern—if you change ranks or grid, re-check where files appear and that all ranks still participate where the writer requires collective behavior.  
- **Production JSON runs:** To get VTK from `tungsten`-style configs, add a `VTKWriter` in your model or wiring (same header as the examples); there is no VTK branch inside `add_result_writers_from_json` today.

## See also

- [`../io_results.md`](../user_guide/io_results.md) — `VTKWriter` vs `BinaryWriter`  
- [`../examples_catalog.md`](../reference/examples_catalog.md) — catalog entries for `11_write_results`, `12_cahn_hilliard`  
- [`../class_tour.md`](../reference/class_tour.md) — `ResultsWriter`, `DiscreteField`  
