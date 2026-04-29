<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Results I/O (binary, VTK, PNG)

OpenPFC separates the kernel interface `ResultsWriter` from frontend implementations under `include/openpfc/frontend/io/`. How you attach writers depends on whether you use the JSON-driven `App` path or a custom `main`.

## `ResultsWriter` (kernel)

[`include/openpfc/kernel/simulation/results_writer.hpp`](../include/openpfc/kernel/simulation/results_writer.hpp) — abstract hook the `Simulator` calls when it is time to persist fields. Implementations live in the frontend (binary/VTK) or in your app.

## Binary output (MPI-IO)

[`include/openpfc/frontend/io/binary_writer.hpp`](../include/openpfc/frontend/io/binary_writer.hpp) — `BinaryWriter`: raw binary, collective MPI-IO. Documented caveats: all ranks in the communicator must participate consistently in `write()` to avoid deadlock.

### JSON-driven `App` path

[`simulation_wiring.hpp`](../include/openpfc/frontend/ui/simulation_wiring.hpp) `add_result_writers_from_json` registers `BinaryWriter` only: for each `fields[]` entry it uses `field["data"]` as the path template. There is no VTK branch in that helper today—VTK is attached in code (see below).

Requirements in settings: `saveat > 0`, `fields` array with `name` and `data`.

## VTK (ParaView / VisIt)

[`include/openpfc/frontend/io/vtk_writer.hpp`](../include/openpfc/frontend/io/vtk_writer.hpp) — `VTKWriter`: `.vti` / `.pvti` output. Typical use is programmatic: construct `VTKWriter`, `set_domain`, `set_origin`, `set_spacing`, then `add_results_writer` or call from your step loop. See `examples/11_write_results.cpp` and Doxygen on `VTKWriter`.

## PNG (2D grayscale, quick look)

[`include/openpfc/frontend/io/png_writer.hpp`](../include/openpfc/frontend/io/png_writer.hpp) — `pfc::io::write_mpi_scalar_field_png_xy`: gathers a single z-slab (`nz == 1` globally) to rank 0 and writes an 8-bit grayscale PNG. Used for lightweight visualization (e.g. Allen–Cahn `apps/allen_cahn`), not the main spectral `App` JSON pipeline.

## Choosing a path

| Goal | Mechanism |
|------|-----------|
| Large production runs, restarts | `BinaryWriter` + `BinaryReader`; match JSON `fields`/`data` paths. |
| Interactive visualization | `VTKWriter` from code or extend wiring to register it. |
| Quick 2D snapshot | `png_writer.hpp` helpers |

## See also

- [`app_pipeline.md`](app_pipeline.md) — where `add_result_writers_from_json` runs  
- [`configuration.md`](configuration.md) — config file overview  
- [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md) — run once, inspect binary or PNG output  
- [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md) — `11_write_results` / `12_cahn_hilliard` + ParaView  
- [`learning_paths.md`](learning_paths.md) — documentation tracks by role  
