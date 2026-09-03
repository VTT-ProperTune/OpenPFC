<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Results I/O (binary, VTK, PNG)

OpenPFC separates the kernel interface `ResultsWriter` from frontend implementations under `include/openpfc/frontend/io/`. How you attach writers depends on whether you use the JSON-driven `App` path or a custom `main`.

## `ResultsWriter` (kernel)

[`include/openpfc/kernel/simulation/results_writer.hpp`](../../include/openpfc/kernel/simulation/results_writer.hpp) — abstract writer that sessions call from the `on_save` hook of `pfc::sim::run` when `pfc::time::do_save` is true. Implementations live in the frontend (binary/VTK) or in your app.

### Session wiring

Sessions build named writers from JSON `fields[]` with
`pfc::ui::parse_result_writers_from_json` (below), call
`pfc::apply_writer_domain(writer, field)` so global size / local box come from
the `Field`, and write each field on the `on_save` hook while bumping their own
result counter. Unit tests can drive a writer directly with a small `Field` and
a mock writer; no simulator object is needed.

## Binary output (MPI-IO)

[`include/openpfc/frontend/io/binary_writer.hpp`](../../include/openpfc/frontend/io/binary_writer.hpp) — `BinaryWriter`: raw binary, collective MPI-IO. Documented caveats: all ranks in the communicator must participate consistently in `write()` to avoid deadlock. Both `BinaryWriter::set_domain` and kernel `BinaryReader::set_domain` fail closed on invalid geometry (non-positive extents, negative offsets, or a piece outside the global box); see [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md).

**Format (layout, filename `printf` pattern, collectives):** [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md).

### JSON-driven `App` path

[`simulation_wiring_writers.hpp`](../../include/openpfc/frontend/ui/simulation_wiring_writers.hpp) `parse_result_writers_from_json` takes a **`ResultsWriterCatalog`** at the call site (e.g. `default_results_writer_catalog()` for built-in `binary`). It returns named `BinaryWriter`s (or catalog `vtk`/`hdf5`): for each `fields[]` entry it uses `field["data"]` as the path template. Sessions attach those writers on the `on_save` hook.

Requirements in settings: `saveat > 0`, `fields` array with `name` and `data`.

## VTK (ParaView / VisIt)

[`include/openpfc/frontend/io/vtk_writer.hpp`](../../include/openpfc/frontend/io/vtk_writer.hpp) — `VTKWriter`: `.vti` / `.pvti` output. Extent/origin/spacing and local point-count checks are implemented in [`vtk_writer_validate.hpp`](../../include/openpfc/frontend/io/vtk_writer_validate.hpp) (`pfc::io::vtk_validate`), separate from XML and file I/O. Typical use is programmatic: construct `VTKWriter`, `set_domain`, `set_origin`, `set_spacing`, then `add_results_writer` or call from your step loop. See `examples/11_write_results.cpp` and Doxygen on `VTKWriter`.

## PNG (2D grayscale, quick look)

[`include/openpfc/frontend/io/png_writer.hpp`](../../include/openpfc/frontend/io/png_writer.hpp) — `pfc::io::write_mpi_scalar_field_png_xy`: gathers a single z-slab (`nz == 1` globally) to rank 0 and writes an 8-bit grayscale PNG. Used for lightweight visualization (e.g. Allen–Cahn `apps/allen_cahn`), not the main spectral `App` JSON pipeline.

**Collective contract:** All ranks in the communicator must participate consistently. Buffer size validation is fail-closed (communicator-wide agreement via `MPI_Allreduce`) before the `MPI_Allgather`/`MPI_Gatherv` calls, preventing deadlocks when a rank provides an incorrectly sized `local_field`.

## Choosing a path

| Goal | Mechanism |
|------|-----------|
| Large production runs, restarts | `BinaryWriter` + `BinaryReader`; match JSON `fields`/`data` paths. |
| Interactive visualization | `VTKWriter` from code or extend wiring to register it. |
| Quick 2D snapshot | `png_writer.hpp` helpers |

## See also

- [`binary_field_io_spec.md`](../reference/binary_field_io_spec.md) — normative binary field file description  
- [`postprocess_binary_fields.md`](postprocess_binary_fields.md) — offline analysis of raw binary fields  
- [`app_pipeline.md`](app_pipeline.md) — where `parse_result_writers_from_json` runs  
- [`configuration.md`](configuration.md) — config file overview  
- [`tutorials/end_to_end_visualization.md`](../tutorials/end_to_end_visualization.md) — run once, inspect binary or PNG output  
- [`tutorials/vtk_paraview_workflow.md`](../tutorials/vtk_paraview_workflow.md) — `11_write_results` / `12_cahn_hilliard` + ParaView  
- [`learning_paths.md`](../learning_paths.md) — documentation tracks by role  
