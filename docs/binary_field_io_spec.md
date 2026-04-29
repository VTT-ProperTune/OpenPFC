<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Binary field I/O specification (MPI-IO)

This document describes the **raw binary** files produced by `pfc::BinaryWriter` (including the JSON-driven path in `add_result_writers_from_json`). It is the contract you rely on for **checkpoints**, **restarts**, and **post-processing** when you are not using VTK or PNG.

**Implementation references:** [`include/openpfc/frontend/io/binary_writer.hpp`](../include/openpfc/frontend/io/binary_writer.hpp), [`include/openpfc/kernel/simulation/binary_reader.hpp`](../include/openpfc/kernel/simulation/binary_reader.hpp), [`include/openpfc/frontend/utils/utils.hpp`](../include/openpfc/frontend/utils/utils.hpp) (`format_with_number`).

## File contents

| Property | Value |
|----------|--------|
| **Layout** | A single **global** 3D array in **Fortran (column-major) order** (`MPI_ORDER_FORTRAN` in `MPI_Type_create_subarray`). |
| **Element type** | `double` for real fields (`MPI_DOUBLE`). `BinaryWriter` also supports `std::complex<double>` (`MPI_DOUBLE_COMPLEX`) when writing complex buffers. |
| **Byte order** | **Native** (`"native"` in `MPI_File_set_view`) — endianness matches the machine that wrote the file. |
| **Header / magic** | **None.** The file is only the raw payload for the MPI file view (no metadata block, no version tag). |
| **Per-rank data** | Each MPI rank writes its **local** brick; together the ranks cover the global grid without overlap. The view is built from `(global_size, local_size, local_offset)` passed to `set_domain()`. |

Each `write(increment, field)` call:

1. Builds the output filename from the template and `increment` (see below).  
2. Opens the file (truncate), sets the MPI-IO file view, performs **`MPI_File_write_all`**, closes the file.  
3. Is **collective** over the writer’s communicator (default `MPI_COMM_WORLD`): **every rank** in that communicator must participate in `write()` with a consistent domain layout, or the job can **deadlock**.

## Filename template and `increment`

The JSON `fields[].data` string is passed to `BinaryWriter` as the filename template.

- If the string contains **`%`**, it is passed to `printf`-style formatting with the **integer `increment`** supplied by the simulator (see [`simulation_wiring_writers.hpp`](../include/openpfc/frontend/ui/simulation_wiring_writers.hpp) and `BinaryWriter::write_mpi_binary`).  
  Examples: `./psi_%d.bin`, `./data/u_%04d.bin`.  
- If there is **no `%`**, the same path is used on every write (overwrites each time).

The `increment` value is advanced by the simulator according to configuration (see [`app_pipeline.md`](app_pipeline.md) and the `simulator` section in JSON).

## Reading back (`BinaryReader`)

`BinaryReader` uses the same **Fortran-ordered 3D subarray** view and **`MPI_File_read_all`** at displacement **0**, with element type **`MPI_DOUBLE`** in the public `read()` API shown in the header.

**Implications:**

- Files written as **pure `double`** real fields with the same `(global, local, offset)` decomposition can be read back for restart or `FileReader` initial conditions.  
- Match **communicator**, **decomposition**, and **datatype** when reading: a file written on *N* ranks is not automatically loadable on a different *N* without re-decomposition logic outside this low-level format.  
- For **complex** fields, prefer treating restart paths as **implementation-defined** unless your build uses matching writer/reader pairs for complex data (consult headers for your version).

## JSON configuration surface

When `saveat > 0` and `fields` is present, `add_result_writers_from_json` registers one `BinaryWriter` per `fields[]` entry:

- **`name`** — field label used by the simulator when dispatching the writer.  
- **`data`** — filename template as above.

Rank 0 may create parent directories for `data`. See [`io_results.md`](io_results.md) and [`configuration.md`](configuration.md).

## Post-processing without OpenPFC

Because there is **no file header**, external tools need **out-of-band** metadata: global `Lx, Ly, Lz`, dtype (`float64`), Fortran ordering, and how ranks map to subdomains if you concatenate manually. In practice, teams either:

- Read with **OpenPFC** (`BinaryReader` or existing tooling), or  
- Record metadata alongside runs (JSON/YAML sidecar), or  
- Use **VTK** export for visualization (`VTKWriter`; see [`tutorials/vtk_paraview_workflow.md`](tutorials/vtk_paraview_workflow.md)).

## See also

- [`io_results.md`](io_results.md) — writers overview  
- [`tutorials/end_to_end_visualization.md`](tutorials/end_to_end_visualization.md) — run that produces binaries  
- [`app_pipeline.md`](app_pipeline.md) — when writers are wired  
