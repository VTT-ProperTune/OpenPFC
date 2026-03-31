<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Performance profiling

OpenPFC records **per-step frames in memory** during a run and performs a **single MPI collective export** at the end (no per-step disk I/O). The API lives under **`include/openpfc/kernel/profiling/`**; the JSON-driven **`pfc::ui::App`** reads a **`profiling`** configuration block.

## Configuration (`App` JSON / TOML)

```json
"profiling": {
  "enabled": true,
  "format": "json",
  "output": "my_run_profile",
  "memory_samples": false,
  "print_report": false,
  "regions": ["custom/kernel", "custom/solver"]
}
```

| Field | Meaning |
|-------|---------|
| **enabled** | If true, allocate a `ProfilingSession` with a **metric catalog** and record one frame per time step on each MPI rank. |
| **format** | `json` (default if unknown), `hdf5`, `csv`, `both` (JSON + HDF5), or `csv_hdf5` / `csv+hdf5` (CSV + HDF5). Extensions: **`.json`**, **`.h5`**, **`.csv`**. |
| **output** | Base path **without** extension. |
| **memory_samples** | If true, each frame stores RSS (`/proc/self/status` on Linux), model heap bytes, and FFT heap bytes (extra cost per step). |
| **print_report** | If true, **rank 0** prints a [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)-style table after file export. Data is **this rank’s** frames only (not MPI-reduced). |
| **regions** | Optional array of extra **`/`-separated paths**. Each entry adds that path and **all parent prefixes** to the catalog (e.g. `a/b` adds `a` and `a/b`). **All MPI ranks must use the same config** so the catalog matches for `MPI_Gatherv`. |

Default catalog paths (always present): **`communication`**, **`fft`**, **`gradient`**.

Avoid extra region names that equal HDF5 top-level dataset names under **`/openpfc/profiling/`** (`step`, `mpi_rank`, `wall_step`, …) so file creation does not fail.

HDF5 export requires configuring CMake with **`OpenPFC_ENABLE_HDF5=ON`** and a system HDF5 installation.

## Tungsten CPU (spectral pipelines)

The CPU Tungsten model ([`apps/tungsten/include/tungsten/cpu/tungsten_model.hpp`](../apps/tungsten/include/tungsten/cpu/tungsten_model.hpp)) uses **`OPENPFC_PROFILE`** with nested paths:

- **`gradient/mean_field`** — mean-field filter: **`gradient/mean_field/forward`**, **`gradient/mean_field/multiply`**, **`gradient/mean_field/backward`**
- **`gradient/evolve`** — exponential integration step: **`gradient/evolve/forward`**, **`gradient/evolve/multiply`**, **`gradient/evolve/backward`**

List these under **`profiling.regions`** in the input file if you want a fixed catalog from step one. **`format`** values **`hdf5`**, **`both`**, **`csv_hdf5`**, etc. require **`OpenPFC_ENABLE_HDF5=ON`** at build time.

## What each frame stores (schema version 2)

After gather on rank 0, each row has:

| Field | Description |
|-------|-------------|
| **step** | Simulation step index (`Time::get_increment()`). |
| **mpi_rank** | MPI rank of the row. |
| **wall_step** | **`App`:** MPI barrier-wrapped step time via **`set_frame_wall_step`**. Otherwise: **`steady_clock`** span from **`begin_step_frame`** to **`end_step_frame(rss, …)`** if **`set_frame_wall_step`** was not used (seconds). |
| **rss_bytes** | Process RSS if `memory_samples` is true, else `0`. |
| **model_heap_bytes** / **fft_heap_bytes** | Allocator-reported bytes if `memory_samples` is true. |
| **Per-catalog-path** | **Inclusive** and **exclusive** seconds for each path in the catalog (see below). |

**Inclusive** time for a path is the wall time of that region (nested scopes included). **Exclusive** time is inclusive minus the sum of **children’s inclusive** times (TimerOutputs-style nesting). Manual **`record_time(path, dt)`** adds `dt` to both inclusive and exclusive for that path.

**`wall_step`:** either **`set_frame_wall_step(seconds)`** after **`begin_step_frame`** (e.g. MPI barrier duration from **`measure_barriered`**) or, if you omit that, the elapsed **`steady_clock`** time from **`begin_step_frame`** to **`end_step_frame(rss, …)`**.

**Region times:** use **`ProfilingTimedScope`**, **`add_recorded_time`** (additive), or **`assign_recorded_time(path, seconds)`** (overwrite scratch for that path). **`App`** records the FFT meter with **`assign_recorded_time("fft", fft_seconds)`** after each step (path must exist in the catalog; default **`with_defaults_and_extras`** includes **`fft`**). Any other instrumented quantity can use **`assign_recorded_time`** or **`add_recorded_time`** for paths in the catalog.

A legacy overload **`end_step_frame(wall_step, fft_seconds, …)`** still overwrites the **`fft`** path and sets **`wall_step`** in one call (mainly for tests).

**`communication`:** halo wait paths call **`record_time`** when a profiling context is active during `step()`.

**Interpretation:** `wall_step` includes all work in `step()`. Compare with the sum of region times only where regions are non-overlapping or use exclusive times for nested hierarchies.

## Runtime API

- **`ProfilingMetricCatalog`** — immutable ordered path list; **`with_defaults_and_extras(extra_paths)`** or **`from_paths_only(paths)`** (no built-in regions).
- **`ProfilingSession`** — owns frame buffers and the catalog; **`ensure_path`** (usually implicit from scopes / **`add_recorded_time`**); **`begin_step_frame`**, optional **`set_frame_wall_step`**, **`end_step_frame(rss, model_heap, fft_heap)`**, **`finalize_and_export`**; optional **`reset_report_clock`** for console report wall-clock line.
- **`ProfilingContextScope`** — RAII: set the **thread-local** active session for the duration of `step()` so low-level code can call **`record_time`** without passing pointers everywhere.
- **`record_time(std::string_view path, double seconds)`** — add elapsed time to a catalog path (no-op if no session or unknown path).
- **`assign_recorded_time(path, seconds)`** on the session — set inclusive/exclusive for that path for the current frame (authoritative meter values, etc.).
- **`ProfilingTimedScope`** — RAII nested timer using **`steady_clock`**; push/pop stack on the active session for **inclusive/exclusive** accounting. Paths are **auto-registered** in the catalog on first use (**`ensure_path`**); all MPI ranks must use the same path strings.
- **`ProfilingManualScope`** — same stack as **`ProfilingTimedScope`**, but **`stop()`** ends the interval before destruction, **`restart(path)`** stops and starts a new region (reuse one variable without nested braces), optional default construct + **`start(path)`**. Still **LIFO** with other scopes.
- **`OPENPFC_PROFILE("path") { … }`** — macro (unique local name via **`__LINE__` / `__COUNTER__`**) wrapping **`ProfilingTimedScope`**. **`PFC_PROFILE_SCOPE("path")`** is the single-statement form.
- **`print_profiling_timer(std::ostream &, const ProfilingSession &, const ProfilingPrintOptions &)`** — aggregate all committed frames and print a hierarchical table (section / ncalls / time / %tot / avg). **`print_profiling_timer(std::ostream &, const ProfilingPrintOptions &)`** uses **`current_session()`** when it has frames (no session pointer).

`ProfilingPrintOptions` controls **title**, **ascii_lines**, **sort_by_time**, and **show_exclusive_column**.

Canonical string constants: **`kProfilingRegionFft`**, **`kProfilingRegionCommunication`**, **`kProfilingRegionGradient`** in **`profiling/names.hpp`**.

Include the umbrella header:

```cpp
#include <openpfc/kernel/profiling/profiling.hpp>
```

**Example:** **`examples/profiling_timer_report.cpp`** builds a session with an empty catalog, auto-registers paths via **`OPENPFC_PROFILE`** and **`ProfilingManualScope`**, and prints **`print_profiling_timer`**.

## Export formats (schema version 2)

### JSON

Rank 0 writes **`schema_version`: 2**, **`catalog`** (array of path strings), **`n_frames`**, **`n_ranks`**, and **`frames`**: an array of objects with **`step`**, **`mpi_rank`**, **`wall_step`**, **`rss_bytes`**, **`model_heap_bytes`**, **`fft_heap_bytes`**, and **`regions`**: a nested object tree mirroring `/`-separated paths, with each leaf (and internal catalog node) holding **`inclusive`** and **`exclusive`** numbers.

### HDF5

Under **`/openpfc/profiling/`**: 1D datasets **`step`**, **`mpi_rank`**, **`wall_step`**, **`rss_bytes`**, **`model_heap_bytes`**, **`fft_heap_bytes`**. For each catalog path, **nested groups** follow the path segments (e.g. `communication` → group `communication`; `outer/inner` → groups `outer` then `inner`), each leaf group containing 1D datasets **`inclusive`** and **`exclusive`** (length `n_frames`). Attributes **`schema_version`** (2) and **`openpfc_version`** sit on the profiling group.

### CSV

Flattened columns: metadata fields above, then for each catalog path (with `/` replaced by `_` in the header) **`path_inclusive`** and **`path_exclusive`**.

### Python (JSON v2)

```python
import json
with open("my_run_profile.json") as f:
    d = json.load(f)
assert d["schema_version"] == 2
wall = [fr["wall_step"] for fr in d["frames"]]
fft_inc = [fr["regions"]["fft"]["inclusive"] for fr in d["frames"]]
```

### Python (HDF5 v2)

```python
import h5py
with h5py.File("my_run_profile.h5", "r") as f:
    g = f["openpfc/profiling"]
    wall = g["wall_step"][:]
    fft_i = g["fft/inclusive"][:]
```

## MPI and memory

`finalize_and_export` uses **`MPI_Gatherv`** to assemble all ranks’ frames on **rank 0**, then writes files. Memory on root scales with **`n_frames_total × (6 + 2 × |catalog|) × sizeof(double)`** plus JSON overhead. **The catalog must be identical on every rank** (same `profiling.regions` and defaults).

## CMake

| Option | Role |
|--------|------|
| `OpenPFC_ENABLE_HDF5` | Optional HDF5 linkage for `.h5` export. |
| `OpenPFC_PROFILING_LEVEL` | `0` strips **`OPENPFC_PROFILE`** / **`PFC_PROFILE_SCOPE`** to no-ops; `>0` enables timed scopes. |

## Related headers

| Header | Role |
|--------|------|
| `profiling/profiling.hpp` | Umbrella include (session, scopes, macros, timer report, …) |
| `profiling/session.hpp` | `ProfilingSession`, `ProfilingExportOptions` |
| `profiling/timer_report.hpp` | `ProfilingPrintOptions`, `print_profiling_timer` |
| `profiling/metric_catalog.hpp` | `ProfilingMetricCatalog` |
| `profiling/context.hpp` | `ProfilingContextScope`, `record_time`, `current_session` |
| `profiling/region_scope.hpp` | `ProfilingTimedScope`, `ProfilingManualScope` |
| `profiling/profile_scope_macro.hpp` | `OPENPFC_PROFILE`, `PFC_PROFILE_SCOPE` (includes `config.hpp`) |
| `profiling/config.hpp` | `OPENPFC_PROFILING_LEVEL` |
| `profiling/names.hpp` | `kProfilingRegion*` string constants |
| `profiling/wall_clock.hpp` | `measure_barriered`, `mpi_wtime_now` |
| `profiling/mpi_stats.hpp` | `reduce_max_to_root`, `RankStats` |
| `profiling/memory_sample.hpp` | `try_read_process_rss_bytes` |
| `profiling/format.hpp` | `format_bytes` (memory reports) |
