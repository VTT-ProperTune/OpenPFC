<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Performance profiling

OpenPFC records per-step frames in memory during a run and performs a single MPI collective export at the end (no per-step disk I/O). The API lives under `include/openpfc/kernel/profiling/`; the JSON-driven `pfc::ui::App` reads a `profiling` configuration block.

## Configuration (`App` JSON / TOML)

```json
"profiling": {
  "enabled": true,
  "format": "json",
  "output": "my_run_profile",
  "memory_samples": false,
  "print_report": false,
  "regions": ["custom/kernel", "custom/solver"],
  "run_id": "",
  "export_metadata": {}
}
```

| Field | Meaning |
|-------|---------|
| enabled | If true, allocate a `ProfilingSession` with a metric catalog and record one frame per time step on each MPI rank. |
| format | `json` (default if unknown), `hdf5`, or `both` (JSON + HDF5). Extensions: `.json`, `.h5`. Legacy values `csv`, `csv_hdf5`, `csv+hdf5` are accepted but CSV is no longer written: `csv` maps to JSON only; `csv_hdf5` / `csv+hdf5` map to `both` (JSON + HDF5), with a rank 0 stderr notice. |
| output | Base path without extension. |
| memory_samples | If true, each frame stores RSS (`/proc/self/status` on Linux), model heap bytes, and FFT heap bytes (extra cost per step). |
| print_report | If true, print a [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)-style table after file export. `App` uses `print_profiling_timer` with MPI gather: all ranks participate in a collective gather (same packed layout as export), and rank 0 prints a table that combines per-rank timer totals (default: mean across ranks of each rank’s summed inclusive time per path; see `ProfilingPrintOptions::mpi_aggregate_stat`). `%tot` uses the sum of `wall_denominator_metric` (default `wall_step`) over all gathered frames. |
| regions | Optional array of extra `/`-separated paths. Each entry adds that path and all parent prefixes to the catalog (e.g. `a/b` adds `a` and `a/b`). All MPI ranks must use the same config so the catalog matches for `MPI_Gatherv`. |
| run_id | If non-empty, profiling export uses schema v3 (namespaced HDF5/JSON under `runs/<id>/`). If empty, `SLURM_JOB_ID` or `OPENPFC_PROFILING_RUN_ID` is used when set; otherwise export stays schema v2 (legacy single-run layout). |
| export_metadata | Optional object merged into export metadata (HDF5 attributes on the run group, JSON `metadata`). `App` also fills `domain_lx`, `domain_ly`, `domain_lz` from `domain` and Slurm-related keys when the corresponding environment variables exist. |

Default catalog paths (always present): `communication`, `fft`, `gradient`.

Avoid extra region names that collide with reserved HDF5 names under `/openpfc/profiling/` (`frame_scalars`, `frame_metric_names`) so file creation does not fail.

HDF5 export requires configuring CMake with `OpenPFC_ENABLE_HDF5=ON` and a system HDF5 installation.

## Tungsten CPU (spectral pipelines)

The CPU Tungsten model ([`apps/tungsten/include/tungsten/cpu/tungsten_model.hpp`](../../apps/tungsten/include/tungsten/cpu/tungsten_model.hpp)) uses `OPENPFC_PROFILE` with nested paths:

- `gradient/mean_field` — mean-field filter: `gradient/mean_field/forward`, `gradient/mean_field/multiply`, `gradient/mean_field/backward`
- `gradient/evolve` — exponential integration step: `gradient/evolve/forward`, `gradient/evolve/multiply`, `gradient/evolve/backward`

List these under `profiling.regions` in the input file if you want a fixed catalog from step one. `format` values `hdf5` and `both` require `OpenPFC_ENABLE_HDF5=ON` at build time.

## What each frame stores (export schema)

After gather on rank 0, each frame has:

1. `scalars`: array of `double` values, in the same order as root `frame_metric_names` (see below). Semantics:

| Index | Name (`frame_metric_names`) | Description |
|-------|-----------------------------|-------------|
| 0 | step | Simulation step index (`Time::get_increment()`). |
| 1 | mpi_rank | MPI rank of the row. |
| 2 | wall_step | `App`: MPI barrier-wrapped step time passed to `openpfc_end_frame_step_wall_and_memory`. Demos may use `set_frame_metric_elapsed_since_begin("wall_step")` or `openpfc_end_frame_memory_only_wall_from_clock` (seconds). |
| 3 | rss_bytes | Process RSS if `memory_samples` is true, else `0`. |
| 4 | model_heap_bytes / 5 heap_secondary_bytes | Allocator-reported bytes (OpenPFC: model vs second bucket, e.g. FFT workspace) if `memory_samples` is true. |

2. `regions`: nested JSON object tree mirroring `/`-separated paths in `region_paths`, each node holding `inclusive` and `exclusive` seconds (same semantics as before).

Inclusive time for a path is the wall time of that region (nested scopes included). Exclusive time is inclusive minus the sum of children’s inclusive times (TimerOutputs-style nesting). Manual `record_time(path, dt)` adds `dt` to both inclusive and exclusive for that path.

`wall_step`: set explicitly (e.g. barrier duration from `measure_barriered`) or via `set_frame_metric_elapsed_since_begin("wall_step")` after `begin_frame`. OpenPFC `App` uses `openpfc_begin_frame_with_step_and_rank` and `openpfc_end_frame_step_wall_and_memory` from `openpfc_frame_metrics.hpp`.

Region times: use `ProfilingTimedScope`, `add_recorded_time` (additive), or `assign_recorded_time(path, seconds)` (overwrite scratch for that path). `App` records the FFT meter with `assign_recorded_time("fft", fft_seconds)` after each step (path must exist in the catalog; default `with_defaults_and_extras` includes `fft`). Tests may use `openpfc_end_frame_with_fft_region_wall_and_memory` to set the `fft` region time and default scalars in one call.

`communication`: halo wait paths call `record_time` when a profiling context is active during `step()`.

Interpretation: `wall_step` includes all work in `step()`. Compare with the sum of region times only where regions are non-overlapping or use exclusive times for nested hierarchies.

## Runtime API

- `ProfilingMetricCatalog` — immutable ordered path list; `with_defaults_and_extras(extra_paths)` or `from_paths_only(paths)` (no built-in regions).
- `ProfilingSession(catalog, frame_metric_names)` — second argument is the ordered list of per-frame scalar names (use `{}` for region timers only). `set_frame_metric_elapsed_since_begin(name)` fills a metric with elapsed time since `begin_frame`. OpenPFC `App` uses helpers in `openpfc_frame_metrics.hpp` (`openpfc_begin_frame_with_step_and_rank`, `openpfc_end_frame_step_wall_and_memory`, etc.) and `ProfilingSession::openpfc_default_frame_metrics()` for the default name list.
- `ProfilingContextScope` — RAII: set the thread-local active session for the duration of `step()` so low-level code can call `record_time` without passing pointers everywhere.
- `record_time(std::string_view path, double seconds)` — add elapsed time to a catalog path (no-op if no session or unknown path).
- `assign_recorded_time(path, seconds)` on the session — set inclusive/exclusive for that path for the current frame (authoritative meter values, etc.).
- `ProfilingTimedScope` — RAII nested timer using `steady_clock`; push/pop stack on the active session for inclusive/exclusive accounting. Paths are auto-registered in the catalog on first use (`ensure_path`); all MPI ranks must use the same path strings.
- `ProfilingManualScope` — same stack as `ProfilingTimedScope`, but `stop()` ends the interval before destruction, `restart(path)` stops and starts a new region (reuse one variable without nested braces), optional default construct + `start(path)`. Still LIFO with other scopes.
- `OPENPFC_PROFILE("path") { … }` — macro (unique local name via `__LINE__` / `__COUNTER__`) wrapping `ProfilingTimedScope`. `PFC_PROFILE_SCOPE("path")` is the single-statement form.
- `print_profiling_timer(std::ostream &, const ProfilingSession &, const ProfilingPrintOptions &)` — aggregate all committed frames and print a hierarchical table (section / ncalls / time / %tot / avg). `print_profiling_timer(std::ostream &, const ProfilingPrintOptions &)` uses `current_session()` when it has frames (no session pointer).
- `print_profiling_timer(std::ostream &, MPI_Comm, const ProfilingSession &, const ProfilingPrintOptions &)` — optional MPI path: when `mpi_aggregate_stdout` is true and `MPI_Comm_size` > 1, all ranks must call it; rank 0 gathers and prints cross-rank combined statistics (`mpi_aggregate_stat`: `mean` / `sum` / `min` / `max` / `median` on per-rank per-path totals). When `mpi_aggregate_stdout` is false or the communicator has size 1, only rank 0 prints the local session (same as the overload without `MPI_Comm`).

`ProfilingPrintOptions` controls title, ascii_lines, `sort_by_time`, `show_exclusive_column`, `wall_denominator_metric` (%tot denominator), `mpi_aggregate_stdout`, and `mpi_aggregate_stat`.

Canonical string constants: `kProfilingRegionFft`, `kProfilingRegionCommunication`, `kProfilingRegionGradient` in `profiling/names.hpp`.

Include the umbrella header:

```cpp
#include <openpfc/kernel/profiling/profiling.hpp>
```

Example: `examples/profiling_timer_report.cpp` builds a session with an empty catalog, auto-registers paths via `OPENPFC_PROFILE` and `ProfilingManualScope`, and prints `print_profiling_timer`.

## Export format (schema versions 2 and 3)

Full specification, HDF5 layout, gather row order, and migration from schema 1: [profiling_export_schema.md](profiling_export_schema.md).

Summary: rank 0 writes `schema_version`: 2 or 3 with `ranks`: one entry per MPI process, each with `mpi_rank`, `n_frames`, and `frames` (nested `regions` mirror `/`-separated paths). `total_frames` is the sum of per-rank frame counts. HDF5 v2 stores the hierarchy under `/openpfc/profiling/ranks/<id>/`; v3 nests the same payload under `/openpfc/profiling/runs/<run_id>/…` for collision-free merges (see the linked doc).

Local / CI without Slurm: set `export OPENPFC_PROFILING_RUN_ID=mytest` so exports use v3 and a stable namespace for debugging.

### Python (JSON, schema 2)

```python
import json
with open("my_run_profile.json") as f:
    d = json.load(f)
assert d["schema_version"] == 2
names = d["frame_metric_names"]
wall_i = names.index("wall_step")
wall = []
fft_inc = []
for rank_entry in d["ranks"]:
    for fr in rank_entry["frames"]:
        wall.append(fr["scalars"][wall_i])
        fft_inc.append(fr["regions"]["fft"]["inclusive"])
```

### Python (HDF5, schema 2)

```python
import h5py
with h5py.File("my_run_profile.h5", "r") as f:
    g = f["openpfc/profiling"]
    names = [n.decode() if isinstance(n, bytes) else n for n in g["frame_metric_names"][:]]
    wall_i = names.index("wall_step")
    r0 = g["ranks/0"]
    fs = r0["frame_scalars"][:, :]
    wall = fs[:, wall_i]
    fft = r0["fft/inclusive"][:]
```

Schema v3 (namespaced run): if `g.attrs["schema_version"] == 3`, open `run_id = list(f["openpfc/profiling/runs"].keys())[0]` (or use a known job id), then replace `g` with `f["openpfc/profiling/runs"][run_id]` in the snippet above (`frame_metric_names`, `ranks/0`, …).

## MPI and memory

`finalize_and_export` uses `MPI_Gatherv` to assemble all ranks’ frames on rank 0, then writes files. Memory on root scales with `n_frames_total × (|frame_metric_names| + 2 × |region_paths|) × sizeof(double)` plus JSON overhead. The catalog must be identical on every rank (same `profiling.regions`, same frame metric name list, and defaults).

## CMake

| Option | Role |
|--------|------|
| `OpenPFC_ENABLE_HDF5` | Optional HDF5 linkage for `.h5` export. |
| `OpenPFC_PROFILING_LEVEL` | `0` strips `OPENPFC_PROFILE` / `PFC_PROFILE_SCOPE` to no-ops; `>0` enables timed scopes. |

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
