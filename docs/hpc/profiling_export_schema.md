<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Profiling export schema (JSON and HDF5)

This document describes the on-disk layout written by `ProfilingSession::finalize_and_export` on MPI rank 0 after gathering packed frames from all ranks. The MPI transport is unchanged: each rank contributes a packed row buffer; rank 0 concatenates them in increasing MPI rank order.

## Schema versions

| Version | Summary |
|--------|---------|
| 1 (legacy) | Single flat `frames` array of length `n_frames` (all ranks’ frames concatenated). HDF5: global `frame_scalars` and per-path datasets of length `n_frames`. |
| 2 | Hierarchical: `ranks[]`, each with `mpi_rank`, `n_frames`, and `frames[]`. HDF5: `openpfc/profiling/` holds payload directly (`schema_version` = 2 on that group). |
| 3 (namespaced run) | Same payload as v2, but nested for merge-friendly multi-job files. JSON: `schema_version` = 3, `run_id`, `metadata`, then the same fields as v2. HDF5: `openpfc/profiling/` has `schema_version` = 3; payload lives under `openpfc/profiling/runs/<sanitized_run_id>/` (inner group still carries v2-style `schema_version` = 2, datasets, and `ranks/`). |

Exports default to v2 when `ProfilingExportOptions::run_id` is empty. When `run_id` is set (e.g. Slurm job id via `App` or `OPENPFC_PROFILING_RUN_ID`), exports use v3.

## Gather layout (all versions)

Let `stride = |frame_metric_names| + 2 × |region_paths|` doubles per frame (same catalog on every rank).

After `MPI_Gatherv`, rank 0 holds a contiguous buffer `all_flat` of `total_rows × stride` doubles, where `total_rows = Σ_r n_frames[r]`.

Row order: rows `0 … n_frames[0]-1` belong to MPI rank `0`, then `n_frames[1]` rows for rank `1`, and so on. So:

```text
row_index = offset[r] + local_frame_index
offset[0] = 0
offset[r+1] = offset[r] + n_frames[r]
```

If `mpi_rank` is stored in frame scalars, it should match the rank implied by this partitioning (OpenPFC defaults include `mpi_rank`).

## JSON schema version 2

Root object fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always `2`. |
| `openpfc_version` | string | Build/version string. |
| `n_mpi_ranks` | int | `MPI_Comm_size`. |
| `total_frames` | int | Sum of all ranks’ frame counts. |
| `frame_metric_names` | array of string | Column names for each entry in `frames[*].scalars` (same order on every rank). |
| `region_paths` | array of string | Ordered catalog paths (`/` hierarchy). |
| `ranks` | array | One object per MPI rank `0 … n_mpi_ranks-1`. |

Each element of `ranks`:

| Field | Type | Description |
|-------|------|-------------|
| `mpi_rank` | int | MPI rank id (matches index in `ranks` for typical runs). |
| `n_frames` | int | Number of committed frames on this rank. |
| `frames` | array | Length `n_frames`. Each element: |

Per-frame object:

| Field | Type | Description |
|-------|------|-------------|
| `scalars` | array of number | Values aligned with `frame_metric_names`. |
| `regions` | object | Nested tree: path segments become nested objects; leaves have `inclusive` and `exclusive` (seconds). |

### Minimal JSON example (schema 2)

```json
{
  "schema_version": 2,
  "n_mpi_ranks": 2,
  "total_frames": 2,
  "frame_metric_names": ["step", "mpi_rank", "wall_step"],
  "region_paths": ["main", "main/foo"],
  "ranks": [
    {
      "mpi_rank": 0,
      "n_frames": 1,
      "frames": [
        {
          "scalars": [0, 0, 1.0],
          "regions": {
            "main": {
              "inclusive": 0.5,
              "exclusive": 0.2,
              "foo": { "inclusive": 0.3, "exclusive": 0.3 }
            }
          }
        }
      ]
    },
    {
      "mpi_rank": 1,
      "n_frames": 1,
      "frames": [
        {
          "scalars": [0, 1, 1.1],
          "regions": {
            "main": { "inclusive": 0.6, "exclusive": 0.6 }
          }
        }
      ]
    }
  ]
}
```

## HDF5 schema version 2

File root group: `openpfc/profiling/`.

| Path | Description |
|------|-------------|
| `profiling` | Root group; attribute `schema_version` = 2; attribute `openpfc_version` (string). |
| `profiling/frame_metric_names` | 1D variable-length strings, length `nmeta` (omitted if `nmeta == 0`). |
| `profiling/region_paths` | 1D variable-length strings, length `npaths` (optional but written when `npaths > 0`). |
| `profiling/ranks` | Group containing one subgroup per MPI rank. |
| `profiling/ranks/<r>/` | `<r>` is the decimal string of the MPI rank (e.g. `0`, `1`, `15`). Attribute `n_frames` (int). |
| `profiling/ranks/<r>/frame_scalars` | 2D dataset `[n_frames, nmeta]`, omitted if `nmeta == 0` or `n_frames == 0`. |
| `profiling/ranks/<r>/<path_segments>/inclusive` | 1D `double[n_frames]` for that region path. |
| `profiling/ranks/<r>/<path_segments>/exclusive` | 1D `double[n_frames]`. |

Path segments follow the `/`-split catalog path (same as JSON nesting). Example: path `main/foo` → groups `main` then `foo`, datasets `inclusive` / `exclusive` on the `foo` group.

Empty ranks: If `n_frames == 0`, the rank subgroup exists with `n_frames = 0` and no `frame_scalars` or region datasets.

## JSON schema version 3

Same root fields as version 2, plus:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always `3`. |
| `run_id` | string | Opaque id (e.g. `SLURM_JOB_ID`) for this export. |
| `metadata` | object | Optional string/number/bool (or nested JSON, implementation-dependent) merged from config and environment; see `performance_profiling.md`. |

All other fields (`openpfc_version`, `n_mpi_ranks`, `total_frames`, `frame_metric_names`, `region_paths`, `ranks`) match v2.

## HDF5 schema version 3

| Path | Description |
|------|-------------|
| `openpfc/profiling` | Attribute `schema_version` = 3; attribute `openpfc_version`. |
| `openpfc/profiling/runs` | Container group. |
| `openpfc/profiling/runs/<id>/` | `<id>` = sanitized `run_id` (only letters, digits, `-`, `_`; other characters replaced with `_`). User metadata from `export_metadata` are stored as string attributes on this group (keys sanitized; `schema_version` / `openpfc_version` in metadata are skipped to avoid clashing with payload attrs). |
| `openpfc/profiling/runs/<id>/…` | Same layout as HDF5 schema version 2 under this group (`schema_version` = 2, `openpfc_version`, `frame_metric_names`, `region_paths`, `ranks/…`). |

Merging jobs: copy each run subtree into one file, e.g. with `h5copy` from `openpfc/profiling/runs/JOB_A` to the same path in a destination file (unique `run_id` ⇒ no collisions).

## Migration from schema 1 to 2

- JSON: Replace a single `frames` list with `ranks[].frames`. If you only need a flat list, concatenate `ranks[r].frames` in order of increasing `ranks[r].mpi_rank`.
- HDF5: Replace global-length datasets with per-rank groups under `ranks/<id>/`.

## Console table (stdout, optional)

The `App` option `profiling.print_report` calls `print_profiling_timer(std::ostream &, MPI_Comm, …)` with `mpi_aggregate_stdout = true`. All MPI ranks participate in a gather (same packed layout as `finalize_and_export`); rank 0 prints a TimerOutputs-style table. For each region path, the time column shows per-rank totals (sum over that rank’s frames on that path) combined across ranks using `ProfilingPrintOptions::mpi_aggregate_stat` (default `mean`; `sum`, `min`, `max`, `median`). `ncalls` is the sum across ranks of per-rank frame-hit counts. `%tot` is relative to the sum of `wall_denominator_metric` (default `wall_step`) over all gathered frames. This is not written to the JSON/HDF5 file; it is a separate, second gather for the console.

## Cross-rank statistics (offline)

The export is raw per-frame, per-rank data. Aggregations (min, max, mean, median across ranks or across frames) are not stored in the file; compute them in analysis code, for example:

```python
# Pseudocode: mean inclusive time for path "gradient" at frame index f across ranks
import json
with open("profile.json") as f:
    data = json.load(f)
vals = []
for rank_entry in data["ranks"]:
    for fr in rank_entry["frames"]:
        v = fr["regions"]["gradient"]["inclusive"]  # adjust for nesting
        vals.append(v)
mean = sum(vals) / len(vals)
```

## OpenPFC vs generic use

- Generic: Construct `ProfilingSession` with your own `frame_metric_names` and region catalog; call `begin_frame` / `set_frame_metric` / `end_frame` (see `ProfilingSession` API).
- OpenPFC defaults: Helpers and default scalar names live in `include/openpfc/kernel/profiling/openpfc_frame_metrics.hpp`.

See also [performance_profiling.md](performance_profiling.md) for runtime configuration.
