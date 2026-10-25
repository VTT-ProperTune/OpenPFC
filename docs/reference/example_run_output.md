<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Example run output (what “success” looks like)

This page shows representative stdout/stderr for common entry points so you can compare your terminal to a known-good shape. Exact numbers and formatting depend on OpenPFC version, MPI, and log settings; use this as a shape check, not a byte-for-byte diff.

## `examples/05_simulator`

Source: [the driver-hooks model](../../examples/heat/driver-hooks/model.cpp).
With its starting input, rank zero prints:

```text
Final time 0.2
```

The detailed evidence is in `results/events_rank_*.csv` and
`results/field_rank_*.csv`. The ledger records the start/apply/step/save
hooks and the actual intervals .07,.07,.06. The field can be checked
against the product of its backward-Euler mode multipliers.
See [the Quarto lesson](../../examples/heat/driver-hooks/index.qmd) for
commands and the mathematical check. A printed completion message alone
does not establish numerical accuracy.

## `apps/tungsten/tungsten` (CPU)

The `App` path logs with the `[app]` prefix on rank 0 (see `include/openpfc/frontend/ui/app.hpp`). You should see, among others:

- `[app] Reading JSON configuration from …` or the TOML equivalent  
- `[app] Effective configuration (JSON):` followed by an indented dump of the parsed settings  
- `[app] Domain: …` summarizing the grid  
- `[app] Initializing model...`  
- `[app] Starting time integration (Simulator integrator API)`  

If `model.params` validation is enabled in your binary, you may also see a Configuration Validation Summary block ([`parameter_validation.md`](../user_guide/parameter_validation.md)) before the time loop.

Exit code 0 from `mpirun` and completion without an exception mean the driver finished; output files appear only when `saveat` and `fields` / writers are configured (see [`io_results.md`](../user_guide/io_results.md)).

## GPU binaries (`tungsten_cuda`, `tungsten_hip`, …)

Log shape matches the CPU `App` flow. Additional lines may appear when GPU-aware MPI or device backends are compiled in (see [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) and [`INSTALL.LUMI.md`](../hpc/INSTALL.LUMI.md)). Your JSON/TOML should set `plan_options` / `backend` consistently with the build ([`examples/fft_backend_selection.toml`](../../examples/fft_backend_selection.toml)).

## See also

- [`quickstart.md`](../quickstart.md) — §2A / §2B success hints  
- [`faq.md`](../faq.md) — “How do I know an example or tungsten run succeeded?”  
- [`troubleshooting.md`](../troubleshooting.md) — when nothing matches the above  
