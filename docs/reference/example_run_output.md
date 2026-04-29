<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Example run output (what “success” looks like)

This page shows representative stdout/stderr for common entry points so you can compare your terminal to a known-good shape. Exact numbers and formatting depend on OpenPFC version, MPI, and log settings; use this as a shape check, not a byte-for-byte diff.

## `examples/05_simulator`

Source: `examples/05_simulator.cpp`. Rank 0 prints:

1. `Create initial condition` — once, when the Gaussian IC is applied.
2. `n = …, t = …, min = …, max = …` — one line after initialization and one per timestep until the run finishes (the `Time` range is fixed in code: `t1` and 42 steps).
3. `Test pass!` — if the final `psi` maximum is within tolerance of 0.5; otherwise `Test failed!` on stderr.

Example shape (middle lines omitted):

```text
Create initial condition
n = 0, t = 0.000000000000, min = ..., max = ...
n = 1, t = ..., min = ..., max = ...
...
n = 42, t = ..., min = ..., max = ...
Test pass!
```

Run from the build tree, e.g. `mpirun -n 4 ./examples/05_simulator`. Other ranks stay quiet for the `std::cout` paths above.

## `apps/tungsten/tungsten` (CPU)

The `App` path logs with the `[app]` prefix on rank 0 (see `include/openpfc/frontend/ui/app.hpp`). You should see, among others:

- `[app] Reading JSON configuration from …` or the TOML equivalent  
- `[app] Effective configuration (JSON):` followed by an indented dump of the parsed settings  
- `[app] World: …` summarizing the grid  
- `[app] Initializing model...`  
- `[app] Starting time integration (Simulator integrator API)`  

If `model.params` validation is enabled in your binary, you may also see a Configuration Validation Summary block (root `README.md`, [`parameter_validation.md`](../user_guide/parameter_validation.md)) before the time loop.

Exit code 0 from `mpirun` and completion without an exception mean the driver finished; output files appear only when `saveat` and `fields` / writers are configured (see [`io_results.md`](../user_guide/io_results.md)).

## GPU binaries (`tungsten_cuda`, `tungsten_hip`, …)

Log shape matches the CPU `App` flow. Additional lines may appear when GPU-aware MPI or device backends are compiled in (see [`tutorials/gpu_app_quickstart.md`](../tutorials/gpu_app_quickstart.md) and [`INSTALL.LUMI.md`](../hpc/INSTALL.LUMI.md)). Your JSON/TOML should set `plan_options` / `backend` consistently with the build ([`examples/fft_backend_selection.toml`](../../examples/fft_backend_selection.toml)).

## See also

- [`quickstart.md`](../quickstart.md) — §2A / §2B success hints  
- [`faq.md`](../faq.md) — “How do I know an example or tungsten run succeeded?”  
- [`troubleshooting.md`](../troubleshooting.md) — when nothing matches the above  
