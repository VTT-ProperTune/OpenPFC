<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Allen–Cahn demo (`apps/allen_cahn`)

Minimal 2D Allen–Cahn example on a structured grid: explicit time stepping, finite differences with separated halos, optional PNG snapshots. This app does not use the JSON/TOML `App` path; it parses simple command-line arguments and calls MPI and decomposition APIs directly.

## Binaries

| Target | When built |
|--------|------------|
| `allen_cahn` | CPU (always when apps are enabled) |
| `allen_cahn_cuda` | `OpenPFC_ENABLE_CUDA` |
| `allen_cahn_hip` | `OpenPFC_ENABLE_HIP` |

## Build

With `OpenPFC_BUILD_APPS=ON`:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Executable: `build/apps/allen_cahn/allen_cahn`.

## Usage

```text
mpirun -n <P> ./allen_cahn <nx> <ny> <n_steps> [dt] [M] [epsilon] [driving_force] [png_initial] [png_final]
```

Defaults if omitted: `nx=ny=64`, `n_steps=5000`, `dt=0.00009`, `M=8.0`, `epsilon=0.19`, `driving_force=10.0`.
The app counts the global visible seed area as cells with `phi > 0` at the beginning and end of the run, prints `N1/N0`, and exits with failure unless the final area is at least 5× the initial area. The optional positive `driving_force` favors the `phi≈+1` seed over the `phi≈-1` matrix. If you pass one PNG path, it writes the final field; if two, the first is the initial snapshot and the second the final (grayscale, rank 0).
The reported step timing measures the time-stepping loop only, after an MPI barrier and before PNG output or verification; `avg_step_time_s` is based on the slowest rank.

Example:

```bash
cd build
mpirun -n 4 ./apps/allen_cahn/allen_cahn 128 128 5000 0.00009 8.0 0.19 10.0 initial.png final.png
```

## See also

- [`../../docs/halo_exchange.md`](../../docs/halo_exchange.md) — halo policies (this demo uses separated layout for FD) 
- [`../../examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) — related FD + halo pattern in `examples/` 
