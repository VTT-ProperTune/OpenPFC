<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Allen–Cahn demo (`apps/allen_cahn`)

Minimal **2D Allen–Cahn** example on a structured grid: **explicit** time stepping, **finite differences** with **separated halos**, optional **PNG** snapshots. This app does **not** use the JSON/TOML **`App`** path; it parses simple **command-line arguments** and calls MPI and decomposition APIs directly.

## Binaries

| Target | When built |
|--------|------------|
| `allen_cahn` | CPU (always when apps are enabled) |
| `allen_cahn_cuda` | `OpenPFC_ENABLE_CUDA` |
| `allen_cahn_hip` | `OpenPFC_ENABLE_HIP` |

## Build

With **`OpenPFC_BUILD_APPS=ON`**:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Executable: **`build/apps/allen_cahn/allen_cahn`**.

## Usage

```text
mpirun -n <P> ./allen_cahn <nx> <ny> <n_steps> [dt] [M] [epsilon] [png_initial] [png_final]
```

Defaults if omitted: `nx=ny=64`, `n_steps=3000`, `dt=0.0015`, `M=2.0`, `epsilon=0.35`.  
If you pass **one** PNG path, it writes the **final** field; if **two**, the first is the **initial** snapshot and the second the **final** (grayscale, rank 0).

Example:

```bash
cd build
mpirun -n 4 ./apps/allen_cahn/allen_cahn 128 128 500 0.0015 2.0 0.35 initial.png final.png
```

## See also

- **[`../../docs/halo_exchange.md`](../../docs/halo_exchange.md)** — halo policies (this demo uses separated layout for FD)  
- **[`../../examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp)** — related FD + halo pattern in `examples/`  
