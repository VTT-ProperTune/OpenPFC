<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications (`apps/`)

These are full programs (not the small `examples/` tutorials). They are built when `OpenPFC_BUILD_APPS=ON` (default). Binaries usually install under `<prefix>/bin` when you `cmake --install`.

All of them expect MPI for realistic runs unless noted. Match the same compiler/MPI/HeFFTe stack you used to build OpenPFC (see [`INSTALL.md`](../INSTALL.md)).

## Tungsten PFC (`apps/tungsten/`)

Overview: [`apps/tungsten/README.md`](../apps/tungsten/README.md) (binaries, inputs, code layout).

| Target | When available |
|--------|----------------|
| `tungsten` | Always (CPU) |
| `tungsten_cuda` | `OpenPFC_ENABLE_CUDA` and CUDA toolkit |
| `tungsten_hip` | `OpenPFC_ENABLE_HIP` and ROCm |
| `verify_gpu_aware_mpi` | HIP + MPI device-buffer smoke test (LUMI-style workflows) |

Inputs: JSON under [`apps/tungsten/inputs_json/`](../apps/tungsten/inputs_json/README.md); TOML in `inputs_toml/`.

Run (from your build directory, CPU binary):

```bash
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

Other samples in the same folder: `tungsten_fixed_bc.json`, `tungsten_moving_bc.json`, `tungsten_performance.json`; TOML under `../apps/tungsten/inputs_toml/`.

GPU-aware MPI and Slurm examples: [`INSTALL.LUMI.md`](INSTALL.LUMI.md), [`lumi_slurm/README.md`](lumi_slurm/README.md).

## Aluminum (`apps/aluminumNew/`)

| Target | Notes |
|--------|--------|
| `aluminumNew` | Sample application using OpenPFC + nlohmann_json + HeFFTe |

See [`apps/aluminumNew/README.md`](../apps/aluminumNew/README.md) (minimal; source and CMake are the reference).

## Allen–Cahn (`apps/allen_cahn/`)

| Target | When available |
|--------|----------------|
| `allen_cahn` | CPU |
| `allen_cahn_cuda` | CUDA enabled |
| `allen_cahn_hip` | HIP enabled |

CLI-driven 2D Allen–Cahn demo (no JSON `App`). See [`apps/allen_cahn/README.md`](../apps/allen_cahn/README.md) for arguments and example `mpirun`.

MPI: Use `mpirun` from Open MPI, the same stack as at configure time — typically Open MPI 4.1.1 with GCC 11.2 on cluster setups documented in [`INSTALL.md`](../INSTALL.md) (§1). A mismatched launcher (e.g. system MPICH) causes confusing runtime failures.

Arguments (CPU binary): `nx ny n_steps dt M epsilon [png_final]` or, for an initial and final snapshot, `[png_initial] [png_final]` (two paths). Optional PNG paths trigger a gather on rank 0 and grayscale export via `pfc::io` (see `include/openpfc/frontend/io/png_writer.hpp`).

Dynamics: For visible motion on the grid, use moderate ε and large M (mean-curvature scaling: shrinking ε alone makes interfaces sharp but slow). Many `n_steps` help; reduce `dt` if the explicit step blows up. PNGs use a fixed [-1,1] scale so initial vs final are comparable. Defaults in `allen_cahn/common.hpp` target obvious nucleus growth.

## Choosing apps vs examples

- Use `examples/` to learn APIs and patterns in short programs.
- Use `apps/` when you want a deployable binary with model-specific parameters and inputs closer to production PFC runs.

## See also

- [`quickstart.md`](quickstart.md) — run an app after building
- [`app_pipeline.md`](app_pipeline.md) — JSON/TOML → `Simulator` for `App`-driven binaries
- [`io_results.md`](io_results.md) — binary vs VTK vs PNG output
- [`extending_openpfc/README.md`](extending_openpfc/README.md) — how to build your own app the same way
