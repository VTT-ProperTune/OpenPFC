<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications

The programs under `apps/` are full OpenPFC applications. They are different from the short executables under `examples/`: examples teach one API pattern at a time, while applications are meant to be run as model-specific binaries with realistic inputs. They are built when `OpenPFC_BUILD_APPS=ON`, which is the default, and they usually install under `<prefix>/bin` when you run `cmake --install`.

For realistic runs, assume MPI is involved. Use the same compiler, MPI and HeFFTe stack that you used to build OpenPFC; the install details are in [`INSTALL.md`](../../INSTALL.md). If you are still learning the library, run an example first through [`../quickstart.md`](../quickstart.md), then come back here.

## Which application should I run?

Start with tungsten if you want the production-style PFC path. It reads JSON or TOML, uses the `App` pipeline, writes configured fields, and has CPU, CUDA and HIP variants when the build enables them. Start with Allen–Cahn if you want a small visual sanity check with optional PNG output and fewer moving pieces. Use Heat3D when your question is about finite-difference orders, the spectral heat-equation path, timings or scaling comparisons. Use **wave2d** for a minimal **coupled first-order** wave-equation demo (displacement + velocity) with mixed periodic / physical y-boundaries. AluminumNew is mostly useful as a compact example of an `App<Model>` program wired through JSON.

If you want declarative configuration, read [`app_pipeline.md`](app_pipeline.md) before writing your own input files. If your immediate question is “what file did this run write?”, read [`io_results.md`](io_results.md).

## Tungsten PFC

Tungsten is the main 3D PFC application. The CPU binary is `tungsten`; GPU-enabled builds may also provide `tungsten_cuda` or `tungsten_hip`, and HIP builds may include `verify_gpu_aware_mpi` as a device-buffer smoke test for LUMI-style workflows. The application overview, code layout and input directories are documented in [`apps/tungsten/README.md`](../../apps/tungsten/README.md).

From your build directory, a first CPU run looks like this:

```bash
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

The JSON inputs live under [`apps/tungsten/inputs_json/`](../../apps/tungsten/inputs_json/README.md), with TOML equivalents under `inputs_toml/`. Other sample JSON files include `tungsten_fixed_bc.json`, `tungsten_moving_bc.json` and `tungsten_performance.json`. For GPU-aware MPI and Slurm examples, use [`../hpc/INSTALL.LUMI.md`](../hpc/INSTALL.LUMI.md) and [`../lumi_slurm/README.md`](../lumi_slurm/README.md).

## AluminumNew

`aluminumNew` is a sample 3D application using OpenPFC, nlohmann_json and HeFFTe. It is useful when you want to see an `App<Model>` target without the full tungsten complexity. Its README is intentionally small; the source and CMake target are the reference. See [`apps/aluminumNew/README.md`](../../apps/aluminumNew/README.md).

## Heat3D

`heat3d` solves the 3D heat equation either with finite differences or with a spectral FFT step. The finite-difference path supports even orders from 2 to 20, and the app can use OpenMP over interior \((i_y,i_z)\) lines when the build enables it (the Laplacian along \(i_x\) stays serial in `finite_difference.hpp`). On **Linux**, a **single MPI rank** resets CPU affinity after `MPI_Init` (via `pfc::runtime::reset_cpu_affinity_if_single_mpi_rank` in [`runtime/common/cpu_affinity.hpp`](../../include/openpfc/runtime/common/cpu_affinity.hpp)) so OpenMP is not stuck on one core under default `mpirun` pinning (set `OPENPFC_NO_RESET_AFFINITY` to opt out). For several ranks per node, tune `mpirun` binding and `OMP_NUM_THREADS` as in [`apps/heat3d/README.md`](../../apps/heat3d/README.md).

## Allen–Cahn

Allen–Cahn is a CLI-driven 2D demo. It does not use the JSON `App` frontend, which makes it a good quick visual check. The CPU binary is `allen_cahn`; CUDA or HIP builds may provide `allen_cahn_cuda` or `allen_cahn_hip`. See [`apps/allen_cahn/README.md`](../../apps/allen_cahn/README.md) for the current arguments and example `mpirun` commands.

MPI: Use `mpirun` from Open MPI, the same stack as at configure time — typically Open MPI 4.1.1 with GCC 11.2 on cluster setups documented in [`INSTALL.md`](../../INSTALL.md) (§1). A mismatched launcher (e.g. system MPICH) causes confusing runtime failures.

Arguments (CPU binary): `nx ny n_steps dt M epsilon [driving_force] [png_final]` or, for an initial and final snapshot, `[png_initial] [png_final]` (two paths). The optional `driving_force` is detected when the next argument is numeric; otherwise that argument is treated as a PNG path for backward compatibility. Optional PNG paths trigger a gather on rank 0 and grayscale export via `pfc::io` (see `include/openpfc/frontend/io/png_writer.hpp`).

For visible motion on the grid, use moderate ε and large M; shrinking ε alone makes interfaces sharp but slow. A positive `driving_force` favours the `φ≈+1` seed over the `φ≈-1` matrix. The app reports step-loop timing and can gather PNG output on rank zero through the frontend PNG writer.

## wave2d

`wave2d` integrates the 2D acoustic wave equation \(u_{tt} = c^2 \Delta u\) as \(\partial_t u = v\), \(\partial_t v = c^2 \Delta u\) with explicit Euler in time. **x** is periodic (MPI halos); **y** supports homogeneous **Dirichlet** or **Neumann** physical boundaries via ghost correction after the periodic exchange. CPU binaries: `wave2d_fd_manual` (fixed second-order stencil on `PaddedBrick`) and `wave2d_fd` (even orders 2–20). Optional **VTK** output (`--vtk` / `--vtk-every`) uses `pfc::VTKWriter` for ParaView time series on CPU and GPU binaries alike. CUDA/HIP builds may add `wave2d_cuda` / `wave2d_hip`. See [`apps/wave2d/README.md`](../../apps/wave2d/README.md) for CLI, CFL guidance, and tests.

## Building your own application

If none of these binaries matches your problem, the next step is not to copy an application wholesale. First read [`app_pipeline.md`](app_pipeline.md) so you understand how JSON and TOML become a `Simulator`, then work through [`../tutorials/custom_app_minimal.md`](../tutorials/custom_app_minimal.md). The extension overview is [`../extending_openpfc/README.md`](../extending_openpfc/README.md).
