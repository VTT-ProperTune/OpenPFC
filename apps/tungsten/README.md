<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tungsten PFC application

Production tungsten phase-field crystal binary: JSON/TOML → `pfc::ui::SpectralETDSession<TungstenPhysics, Stack>` (aliases `TungstenSession` (CPU), `TungstenCUDASession`, `TungstenHIPSession` in `include/tungsten/tungsten_session.hpp`). Model parameters are validated at startup (see root `README.md` — Configuration Validation).

## Binaries (after `OpenPFC_BUILD_APPS=ON`)

| Target | When |
|--------|------|
| `tungsten` | Always (CPU FFT / HeFFTe) — `TungstenSession` |
| `tungsten_etd` | Always — alias of `tungsten` |
| `tungsten_cuda` | CUDA spectral — `TungstenCUDASession` |
| `tungsten_etd_cuda` | CUDA spectral — alias of `tungsten_cuda` |
| `tungsten_hip` | HIP spectral — `TungstenHIPSession` |
| `tungsten_etd_hip` | HIP spectral — alias of `tungsten_hip` |
| `verify_gpu_aware_mpi` | HIP + MPI device-buffer check |

Install path when using `cmake --install`: `<prefix>/bin/`.

## Inputs

| Location | Format |
|----------|--------|
| [`inputs_json/`](inputs_json/README.md) | JSON (mirrors TOML structure) |
| `inputs_toml/` | TOML (same scenarios as JSON; no separate README) |

Start from `inputs_json/tungsten_single_seed.json` or `inputs_toml/tungsten_single_seed.toml`. Heavy performance cases: `tungsten_performance.*`.

## Run (from build tree)

```bash
cd build
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

Use `tungsten_cuda` / `tungsten_hip` when built; pass the same config path. On LUMI-G, see [`docs/INSTALL.LUMI.md`](../../docs/hpc/INSTALL.LUMI.md) and [`docs/lumi_slurm/README.md`](../../docs/lumi_slurm/README.md).

## Code map

| Area | Path |
|------|------|
| Physics (schema, k-space symbols, `pointwise()`) | `include/tungsten/tungsten_physics.hpp`, `tungsten_pointwise.hpp` |
| Session aliases + catalog registration | `include/tungsten/tungsten_session.hpp` (`pfc::ui::SpectralETDSession`) |
| ICs / BCs / writers | Framework catalogs: `constant`, `single_seed`, `seed_grid` ICs; `fixed` / `moving` BCs from `apps/common` (`tungsten::register_catalog()`); `fields[]` writers (`binary`, `vtk`, `hdf5`) |
| Device instantiation of the nonlinearity | `src/gpu/tungsten_pointwise.inc` (stamped into `.cu` / `.hip`) |
| `main()` | `src/{cpu,cuda,hip}/tungsten.cpp` via `pfc::ui::run_json_session_main` |

## See also

- [`docs/app_pipeline.md`](../../docs/user_guide/app_pipeline.md) — JSON → session pipeline
- [`docs/applications.md`](../../docs/user_guide/applications.md) — all shipped apps
- [`docs/io_results.md`](../../docs/user_guide/io_results.md) — binary result writers from `fields`
