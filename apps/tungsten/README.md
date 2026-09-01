<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tungsten PFC application

Production tungsten phase-field crystal binary: JSON/TOML → `TungstenETDSession` (CPU) or `TungstenETDGPUSession` (CUDA/HIP). Model parameters are validated at startup (see root `README.md` — Configuration Validation).

## Binaries (after `OpenPFC_BUILD_APPS=ON`)

| Target | When |
|--------|------|
| `tungsten` | Always (CPU FFT / HeFFTe) — 0.2 `TungstenETDSession` |
| `tungsten_etd` | Always — alias of `tungsten` |
| `tungsten_cuda` | CUDA spectral — 0.2 `TungstenETDCUDASession` |
| `tungsten_etd_cuda` | CUDA spectral — alias of `tungsten_cuda` |
| `tungsten_hip` | HIP spectral — 0.2 `TungstenETDHIPSession` |
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
| Physics + ETD sessions | `include/tungsten/tungsten_physics.hpp`, `tungsten_etd_session.hpp`, `tungsten_etd_gpu_session.hpp` |
| ICs / BCs / writers | `tungsten_field_modifiers.hpp`, `tungsten_etd_io.hpp` |
| Shared `main()` | `include/tungsten/common/tungsten_app_main.hpp` |

## See also

- [`docs/app_pipeline.md`](../../docs/user_guide/app_pipeline.md) — JSON → session pipeline
- [`docs/applications.md`](../../docs/user_guide/applications.md) — all shipped apps
- [`docs/io_results.md`](../../docs/user_guide/io_results.md) — binary result writers from `fields`
