<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tungsten PFC application

Production-style tungsten phase-field crystal binary built from `pfc::ui::App<TungstenModel>` (CPU, optional CUDA/HIP). Reads JSON or TOML; model parameters are validated at startup (see root `README.md` — Configuration Validation).

## Binaries (after `OpenPFC_BUILD_APPS=ON`)

| Target | When |
|--------|------|
| `tungsten` | Always (CPU FFT / HeFFTe) |
| `tungsten_cuda` | `OpenPFC_ENABLE_CUDA` |
| `tungsten_hip` | `OpenPFC_ENABLE_HIP` |
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

Use `tungsten_cuda` / `tungsten_hip` when built; pass the same config path. On LUMI-G, see [`docs/INSTALL.LUMI.md`](../../docs/INSTALL.LUMI.md) and [`docs/lumi_slurm/README.md`](../../docs/lumi_slurm/README.md).

## Code map

| Area | Path |
|------|------|
| CPU model | `include/tungsten/cpu/` |
| CUDA / HIP | `include/tungsten/cuda/`, `include/tungsten/hip/` |
| Shared params / validation | `include/tungsten/common/tungsten_input.hpp` |

## See also

- [`docs/app_pipeline.md`](../../docs/app_pipeline.md) — how JSON maps to `Simulator`
- [`docs/applications.md`](../../docs/applications.md) — all shipped apps
- [`docs/io_results.md`](../../docs/io_results.md) — binary result writers from `fields`
