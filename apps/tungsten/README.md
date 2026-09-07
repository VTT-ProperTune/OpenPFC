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

Optional directional-solidification keys on `model.params` (same names as aluminum): `G_grid`, `V_grid`, `x_initial`. Default is isothermal (`G_grid = 0`). The linear operator stays at JSON `T`; the pointwise cubic uses `T + G (x' - x_initial - V t)`. CPU / CUDA / HIP share `tungsten_pointwise.hpp`.

## Run (from build tree)

```bash
cd build
mpirun -n 4 ./apps/tungsten/tungsten ../apps/tungsten/inputs_json/tungsten_single_seed.json
```

Use `tungsten_cuda` / `tungsten_hip` when built; pass the same config path. On LUMI-G, see [`docs/INSTALL.LUMI.md`](../../docs/hpc/INSTALL.LUMI.md) and [`docs/lumi_slurm/README.md`](../../docs/lumi_slurm/README.md).

## Restart a moving front

Use a checkpoint bundle to continue both the field and its tracked front.
A raw `psi_*.bin` dump contains no boundary state.

For a first leg, copy `inputs_json/tungsten_moving_bc.json`, set
`timestepping.t1` to `100`, change `fields[0].data` to
`results/tungsten/psi_%04d.bin`, and add:

```json
"checkpoint": {"every": 100, "directory": "results/tungsten/checkpoints"}
```

Create `results/tungsten`, run that config, then use
`inputs_json/tungsten_restart.json` to continue
from step 100 to step 200. The TOML restart input describes the same case.
Create `results/tungsten/restarted` before running the second leg; its field
dumps and new checkpoints use this separate output directory.

Keep the original `t0`, `dt`, domain, physics parameters, and the ordered
`boundary_conditions` configuration. The moving-BC keys are `type`, `target`,
`rho_low`, `rho_high`, `width`, `alpha`, `disp`, and `xpos`. Here `xpos` is
the original configured position; restart replaces it with the saved front.
Do not combine `restart_from` with `simulator.increment` or
`simulator.result_counter`. Initial conditions may be omitted on restart.

The bundle's `metadata.json` stores each BC's configuration and state.
For `moving`, state consists of unwrapped `xpos`, scan index `idx`, and
the first-detection flag `first`. Restored fronts continue scanning without
repeating initial detection. Missing/invalid front state or a changed BC
configuration is a startup error. Older bundles remain loadable with
stateless boundaries. CPU regression tests compare uninterrupted and split
runs on one and two MPI ranks: local fields agree within `1e-12` and saved
front state agrees exactly.
Checkpoints publish after scheduled field output, so the restored result
counter identifies the next filename, including when a save and checkpoint
fall on the same step.

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
