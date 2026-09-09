<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tungsten PFC application

Production tungsten phase-field crystal binary: JSON/TOML → `pfc::ui::SpectralETDSession<TungstenPhysics, Stack>` (aliases `TungstenSession` (CPU), `TungstenCUDASession`, `TungstenHIPSession` in `include/tungsten/tungsten_session.hpp`). Model parameters are validated at startup (see root `README.md` — Configuration Validation).

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry. Two shipped configurations are described: `tungsten_moving_bc.json`,
the directional-solidification case, and `tungsten_single_seed.json`, the
smallest one. **No shipped input is executed by ctest** -- the tests build
equivalent configurations in C++ -- so none of these files is a verification
preset in the sense the contract means; they are demonstration and performance
configurations.

| Item | Directional solidification (`tungsten_moving_bc.json`) | Single seed (`tungsten_single_seed.json`) |
|---|---|---|
| **Use case** | A solidification front sweeping through undercooled tungsten, with a grid of differently oriented seeds ahead of it | Smallest runnable configuration |
| **Question** | Where does the front sit, how fast does it advance, and what grain structure does it leave? | Does the stack run end to end? |
| **Domain** | `128 x 64 x 64` cells at `dx = 2*pi/(4*sqrt(2)) = 1.11072`, i.e. `142.2 x 71.1 x 71.1` reduced PFC length units, origin at the corner. Eight cells per BCC lattice constant `a = 2*pi*sqrt(2)` | `32^3` cells at the same spacing (`35.5^3`), origin at the centre |
| **Grid/time** | `t` in `[0, 20000]`, `dt = 1`, field dump every 100 | `t` in `[0, 10]`, `dt = 1`, dump every step |
| **Boundary conditions** | Periodic on all three axes (the FFT has no other mode), plus a `moving` sigmoid density-reservoir band of width 15 tracking the front 40 units ahead of it -- see "Why periodic" below | Periodic on all three axes plus a stationary `fixed` reservoir band |
| **Initial condition** | Uniform `psi = -0.10`, then `seed_grid`: a `2 x 2` array of BCC seeds of radius 25 at `x = 30`, each jittered by `+/-0.2 R` and randomly oriented from a fixed RNG seed (42), so runs are deterministic | Uniform `psi = -0.4`, then `single_seed`: one BCC nucleus from the six `{110}` wave vectors, `amp_eq = 0.215936`, `rho_seed = -0.047` |
| **Key physical parameters** | `T = 3300`, `T0 = 156000` (the TOML comments label both **Kelvin**; only the ratio enters, via `exp(-T/T0)`); `n_sol = -0.047`, `n_vap = -0.464`; `Bx = 0.8582`, `alpha = 0.50`; polynomials `p2..p4`, `q20..q40`. Lengths and times are **reduced PFC units** -- this repository contains no conversion to nm or s | Identical parameter block |
| **Observable** | Tracked front position `xpos` (checkpointed and restartable); `SPECTRAL_CHECKSUM` `sum`/`sumsq`/`l2` of `psi`; binary field dumps | Same checksums; `psi` and `psiMF` dumps |
| **Model maturity** | numerical verification: **regression** -- split-run equivalence to `1e-12`, one pinned CPU checksum literal, CPU/GPU parity; the analytical unit tests cover the ETD coefficients, not the model. Physical completeness: **extended** -- a fitted multi-term equation of state with a vapour branch and a mean-field-filtered density, beyond the canonical single-peak PFC. Calibration: **representative**, with a caveat: no source is cited anywhere in this repository for `Bx`, `T0`, `n_sol`, `n_vap` or the polynomial coefficients, so they cannot be traced from it | Same |

### Why periodic, and what `fixed`/`moving` really are

The FFT treats the domain as periodic on all three axes, always. `fixed` and
`moving` are **not** boundary conditions in the usual sense: they are field
modifiers applied every step that overlay a sigmoid density band near one end
of the box, pulling `psi` towards `rho_low` outside it and `rho_high` inside.
Physically that band is a reservoir keeping melt available for the front to
grow into and stopping the growing solid from meeting its own periodic image
head-on. It is a penalty layer inside a periodic box, not a wall. What it
excludes is a real free surface: no vapour interface, no substrate, no
container.

### Caveat: the single-seed preset does not show a seed

`SingleSeed` hard-codes its radius at 64 reduced units, larger than the
17.8-unit half-width of the `32^3` box, so the "seed" fills the whole domain
and no melt is left around it. Use the `256^3`
[`tungsten_single_seed_256_cuda.json`](tungsten_single_seed_256_cuda.json) when
you want an actual nucleus in melt.

### Caveat: `tungsten_moving_bc_options.json` does not load

That file's `boundary_conditions` entry has `initial_position` but no `xpos`.
`MovingBC::from_json` requires `xpos` and no code reads `initial_position`, so
the file throws at startup. It documents a feature that does not exist.

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
