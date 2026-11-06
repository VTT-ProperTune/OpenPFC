<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tungsten PFC application

Production tungsten phase-field crystal binary: JSON/TOML → `pfc::ui::SpectralETDSession<TungstenPhysics, Stack>` (aliases `TungstenSession` (CPU), `TungstenCUDASession`, `TungstenHIPSession` in `include/tungsten/tungsten_session.hpp`). Model parameters are validated at startup (see [parameter validation](../../docs/user_guide/parameter_validation.md)).

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry. Two shipped configurations are described below:
`tungsten_moving_bc.json`, the directional-solidification case, and
`tungsten_single_seed.json`, the smallest demonstration. Those files are
not executed by ctest. `tungsten_resolved_bcc.json` is: it is the numerical
resolution preset. It does not calibrate tungsten.

| Item | Directional solidification (`tungsten_moving_bc.json`) | Single seed (`tungsten_single_seed.json`) |
|---|---|---|
| **Use case** | A solidification front sweeping through undercooled tungsten, with a grid of differently oriented seeds ahead of it | Smallest runnable configuration |
| **Question** | Where does the front sit, how fast does it advance, and what grain structure does it leave? | Does the stack run end to end? |
| **Domain** | `128 x 64 x 64` cells at `dx = 2*pi/(4*sqrt(2)) = 1.11072`, i.e. `142.2 x 71.1 x 71.1` reduced PFC length units, box starting at 0. Eight cells per BCC lattice constant `a = 2*pi*sqrt(2)` | `32^3` cells at the same spacing (`35.5^3`), box centred on 0 |
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
| `tungsten_dealias_study` | Always — resolution study, see below. Minutes, not a test |

Install path when using `cmake --install`: `<prefix>/bin/`.

## Single-rank FFT

Spectral binaries accept root-level `"fft": "local"` in JSON, or
`fft = "local"` before any table in TOML. This selects a direct FFTW (CPU),
cuFFT (CUDA) or rocFFT (HIP) 3-D real/complex plan. The communicator must have exactly one rank, with the
whole domain on that rank. A distributed run rejects this choice at startup.

Omit `fft`, or set `"fft": "heffte"`, to retain the distributed HeFFTe path.
The hardware `backend` choice is independent of `fft`; existing
`plan_options` affect the HeFFTe path only.
Both paths retain the same spectral layout, inverse normalization and
preserved inverse input, but roundoff differences are possible. Startup
prints `SPECTRAL_FFT implementation=local backend=cufft ranks=1`, with
`backend=fftw` or `backend=rocfft` on those platforms, so a run records which
implementation it used. The host implementation
uses aligned scratch to preserve both input vectors. Its planning/destruction
must be serialized with other FFTW planners; it leaves global FFTW thread
settings and cleanup under application control.

## Inputs

| Location | Format |
|----------|--------|
| [`inputs_json/`](inputs_json/README.md) | JSON (mirrors TOML structure) |
| `inputs_toml/` | TOML (same scenarios as JSON; no separate README) |

Start from `inputs_json/tungsten_single_seed.json` or `inputs_toml/tungsten_single_seed.toml`. A $3\times 3$ polycrystal (grains that actually meet) is `tungsten_polycrystal.json`. Heavy performance cases: `tungsten_performance.*`.

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
| ICs / BCs / writers | Framework catalogs: `constant`, `single_seed`, `seed_grid` ICs; `fixed` / `moving` reservoir BCs local to this app (`tungsten::register_catalog()`); `fields[]` writers (`binary`, `vtk`, `hdf5`) |
| Device instantiation of the nonlinearity | `src/gpu/tungsten_pointwise.inc` (stamped into `.cu` / `.hip`) |
| Grid criteria for dealiasing | `include/tungsten/resolution.hpp`, pinned by `test_tungsten_resolution.cpp` (`[resolution]`, and `[preset]` for the qualified file) |
| Resolution study | `src/cpu/tungsten_dealias_study.cpp` -> `out/report/data/tungsten_dealias_resolution.csv` |
| `main()` | `src/{cpu,cuda,hip}/tungsten.cpp` via `pfc::ui::run_json_session_main` |

## Dealiasing and resolution

This section is a numerical resolution qualification. It does not calibrate
the model and it does not convert the representative coefficients to SI.

`inputs_json/tungsten_resolved_bcc.json` is the selected preset. The box is
one BCC cell, side $a=2\pi\sqrt{2}$, with 16 points along that side. That is
11.31 points per period of $|k|=1$, inside the cubic 1/2 rule (8 points per
period, `dealias_clean_dx`). `model.dealias` is true, so the installed
`TungstenSession` path applies the Orszag 2/3 mask to the nonlinear update.
At this spacing the cut sits above the second harmonic, so the mask does not
delete the crystal.

The gate is that file run through `TungstenSession` for $t\in[0,20]$, $dt=1$,
against the same model, the same box and the same mask on a $32^3$ grid.
The density RMS stays within $10^{-6}$ relative and the density maximum
within $10^{-4}$ relative. The test additionally compares the **full
coarse-grid field** with the 32³ field at the same physical nodes, and
the six complex normalized first-shell BCC coefficients; these comparisons
detect phase or structural disagreements that scalar extrema can hide. Turning the mask off on the $16^3$ preset changes
the maximum by less than $10^{-4}$, and the power in $|k|>2$ by less than
$10^{-4}$. The same mask comparison at the historical spacing ($8^3$ on this
box, $\Delta x=1.11072$, 5.66 points per period) moves the maximum by about
one percent and removes most of that high-mode power. That spacing fails
both grid criteria, so it is not selected.

The cell is bulk crystal. It has no solidification front. `TungstenPhysics`
does not expose a free-energy density, so the gate uses the density maximum,
the density RMS and the $|k|>2$ power. Cost is secondary. ctest also runs
the mask-off twin and the $8^3$ pair, all at 20 steps on one CPU rank.
`tungsten-resolved-bcc-cli` separately executes the actual `tungsten`
binary with this maintained JSON and checks the finite-field checksum.
To capture the numeric acceptance record in a local or source-pinned CI
log, run `build/apps/tungsten/test_tungsten "[preset]"`; it prints
`TUNGSTEN_BCC` with field and Fourier-mode errors.

**Limitation:** `high_mode_power` is the *final density tail* for
`|k|>2`, not a direct estimate of alias energy folded into retained
low-wavenumber modes of the nonlinear RHS. The declared 2/3 output mask
is not a generic cubic 1/2-rule de-aliaser. This one-cell 20-step gate
must not be generalized to all solidification fronts or parameter sets
without an independent nonlinear-product alias benchmark.

The historical solidification presets stay at $\Delta x=1.1107$ and omit
`model.dealias`, so their mask stays off. They are demonstration
configurations. The crystal sits at $k_0=1$ and carries real content at
$2k_0$ and $3k_0$; with that spacing, $k_{\mathrm{Ny}}=2.828$, so the third
harmonic at 3.0 folds back onto $k=2.657$, and the 2/3 cut at 1.886 sits
below the second harmonic at 2.0. Switching the mask on there would delete
part of the crystal. Both conditions clear at $\Delta x\le\pi/3$, six points
per lattice period. Eight points is the cubic 1/2 rule.

An earlier `tungsten_dealias_study` (same seeded solidification, same IC,
1000 steps, mask off then on, not this CI gate) measured:

| points / period | Grid spacing $\Delta x$ | $\Delta$ power | Peak density change $\Delta\max\psi$ | Peak wavenumber change $\Delta k_1$ |
|---|---|---|---|---|
| 5.66 (historical) | 1.1107 | 1.70 % | 1.07 % | 9e-06 |
| 6.00 | 1.0472 | 1.27 % | 1.04 % | 1.4e-04 |
| 6.67 | 0.9425 | 0.48 % | 0.29 % | 4.7e-04 |
| 8.00 | 0.7854 | **0.009 %** | 0.018 % | 2.5e-06 |

On that historical spacing the long study saw about 2% in amplitude and
spectral power, and essentially none in the selected wavenumber $k_1$. The
$k_1$ column is scatter between two nearly identical runs, not a trend.
The qualified preset above is a different grid and a shorter horizon; its
tolerance is the ctest gate, not this table.

Regenerate the data (minutes, single rank):

```bash
./apps/tungsten/tungsten_dealias_study \
  ../out/report/data/tungsten_dealias_resolution.csv 1000
python3 out/report/figures/make_figures.py
```

See the resolution section of the tungsten chapter in the applications report
for the full discussion.

## See also

- [`docs/app_pipeline.md`](../../docs/user_guide/app_pipeline.md) — JSON → session pipeline
- [`docs/applications.md`](../../docs/user_guide/applications.md) — all shipped apps
- [`docs/io_results.md`](../../docs/user_guide/io_results.md) — binary result writers from `fields`
