<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# AluminumNew

Production aluminum binary: JSON/TOML → `pfc::ui::SpectralETDSession<AluminumPhysics, Stack>` (aliases `AluminumSession`, `AluminumCUDASession`, `AluminumHIPSession` in `include/aluminum/aluminum_session.hpp`); the moving-frame mean-field ETD runs on the shared `pfc::sim::SpectralETDSystem`. Initial conditions include `constant` and the app-registered `seed_grid_fcc`.

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry, for the shipped demonstration case
[`aluminumNew.json`](aluminumNew.json). Neither shipped input is a verification
preset: the pinned checksum in the test suite runs a **separate synthetic
`32^3` configuration built in C++** (`X0 = 8`, radius 4, `rseed = 42`), not
either file. `inputs_json/smoke.json` is a `16^3` sanity file that no test or
CMake rule references.

| Item | Description |
|---|---|
| **Use case** | An aluminium melt with a row of FCC seeds, intended for directional solidification |
| **Question** | Where does the front settle under a pulled temperature gradient, and what grain structure follows it? |
| **Domain** | `1024 x 2048 x 256` cells at `dx = 2*pi*sqrt(3)/8 = 1.36035`, i.e. `1393 x 2786 x 348` reduced PFC length units, origin at the corner. Eight cells per FCC lattice constant `a = 2*pi*sqrt(3)` |
| **Grid/time** | `t` in `[0, 2000]`, `dt = 1` (2000 steps), field dump every 200 |
| **Boundary conditions** | Periodic on all three axes (the FFT has no other mode), plus a stationary `fixed` sigmoid density-reservoir band pulling `psi` between `-1.297` and `-0.006` near the high-`x` end -- see "Why periodic" below |
| **Initial condition** | Uniform `psi = -0.006`, then `seed_grid_fcc`: a `2 x 1` row of FCC seeds of radius 120 at `x = 130`, amplitude 0.4, `rho = -0.036`, each jittered by `+/-0.2 R` and randomly oriented. Each seed is built from four `<111>` wave vectors, unlike tungsten's six `{110}` |
| **Key physical parameters** | `T_const = 980`, `T0 = 89285` (no unit is stated anywhere in this app); `n_sol = -0.036`, `n_vap = -1.297`; `Bx = 0.81790`, `alpha = 0.20`; barred polynomial coefficients. **`G_grid = V_grid = 0`, so the shipped run is isothermal** -- the moving-frame machinery is present and tested but no shipped preset exercises it. Reduced PFC units throughout, with no SI mapping in this repository |
| **Observable** | `SPECTRAL_CHECKSUM` `sum`/`sumsq`/`l2` of `psi`; the summed free energy (`last_free_energy_sum`), which this app computes per cell and `tungsten` does not; binary field dumps |
| **Model maturity** | numerical verification: **regression** -- the operators, the pointwise nonlinearity and the free-energy density are checked to `1e-14` against an independently written reference formula, and one synthetic case is pinned; there is no closed-form solution of the model. Physical completeness: **extended** -- the tungsten equation of state plus a separate correlation kernel `P(k)` and a travelling temperature field. Calibration: **representative**, with the same caveat as tungsten: no source is cited anywhere in this repository for any aluminium coefficient |

### Why periodic

Periodicity plus the reservoir band gives a melt column the solid can grow
into, with no free surface, no crucible and no container wall. The domain is
deliberately long and thin, which is the geometry of a
directional-solidification cell rather than a bulk sample. Nothing in the model
represents convection in the melt, solute partitioning, or a second chemical
component.

### Caveat: `T_min` and `T_max` do nothing

The schema declares both as **required** and every shipped input and test
supplies them, but no code reads them. There is no clamping of the temperature
field anywhere in this application. `alpha_farTol` and `alpha_highOrd` are
likewise required but unused here -- the FCC correlation peak does not
reference them (they do matter for `apps/tungsten`).

### Caveat: the temperature profile

`temperature_variation(x, t)` returns `G_grid * (x_unwrapped - x_initial -
V_grid * t)` -- a departure, with no `T0` offset and no clamp. The pointwise
nonlinearity then uses `T_const + T_var`. `x_unwrapped` is `x` unwrapped
relative to the tracked front position, so the profile follows the front around
the periodic box.

## Build

Built with the main OpenPFC tree when `OpenPFC_BUILD_APPS=ON` (default):

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

The executable is `build/apps/aluminumNew/aluminumNew` (path may vary with the generator).
`aluminum_etd` is an alias of `aluminumNew`. GPU binaries are `aluminum_etd_cuda` / `aluminum_etd_hip` when those backends are enabled.

## Run

Pass a configuration file as the first argument (working directory affects relative paths inside the file):

```bash
cd build
mpirun -n 4 ./apps/aluminumNew/aluminumNew ../apps/aluminumNew/aluminumNew.json
```

A matching `aluminumNew.toml` is provided for TOML workflows. Adjust `results`, `fields`, and paths under `model.params` for your machine.

## Source layout

| File | Role |
|------|------|
| `src/aluminum.cpp` | `main`: JSON → `AluminumSession` (`aluminumNew` and its alias `aluminum_etd`) |
| `src/aluminum_cuda.cpp` / `src/aluminum_hip.cpp` | `main` for the GPU sessions on `GPUSpectralStack` |
| `src/gpu/aluminum_pointwise.inc` (`.cu` / `.hip`) | Device instantiation of the pointwise nonlinearity |
| `include/aluminum/aluminum_physics.hpp` | Schema, k-space symbols (`linear_symbol`, `filter_mf`, `correlation_kernel`, `nonlinear_symbol`), `pointwise()` |
| `include/aluminum/aluminum_pointwise.hpp` | `OPENPFC_HD` functor: `N`, free-energy density, `temperature_variation(x, t)` |
| `include/aluminum/seed_grid_fcc.hpp` | `seed_grid_fcc` catalog `FieldModifier` (registered by `aluminum::register_catalog()`) |
| `include/aluminum/aluminum_session.hpp` | Session aliases + `register_catalog()` |
| `SeedFCC.hpp` | FCC seed helper used by `seed_grid_fcc` |

## See also

- [`../../docs/applications.md`](../../docs/user_guide/applications.md) — other shipped apps 
- [`../../docs/configuration.md`](../../docs/user_guide/configuration.md) — config file concepts 
- [`../../docs/quickstart.md`](../../docs/quickstart.md) — first-time setup 
