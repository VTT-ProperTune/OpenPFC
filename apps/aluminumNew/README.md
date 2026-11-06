<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# AluminumNew

Production aluminum binary: JSON/TOML → `pfc::ui::SpectralETDSession<AluminumPhysics, Stack>` (aliases `AluminumSession`, `AluminumCUDASession`, `AluminumHIPSession` in `include/aluminum/aluminum_session.hpp`); the moving-frame mean-field ETD runs on the shared `pfc::sim::SpectralETDSystem`. Initial conditions include `constant` and the app-registered `seed_grid_fcc`.

## Problem setup

The `Problem setup` block the report contract (`#112`) asks every application
to carry, for the shipped demonstration case
[`aluminumNew.json`](aluminumNew.json). No shipped input is a verification
preset: the pinned checksum in the test suite runs a **separate synthetic
`32^3` configuration built in C++** (`X0 = 8`, radius 4, `rseed = 42`), not any
of them. `inputs_json/smoke.json` is a `16^3` sanity file that no test or
CMake rule references, and its initial condition is a *constant* field --
there is no seed in it and it never develops structure.
[`inputs_json/fcc_seed_nucleus.json`](inputs_json/fcc_seed_nucleus.json) is a
`192^3` single-seed case that runs on one core in a few minutes; see "The
isothermal case is a nucleation threshold, not growth" below.
`inputs_json/aluminum_directional.json` is a distinct small pulled
reservoir smoke case, with `G_grid=2`, `V_grid=0.1`, and an independent
temperature station at `x_initial=60`. The field is not a calibrated casting
model. Its front becomes ambiguous during the short run: the test intentionally
reports `not_single_front`, so this is NOT a demonstrated sustained physical
growth-front speed. The moving BC checkpoint/restart parity and zero-pull
and isothermal controls are the bounded results at this stage. The
periodic-unwrapping seam used by the prescribed temperature field remains
fixed at the case's initial `x_initial` station, while the temperature
profile's time dependence uses `V_grid` and the density reservoir tracks its
own `MovingBC` position. There is no feedback from the measured crystal
interface or moving density reservoir into that temperature seam. A reservoir
position must not be interpreted as a physical solidification front.

`tests/test_aluminum_inputs.cpp` (ctest `aluminum-shipped-inputs`) checks that
every file under `inputs_json/` still loads through the schema and the
initial-condition catalog. That is a rot check, not a physics check.

| Item | Description |
|---|---|
| **Use case** | An aluminium melt with a row of FCC seeds, intended for directional solidification |
| **Question** | Where does the front settle under a pulled temperature gradient, and what grain structure follows it? |
| **Domain** | `1024 x 2048 x 256` cells at `dx = 2*pi*sqrt(3)/8 = 1.36035`, i.e. `1393 x 2786 x 348` reduced PFC length units, box starting at 0. Eight cells per FCC lattice constant `a = 2*pi*sqrt(3)` |
| **Grid/time** | `t` in `[0, 2000]`, `dt = 1` (2000 steps), field dump every 200 |
| **Boundary conditions** | Periodic on all three axes (the FFT has no other mode), plus a stationary `fixed` sigmoid density-reservoir band pulling `psi` between `-1.297` and `-0.006` near the high-`x` end -- see "Why periodic" below |
| **Initial condition** | Uniform `psi = -0.006`, then `seed_grid_fcc`: a `2 x 1` row of FCC seeds of radius 120 at `x = 130`, amplitude 0.4, `rho = -0.036`, each jittered by `+/-0.2 R` and randomly oriented. Each seed is built from four `<111>` wave vectors, unlike tungsten's six `{110}` |
| **Key physical parameters** | `T_const = 980` clamped into `[T_min, T_max] = [780, 1280]`, `T0 = 89285` (no unit is stated anywhere in this app); `Bx = 0.81790`, `alpha = 0.20`; barred polynomial coefficients. The coexistence densities `-0.036` / `-1.297` live in the `initial_conditions` / `boundary_conditions` blocks, not in `model.params`. **The original demonstration is isothermal** (`G_grid=V_grid=0`); the separate directional smoke case uses nonzero `G_grid`/`V_grid` with a prescribed temperature profile. Reduced PFC units throughout, with no SI mapping in this repository |
| **Observable** | `SPECTRAL_CHECKSUM` `sum`/`sumsq`/`l2` of `psi`; the summed free energy (`last_free_energy_sum`), which this app computes per cell and `tungsten` does not; binary field dumps |
| **Model maturity** | numerical verification: **regression** -- the operators, the pointwise nonlinearity and the free-energy density are checked to `1e-14` against an independently written reference formula, and one synthetic case is pinned; there is no closed-form solution of the model. Physical completeness: **extended** -- the tungsten equation of state plus a separate correlation kernel `P(k)` and a travelling temperature field. Calibration: **representative**, with the same caveat as tungsten: no source is cited anywhere in this repository for any aluminium coefficient |

### The isothermal case is nucleation, not steady growth

The original isothermal seed experiments set `G_grid = V_grid = 0`,
so their thermal drive is off. It is tempting to describe what is left as seeded
growth. That is not what happens. Measured on
[`inputs_json/fcc_seed_nucleus.json`](inputs_json/fcc_seed_nucleus.json)
(`192^3`, one randomly oriented FCC seed, `n0 = -0.006`, `T_const = 980`),
run as shipped and again with only the seed radius changed, each continued to
`t = 1000`:

| seed radius | effective radius over time | envelope amplitude | outcome |
|---|---|---|---|
| 60 | 59.6 -> 52.0 (minimum, `t ~ 500`) -> 54.6 at `t = 1000`, still accelerating | 0.86 -> 1.06, rising throughout | **survives and grows** |
| 50 | 49.6 -> 41.4 (`t = 200`) -> 26.1 (`t = 600`) | 0.86 -> 0.24, collapsing | dissolves by `t ~ 700` |
| 40 | 39.6 -> 29.9 (`t = 200`) | 0.85 -> 0.31 | dissolves |
| 30 | 29.6 -> gone | 0.84 -> 0.03 | dissolved by `t = 200` |

Every seed shrinks at first, and that is the mechanism rather than an
artefact: `SeedFCC` writes a diffuse profile (amplitude 0.4, peak `psi` 3.16)
that must relax onto the model's own much larger solid amplitude (peak `psi`
near 5), and the radius a seed spends paying for that relaxation decides
whether anything is left to grow. So the melt is only just undercooled, growth
is slow and starts late, and the critical radius at these parameters lies
between 50 and 60 reduced units -- about five FCC lattice constants
(`a = 2*pi*sqrt(3) = 10.88`).

Radii here are the volume where `|psi - n0|`, box-blurred over one lattice
constant, exceeds half its own 99.9th percentile, converted to a
sphere-equivalent radius; the envelope amplitude is that percentile. Rendered
in the applications report as
`out/report/figures/aluminum_fcc_nucleus_comparison.svg` (at `t = 200`, where
the shipped preset stops); the run recipe is in
`out/report/figures/run_field_demos.sh`.

Do not shrink the box below `192^3` without re-checking the melt. At `128^3`
the same radius-60 nucleus leaves only about 70 reduced units between its own
periodic images, and the melt around it fills with a visible interference
pattern from that self-interaction. At `192^3` it does not: beyond 100 reduced
units from the seed centre `|psi - n0|` peaks at 0.106 against the crystal's
4.4.

### Why periodic

Periodicity plus the reservoir band gives a melt column the solid can grow
into, with no free surface, no crucible and no container wall. The domain is
deliberately long and thin, which is the geometry of a
directional-solidification cell rather than a bulk sample. Nothing in the model
represents convection in the melt, solute partitioning, or a second chemical
component.

### `model.params` is enforced, and every field in it is read

Two things were wrong here before 0.2, and they were the same thing seen from
two sides.

**The loader skipped the schema.** `AluminumPhysics::from_json` was guarded by
`if (!params_json.is_null() && !params_json.empty())`, so `"params": {}` --
which is what `inputs_json/smoke.json` shipped -- took the C++ member
initialisers while the schema declared all twenty-five fields `required`. For
a calibrated material model that is the wrong default in both directions: the
parameters of a run were not recoverable from its input file, and the struct
defaults are an uncited second copy of coefficients this repository cites no
source for. The parse is now unconditional; an empty or absent `model.params`
reports every missing field at once, and `smoke.json` spells them out.

**Nine of the twenty-five fields were read nowhere** -- not the four the
`T_min`/`T_max` caveat used to name. They were resolved two ways:

| Field | Resolution |
|---|---|
| `T_min`, `T_max` | **Implemented.** They now clamp `T_const + T_var` (see below). |
| `alpha_farTol`, `alpha_highOrd` | **Removed.** They parametrise *tungsten's* `C2`; the FCC dual-Gaussian peak here has no far-field tolerance and no higher-order exponent. |
| `shift_u`, `shift_s` | **Removed.** They are the vapour shift that *produced* the `p*_bar` / `q*_bar` coefficients; the `_bar` names say it is already applied. |
| `n0`, `n_sol`, `n_vap` | **Removed** from `model.params`. They act in the `initial_conditions` and `boundary_conditions` blocks, which carry their own copies (`constant.n0`, `seed_grid_fcc.rho`, `fixed.rho_low`/`rho_high`). A second copy that nothing reads can disagree with the one that acts. |

Unknown keys are ignored, so an input that still carries the removed seven
loads unchanged -- but it no longer implies that they do anything.

### The temperature profile, and its clamp

`temperature_variation(x, t)` returns `G_grid * (x_unwrapped - x_initial -
V_grid * t)` -- a departure, with no `T0` offset. `x_unwrapped` is `x`
unwrapped relative to the tracked front position, so the profile follows the
front around the periodic box. The pointwise nonlinearity then uses
`T_const + T_var`.

That is a straight line in `x` with no bound of its own, so on a domain 1393
reduced units long any non-trivial `G_grid` drives it arbitrarily far from
`T_const`. `clamp_temperature` now holds `T_const + T_var` inside
`[T_min, T_max]`, and `T_min < T_max` with `T_const` inside the window is
checked at load time (`validate_temperature_window`) rather than silently
freezing the domain at one end. **The original isothermal presets have zero gradient and unchanged
checksums.** The separate directional reference uses the same clamp with
a nonzero imposed gradient.

## Provenance and reduced units

`coefficient_provenance()` in
[`include/aluminum/coefficient_provenance.hpp`](include/aluminum/coefficient_provenance.hpp)
classifies every shipped model coefficient, the derived coefficients
`AluminumParams::recompute_derived` produces, and the densities the initial
and boundary conditions actually write. Representative numbers are uncited:
this repository does not name a paper, a fit, or a measurement for them.
`reduced_unit_convention()` is the reduced-unit convention. Length,
temperature, time, and amplitude have no SI conversion here. The
classification is not a material validation.

`x_initial` is a case station, not intrinsically the same as the seed's
`X0`. The maintained isothermal seeded cases choose equal values. The
shipped `aluminum_directional.json` instead places the prescribed thermal
station at `x_initial=60` and the FCC seed at `X0=18`, deliberately separate.
`fcc_seed_nucleus.json` still lists `n0`, `n_sol`, `n_vap`, `alpha_farTol`,
`alpha_highOrd`, `shift_u`, and `shift_s` under `model.params`. The schema
does not read those keys. The C++ defaults are the same table, not a second
set of coefficients.


### Case and boundary parameter provenance

The table above owns the **19 model coefficients** and the separately listed
derived and density constants. Other inputs in the delivered cases are
**case controls**, not calibrated aluminum properties. The source of their
numeric values is the relevant maintained JSON/TOML file; there is no
cited measurement or fitted uncertainty for them. They must not be copied
into the model-coefficient table as if each preset shared a fixed value.

| Input key | Implemented meaning and source | Reduced units / admissibility |
|---|---|---|
| `model.params.G_grid`, `V_grid`, `x_initial` | Prescribed thermal slope, profile speed and zero-departure station. Each case supplies its own values; these are not material constants. | temperature/length, length/time, length; no SI calibration |
| `domain.size`, `lower`, `upper` | Sampling resolution and physical box; chosen by the case. Grid spacing is computed as `(upper-lower)/size`. | count, reduced length, reduced length |
| `timestepping.t0`, `t1`, `dt`, `saveat` | Numerical run horizon, integrator step and output cadence. | reduced time; `dt > 0` |
| `constant.n0` | Melt-background density, from the initial-condition record (current shipped value `-0.006`). | reduced amplitude, uncited representative |
| `seed_grid_fcc.X0`, `Ny`, `Nz` | Absolute seed station and seed-grid counts; chosen in each case. | reduced length, counts ≥1 |
| `seed_grid_fcc.radius`, `amplitude`, `rho`, `rseed` | Seed-envelope radius, crystalline-wave amplitude (`SeedFCC` adds `2 amplitude cos(k·x)` per wave), solid reference density and random seed. The large demo uses radius 120, the nucleus study 60, and both use amplitude 0.4. | reduced length, reduced amplitude, reduced amplitude, dimensionless pseudorandom seed; `radius>0` |
| `fixed.rho_low`, `rho_high` | Density-reservoir values in the fixed boundary; from the case record. | reduced amplitude, uncited representative |
| `FixedBC::xwidth`, `alpha` | The *currently hardcoded* fixed-reservoir band half-width `20` and sigmoid sharpness `1`; these are not JSON parameters in this implementation. | reduced length, inverse reduced length |
| `moving.rho_low`, `rho_high`, `width`, `alpha`, `disp`, `xpos` | Directional preset's separately configured density reservoir, half-width, sigmoid sharpness, tracking offset and checkpointed position; not a measured physical crystal-front position. | reduced amplitude; length; inverse length; length; length |
| `fields[].name`, `fields[].data`, `checkpoint.*` | Output/restart controls; not physical inputs. | filenames, steps and paths |

No scientifically established range or uncertainty is provided for the
representative free-energy coefficients, seed amplitudes/radii, density
reservoirs, or transport controls. **Unknown is not zero uncertainty.** The
preset values describe particular numerical experiments, not a defensible
reduced-to-SI mapping. In particular, an increased `G_grid` or `V_grid`
does not validate a physical undercooling, heat transfer or pulling speed.

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
