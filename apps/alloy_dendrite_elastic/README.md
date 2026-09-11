<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# `alloy_dendrite_elastic` — thermo-solutal-elastic solidification

Quantitative dilute-alloy phase field with anti-trapping current, coupled to
solute, to temperature with latent heat, **and to quasi-static elasticity
through a composition- and temperature-dependent eigenstrain whose energy
feeds back into the phase-field driving force**: equations (1)–(7) of the
capstone model spec (issue #85), all of them.

Equations (1)–(4) are local and run on high-order finite differences with a
halo exchange. Equations (5)–(7) are elliptic, hence global, and are solved
spectrally on the *same* decomposition inside the same time step — see
[The elastic coupling](#the-elastic-coupling). Local FD physics and a global
FFT solve, coupled, in one application; that combination is the point.

| | |
|---|---|
| Fields | `phi` (−1 liquid, +1 solid), `U` (supersaturation), `theta` (undercooling), `u` (displacement, slaved) |
| Discretisation | high-order central FD via `pfc::gradient::FDGradient`, orders 2–14 |
| Parallelism | MPI on `pfc::Domain` / `Box3i`, halo width `order/2` via `pfc::comm::HaloExchange` |
| Time integration | explicit, four stages, three halo exchanges per step |
| Dimensions | 2-D (`nz = 1`) and 3-D from one templated stepper |
| Elastic solve | `openpfc_apps/microelasticity.hpp` — Khachaturyan Green operator, Eyre–Milton fixed point, HeFFTe on the FD stack's own decomposition |
| Backends | CPU only — see [No HIP twin, and why](#no-hip-twin-and-why) |

## Binaries

| Binary | Stage | What it does |
|---|---|---|
| `alloy_dendrite_planar` | 1 | Isothermal planar front measured against the thin-interface prediction: velocity, kinetic coefficient, solute boundary layer, **effective partition coefficient**, and the two conservation invariants. |
| `alloy_dendrite_growth` | 2 / 3 | Deterministic dendrite, 2-D or 3-D (`--nz`), with tip-velocity and tip-radius diagnostics written to CSV. |

Both take `--key=value` options; `--help` lists them. Unknown keys are an
error rather than a warning, because a typo that silently reverts a parameter
to its default is exactly how a verification app produces a confident wrong
answer.

```bash
# Stage 1, about 40 s on one core. Run it on `standard`, not on a login
# node: login nodes are for editing and inspecting output, and no number
# quoted in this file was measured on one.
alloy_dendrite_planar --nx=854 --dx=0.6 --velocity=0.1 --t-end=600 \
                      --summary=results/planar.csv --run-id=baseline

# The anti-trapping current is what makes k_eff velocity-independent.
# Run these three and read the k_eff column.
alloy_dendrite_planar --at-scale=1  --run-id=at_on
alloy_dendrite_planar --at-scale=0  --run-id=at_off
alloy_dendrite_planar --at-scale=-1 --run-id=at_flipped

# Stage 2, about 3 minutes
alloy_dendrite_growth --csv=results/tip.csv --summary=results/dendrite.csv

# Stage 3 smoke, 3-D
alloy_dendrite_growth --nx=96 --ny=96 --nz=96 --dx=1.0 --t-end=40

# Stage 4: the same dendrite with and without the elastic feedback. The
# eigenstrain, stiffness and coupling default to Al-4.5wt%Cu (material.hpp),
# so `--elastic=1` alone is the calibrated model, not a scaled-down one.
alloy_dendrite_growth --run-id=off --summary=s.csv
alloy_dendrite_growth --run-id=on  --summary=s.csv --elastic=1 --n-el-substep=20

# Field snapshots for figures: phi, U, theta, and with the coupling on also
# f_el, df_el/dphi and the two stress invariants, as raw bricks plus a
# manifest. Correct at any rank count.
alloy_dendrite_growth --elastic=1 --fields-dir=out/ --fields-every=10
```

CSV output is **appended, never truncated**, and every row carries the
`--run-id`. A resolution study is a sequence of runs whose output belongs in
one file, and clobbering a ten-minute run is a worse failure than a file with
two headers in it.

## What was measured

All numbers below are from single-rank runs on a LUMI login node, `k = 0.15`,
`D_l = 2`, `lambda = 1` (so `beta = 0.6069`, deliberately nonzero — see
below), `eps4 = 0`, fourth-order stencils, `512 W0` periodic box, `t_end = 600`.

### Resolution study

`dx/W0` from 1.6 to 0.4, everything else fixed. `V` against
`(Omega − 1)/(k beta)`, `beta` against `a1 (tau0/(lambda W0)) (1 − a2 lambda W0^2/(tau0 D_l))`,
`ell` against `D_l/V`, `k_eff` against the input `k`.

The four relations below are not independent restatements of one another.
`U_i = -beta V` is interface kinetics, `ell = D_l/V` is the outer diffusion
field, `U_s = U_i` (i.e. `k_eff = k`) is local equilibrium at the interface,
and the steady-state mass balance `U_inf = k U_s - 1` says the freshly formed
solid carries exactly the far-field composition. An error in any one term of
the model breaks at least one of them. At `dx = 0.6 W0` the mass balance
closes to `-9.6e-5` absolute against `U_far = -1.009`.

| `dx/W0` | `V` error | `beta` error | `ell` error | `k_eff` | `k_eff` error | solute drift | heat drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.6 | −58.5 % | +963 % | −38.7 % | 0.10835 | −27.8 % | 1.2e−14 | 5.3e−15 |
| 1.2 | −23.1 % | +120 % | −11.9 % | 0.12639 | −15.7 % | 1.2e−15 | 6.3e−15 |
| 0.8 | −1.65 % | +6.3 % | −0.74 % | 0.14927 | −0.49 % | 1.3e−14 | 1.7e−14 |
| 0.6 | −0.19 % | +0.15 % | −0.16 % | 0.15018 | +0.12 % | 3.4e−14 | 3.8e−14 |
| 0.4 | −0.02 % | −0.73 % | −0.09 % | 0.15024 | +0.16 % | 3.4e−13 | 2.6e−13 |

The error collapses faster than fourth order between 1.2 and 0.6 because what
is being resolved there is the `tanh` interface profile, whose truncation
error in `W0/dx` is exponential, not the stencil order. Raising the order at
fixed `dx = 1.2 W0` helps but does not substitute for resolution:

| FD order at `dx/W0 = 1.2` | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|
| `V` error | −59.9 % | −23.1 % | −14.7 % | −10.5 % |
| `k_eff` error | −24.9 % | −15.7 % | −13.2 % | −10.3 % |

**Use `dx <= 0.6 W0`.** At `0.8 W0` everything is inside 2 % except the
kinetic coefficient; at `1.2 W0` and above the model is not quantitative.

### The anti-trapping current

`k_eff = k (1 + (1−k) U_s) / (1 + (1−k) U_i)`, which is `c_s/c_l^i` exactly;
it equals `k` if and only if the freshly formed solid and the extrapolated
outer liquid are at the same chemical potential. `dx = 0.6 W0`, four
velocities, three models:

| target `V` | `at_scale = 1` (physical) | `at_scale = 0` (no current) | `at_scale = −1` (sign flipped) |
|---:|---:|---:|---:|
| 0.05 | 0.15003 (+0.02 %) | 0.16165 (+7.8 %) | 0.17409 (+16.1 %) |
| 0.10 | 0.15024 (+0.16 %) | 0.17554 (+17.0 %) | 0.20807 (+38.7 %) |
| 0.20 | 0.15129 (+0.86 %) | 0.22089 (+47.3 %) | 0.58250 (+288 %) |
| 0.40 | 0.15258 (+1.72 %) | 0.24712 (+64.8 %) | 0.44154 (+194 %) |

The three columns share a target velocity, not a measured one: switching the
anti-trapping current off changes the kinetics too, so at a target of 0.4 the
measured velocities are 0.386, 0.708 and 0.873 respectively. The `V = 0.05`
row is from a `1600 W0` box, the rest from `512 W0`, so that the slowest
front's `40 W0` boundary layer still has room (see below).

With the current on, `k_eff` is flat in velocity to under 2 % over an
eight-fold range. Without it, `k_eff` climbs monotonically with `V` — the
textbook signature of spurious solute trapping. With the sign flipped it
climbs faster still, and one of those runs tripped the `|phi| > 1.5`
instability guard. This is the single most diagnostic measurement in the
model and it says the current is right.

The residual `+1.7 %` at the fastest point is the interface Péclet number
`W0 V / D_l = 0.19` showing through: the thin-interface asymptotics is an
expansion in exactly that quantity, so a residual growing roughly linearly
with `V` is what should be there.

### Conservation

Total solute `sum [P(phi)/(1−k) + P(phi) U]` and the latent-heat balance
`sum theta − (1/2) sum phi` are **exact discrete identities** of the scheme
(the derivation is in `step.hpp`), not approximations. Measured drift over
30 000 to 500 000 steps is `1e−15` to `7e−13` relative, i.e. round-off in a
sum over `10^3`–`10^5` cells, growing with cell count and step count exactly
as a floating-point sum should and with no systematic trend. The same holds
in 3-D (`3e−13` on `48^3`) and for models that are physically wrong
(`at_scale = 0`), because conservation is a property of the discretisation
rather than of the physics.

### The initial condition, and why it is not the answer

A planar front started from a uniform melt spends a very long time getting to
steady state: it starts at `V ~ Omega/beta`, sixteen times its steady value,
and the classical initial transient is `D_l/(k V)` in distance — 133 `W0`
here — hence `~1300 tau0` in time. A uniform-melt run in a `2048 W0` box was
still at `V = 0.194` at `t = 1600`. So the planar driver seeds the analytic
steady boundary layer by default (`--analytic-ic=1`).

That is a shortcut, and it has to be shown not to be the answer. Three
controls, all run:

1. **The velocity is not pinned by the seed.** At `dx = 1.6 W0` the run is
   seeded for `V = 0.1` and measures `V = 0.041`; at `dx = 0.4 W0` it
   measures `0.09998`. The model leaves the seeded state whenever it
   disagrees with it, and the resolution study above is precisely the record
   of it agreeing more and more closely as the grid is refined.
2. **`k_eff` and `ell` are seed-independent.** Seeding the layer for
   `V = 0.05`, `0.1` and `0.2` while holding `Omega` at its `V = 0.1` value
   (`--ic-velocity`) gives `k_eff = 0.1496, 0.1498, 0.1485` — the same number
   — and `ell` tracks `D_l/V` throughout, including while `V` is changing.
3. **The correct seed is stationary and a wrong one relaxes toward it.** Over
   `3000 tau0` in a `1200 W0` box: seeded at `0.1`, `V` goes `0.0981 →
   0.0974` (0.7 %); seeded at `0.2`, `V` goes `0.187 → 0.149` and is still
   falling. Seeded at `0.05` it does not visibly move on this timescale,
   which is consistent with the relaxation being one-sided and slow — the
   front has to lay down `D_l/(k V)` of solid to forget a boundary layer, and
   at `V = 0.05` that is 267 `W0`.

The honest summary: `k_eff`, `ell`, `beta` and the mass balance are strong,
seed-independent, grid-convergent measurements, and the mass balance in
particular is about solid the model laid down during the run rather than
about anything that was seeded. The steady **velocity** is the weakest of the
five on achievable run lengths, because the planar state relaxes on the
initial-transient time and relaxes only slowly from below; what the
resolution study demonstrates is that the discrete model's fixed point
converges to the analytic one, which is the substance of the claim.

### Long runs in a closed box drift, on purpose

Past a few hundred `tau0` the two fronts have solidified a noticeable
fraction of the melt, the far field moves, and the "steady" state moves with
it. Beyond `~10 %` solidified the measurement is of a slowly changing
problem. The driver warns when the half-box is shorter than eight boundary
layers, which is the other way to get a biased `U_far`: the two fronts' tails
then overlap at the periodic seam. At `V = 0.05` in a `512 W0` box that
biases `k_eff` by 4 %; in a `1600 W0` box the same run gives `+0.02 %`.

### 2-D dendrite

Shipped case: `240^2` at `dx = 0.8 W0`, `lambda = D_l/a2` (so `beta = 0`,
`d0/W0 = 0.277`), `Omega = 0.55`, `eps4 = 0.2`, `D_th = 2`, `M_c = 0.5`,
`t_end = 400`. About three minutes on one core.

```
t       x_tip     v_tip     rho_tip   fit_rms   sum(theta)
 52.0   114.840   0.12837   7.303     0.039     1241.6
152.0   124.912   0.09066   5.571     0.055     2969.8
252.0   133.087   0.07979   4.921     0.060     4659.3
400.0   144.137   0.07371   4.563     0.059     7216.1
```

The arm is 48 `W0` long against a tip radius of 4.6 `W0`, so it is a dendrite
rather than a growing disc, and both `v_tip` and `rho_tip` are still
decreasing at `t_end`: **this case does not reach a steady tip**, and no
selection constant should be read off it. Latent heat is doing real work —
`sum theta` grows to a mean `theta` of 0.125, so `M_c theta` removes about
11 % of the driving force by the end, and setting `--Mc=0` visibly speeds the
tip up.


#### How much the tip radius depends on the fit window

`rho` is defined by a least-squares parabola through the `phi = 0`
crossings of `2 * half_width + 1` rows centred on the tip row, and the
half-width is a parameter rather than a constant because the parabolic
description is only good within roughly `rho` of the tip: too narrow and
the fit is dominated by the staircase, too wide and it is biased by the
non-parabolic flanks. That is not a hedge — it is measurable, and it was
measured. `t_end = 400`, everything else at the shipped values, percentages
relative to the narrowest window:

| fit half-width (cells) | 3 | 5 | 8 | 12 |
|---|---:|---:|---:|---:|
| `rho/W0`, `eps4 = 0.1` | 11.51 (+0 %) | 11.94 (+4 %) | 12.72 (+10 %) | 13.83 (+20 %) |
| `rho/W0`, `eps4 = 0.2` | 3.94 (+0 %) | 4.66 (+18 %) | 5.49 (+39 %) | 6.43 (+63 %) |

The trend is monotone and in the expected direction — a wider window
reaches the flatter flanks and reports a larger radius — and it is worse
for the sharper tip: at `eps4 = 0.2` the radius is 4.7 `W0`,
i.e. six cells at `dx = 0.8 W0`, so there is barely a window that is both
wide enough to average the staircase and narrow enough to stay parabolic.
**Tip velocity is unaffected** — identical to five significant figures
across every window — because it is a level-set crossing on one row
and does not involve the fit at all.

Rule of thumb for a science run: choose the half-width so the fit spans
about `rho/2`, and resolve the tip with `rho >= 10 dx`. The shipped
dendrite does not meet the second condition; the `eps4 = 0.1` case
(`rho = 12 W0` = 15 cells) does, at the cost of a stubbier arm.
`fit_rms` is in every CSV row so the choice can be audited rather than
assumed.

Two caveats that a science run has to deal with:

- **`D_th = 2` means Lewis number 1.** A metal's is `10^3`–`10^4`. An
  explicit scheme's step is set by the fastest diffusivity, so a realistic
  Lewis number costs three to four orders of magnitude more. Getting a
  quantitative thermo-solutal dendrite needs an implicit or spectral thermal
  solve, which is not in this application.
- **`eps4 = 0.2` is `eps_eff = 0.043`, not 20 %.** With the spec's
  un-normalised `a_s = 1 + eps4 (n_x^4 + n_y^4)` and
  `n_x^4 + n_y^4 = (3 + cos 4 theta)/4`, the anisotropy that the selection
  theory sees is `(eps4/4)/(1 + 0.75 eps4)` — a factor of about four smaller
  than the number in the input. Using the Karma-Rappel `eps4 = 0.02` here
  gives `eps_eff = 0.005` and a blob.

## The elastic coupling

Equation (2) ends with `− lambda_el (1−phi^2)^2 dF_el/dphi`, and
`elasticity.hpp` is what supplies that term. It is *not* an elastic solver:
`openpfc_apps/microelasticity.hpp` is the solver, Eshelby-validated, with a
finite-difference-checked `d f_el/d phi`. This file is the adapter, and the
three jobs it does are the three places a coupled local-FD / global-FFT
application goes quietly wrong.

**1. Two layouts, one grid.** The phase field lives on a padded FD field
(storage halo `fd_order/2`); the elastic solver lives on flat HeFFTe inbox
fields with no halo. They must describe the same owned cells or the coupling
is silently wrong on every rank but zero. The FFT is therefore built from the
*stack's own* decomposition rather than from `nproc` — `SpectralCPUStack`
would build its own process grid, which agrees with the FD stack's only
below nine ranks — and the constructor then refuses to run if the two owned
boxes still differ.

**2. Three fields, not one.** The solver wants `h(phi)`, the eigenstrain
amplitude, and *both* their `phi`-derivatives. Passing `nullptr` for the
derivatives is accepted and yields `dfel_dphi() == 0` everywhere: a coupled
run that is silently uncoupled. All four are always supplied.

**3. Units.** `lambda_el` is not a free knob. Stiffnesses are expressed in
units of `f_ref = L dT_0 / T_M`, and then `lambda_el = lambda` *is* the
calibrated coupling — so a `lambda_el != lambda` is an explicit statement
that the coupling is being scaled, which is legitimate in a sensitivity scan
and impossible to do by accident. `material.hpp` carries the SI data, the
provenance, and the two conversions (`eps_c` is `d eps*/dU`, not
`d eps*/dc`; the factor `(1-k) c_l^0` between them is two orders of
magnitude in the elastic energy).

### Two things worth knowing before reading a coupled result

**Plane strain in 2-D comes out for free.** A 2-D run has `nz = 1`, so every
wave vector has `k_z = 0`, `Gamma_zzkl` vanishes identically and the solve
returns `eps_zz == 0`. That is plane strain, not plane stress: the
dilatational eigenstrain still has a `zz` component, so `sigma_zz` is nonzero
and does work through equation (7). It is the right 2-D reduction for a
dendrite in a thick sample, and it is a property of the discretisation rather
than something the code imposes.

**The `k = 0` mode is a choice.** `eps_hat(0) = 0` clamps the periodic cell
at its mean strain, so a uniformly transforming body develops a uniform
stress that grows with the solid fraction and shrinks with the box volume —
which makes the elastic driving force depend on the domain size. The default
is instead zero mean *stress*, which is a free body and box-independent.
Both are available (`--el-macro=free|clamped`) and the residual mean pressure
is reported every solve.

### Lagging the solve

Mechanical equilibrium is elliptic, so the displacement is slaved to the
instantaneous state; physically the slaving is exact on the acoustic time
scale, ten or more orders of magnitude below `tau0`. The error of re-solving
only every `n_el_substep` steps is therefore not a relaxation error at all,
only a staleness of `df_el/dphi`, bounded by `N dt |V| |grad(df_el/dphi)|`.
The right way to size `N` is to measure what it does to a dendrite
observable, which is what the Stage-4 tables below do.

### The hook itself

```cpp
alloy_dendrite::Stepper<2> st(stack, params, fd_order);
alloy_dendrite::ElasticCoupling el(stack, elastic_params, rank, comm);
st.set_elastic_driving_force(&el.driving_force());   // params.lambda_el != 0
// ... and in the time loop:
if (el.due(step)) el.solve(st.phi(), st.solute(), st.temperature());
```

The hook is a **field**, not a callback, precisely so that the lagged
solution can be reused. A ctest (`[elastic-hook]`) asserts that installing it
with `lambda_el = 0` changes nothing *bitwise*, and that installing it with
`lambda_el != 0` changes the result in the right direction.

### Decomposition consistency, measured

The FD fields are bitwise rank-independent. The elastic fields cannot be:
HeFFTe composes a different sequence of transforms for each process grid, so
`cmp` reports every elastic file as differing and tells you nothing. What
matters is the size of the difference and where it sits.
`scripts/check_decomposition.py` reports both. Over 375 coupled steps, one
rank against four:

| field | `phi` | `U` | `f_el` | `df_el/dphi` | `tr sigma/3` | `sigma_vM` |
|---|---:|---:|---:|---:|---:|---:|
| rel. max-norm | 1.7e−15 | 9.9e−15 | 1.7e−14 | 2.7e−14 | 2.7e−14 | 7.6e−15 |

and every maximum sits on the interface rather than on a subdomain boundary,
which is the other half of the statement.

## No HIP twin, and why

CPU only, deliberately, and not because of FD-order generality — the library
already has `pfc::runtime::gpu` device gradient evaluators and a device halo
exchange, so an order-general device path is available.

The reason is that a correct HIP twin of *this* model is four kernels, three
device halo exchanges, fourteen device fields, and a CPU-versus-GPU parity
test, and none of it can be run from a login node. Shipping an unexercised
GPU path — with a pinned checksum nobody has ever seen produced — would be
worse than shipping none: it would look verified. `apps/kobayashi`'s HIP
twin, the model this was to follow, is two kernels over two fields at fixed
second order; the step here is an order of magnitude more surface area.

The order to do it in, when there is a GPU: the elastic solve of equations
(5)–(7) is an FFT and will need its own device story, and it is better to
port one coupled step than to port half of it now and re-port it later.

## Layout

| File | Contents |
|---|---|
| `include/alloy_dendrite/parameters.hpp` | `ModelParams` and the closed-form thin-interface relations every test compares against. Also the derivation fixing the anti-trapping coefficient. |
| `include/alloy_dendrite/step.hpp` | The four-stage explicit step, the anisotropy of equation (1), and the elastic hook. Explains why the conserved variable is `P(phi) U`. |
| `include/alloy_dendrite/diagnostics.hpp` | Every measurement, defined operationally: conservation, planar front, `k_eff`, tip position and radius, the append-only CSV sink. |
| `include/alloy_dendrite/cases.hpp` | `run_planar` and `run_dendrite<Dim>`, shared by the drivers and the tests. |
| `include/alloy_dendrite/cli.hpp` | `--key=value` parsing that rejects unknown keys. |
| `src/cpu/alloy_dendrite_planar.cpp` | Stage-1 driver. |
| `src/cpu/alloy_dendrite_growth.cpp` | Stage-2/3 driver and the shipped dendrite preset. |
| `tests/test_alloy_dendrite.cpp` | Closed-form relations, the measurements against analytic input, a real Stage-1 run, dimensional consistency, and the elastic hook. |

## Relation to PRs #103 and #104

Those two unmerged branches are a quantitative Al–Cu implementation of the
same physics by a colleague. This application was written from the papers,
not from that code, but reading them changed several decisions:

- **The anti-trapping sign.** Their implementation, the spec, and Karma's
  papers all agree on `+1/(2 sqrt(2))`; the plausible transport argument for
  the opposite sign is wrong, and `parameters.hpp` records the cancellation
  that actually fixes it.
- **The interface CFL.** PR #103 limits `dt` by `0.8 dx/V` so that the front
  cannot cross a cell in one step and leave the discrete `d_t phi` that feeds
  the anti-trapping current under-resolved. That check is here too.
- **Grid pinning of a level-set crossing.** PR #104 found that differencing a
  staircased tip position amplifies the pinning oscillation and that
  higher-order differences make it worse; a least-squares slope over a window
  is the estimator that converges. Both velocity measurements here are
  least-squares slopes for that reason.

Where this deliberately differs:

- **Collocated high-order central FD, not the Ji isotropic 9/19-point
  stencil.** The spec asks for `pfc::gradient::FDGradient`, so grid
  anisotropy is reduced by *order* rather than by an isotropised stencil.
  That is the right trade here — order 4 at `dx = 0.6 W0` is quantitative —
  but a very coarse, very anisotropic case would be better served by their
  operator.
- **`grad phi / |grad phi|` is formed and gated, not eliminated.** They avoid
  the `0/0` by substituting the equilibrium profile so the anti-trapping term
  becomes `div(alpha grad beta)`, which is elegant but assumes the
  equilibrium profile and uses `W0` rather than the anisotropic `W`. The gate
  used here is cruder and assumption-free.
- **The solute variable is `P(phi) U`, not `c`.** Evolving the conserved
  density directly is what makes total solute conserved to round-off rather
  than to `1e-14`-ish with a floor on `c`; there is no `c` floor, no `log`,
  and no `exp` anywhere in the step.
- **`k_eff` is measured by extrapolating the outer solute profile to the
  interface**, not by sampling the first cell past `phi < −0.9`. The
  pointwise version carries a resolution artefact the same size as the effect.
