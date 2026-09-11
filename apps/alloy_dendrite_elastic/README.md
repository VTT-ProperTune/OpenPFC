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

### 2-D dendrite: a steady tip, and what it takes

Shipped case: `600^2` at `dx = 0.8 W0`, `lambda = D_l/a2` (so `beta = 0`,
`d0/W0 = 0.277`), `Omega = 0.55`, `eps4 = 0.04`, **isothermal**
(`M_c = 0`, `--evolve-theta=0`), `t_end = 2000`.

Latent heat being off by default is a deliberate reversal. A closed periodic
box with latent-heat release has **no steady tip**: equation (4) has no sink,
`sum theta` grows monotonically, `M_c theta` eats the driving force and the
tip decelerates for as long as the run lasts. The previous revision shipped
exactly that case and correctly reported that it never reached a plateau.
The thermal coupling is one flag away (`--Mc=0.5 --evolve-theta=1 --Dth=20`)
rather than on, and the elastic section below shows what it does.

Three conditions have to hold before a selection parameter is worth quoting,
and they are independent.

**Box.** At `eps4 = 0.02`, `dx = 0.8`:

| box (W0) | 320 | 480 | 640 | 800 |
|---|---:|---:|---:|---:|
| `V` | 0.09001 | 0.08758 | 0.087936 | 0.087936 |
| `dV/V` over the fit window | -10.3 % | -3.9 % | -2.0 % | -2.0 % |

640 and 800 agree to six digits. It is the solute *reservoir* that sets this,
not the diffusion length: the tip radius is 10 W0 and `D_l/V` is 23, both
tiny next to 640, but four arms keep rejecting into a closed domain.

**Steadiness.** `dV/V` across the trailing 30 % of a `t = 2000` run:

| `eps4` | 0.010 | 0.015 | 0.020 | 0.025 | 0.030 | 0.040 | 0.050 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dV/V` | -12.5 % | -6.8 % | -3.9 % | -1.1 % | -0.27 % | **+0.15 %** | +0.15 % |

Use `eps4 >= 0.03`. A sharper tip reaches steady state sooner.

**Resolution — and buy it with the stencil, not the grid.** `640 W0` box,
`eps4 = 0.04`. Wall times are a matched `t = 50` run on 16 ranks of one
`standard` node, so they compare directly:

| `dx/W0` | order | `V` | `rho` (1-rho window) | `sigma*` | wall | cost |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 4 | 0.194243 | 3.8983 | 0.37531 | 9.0 s | 1.0 |
| 0.80 | 6 | 0.208537 | 3.6508 | 0.39860 | 9.1 s | 1.0 |
| 0.80 | 8 | 0.212221 | 3.5913 | 0.40475 | 11.0 s | 1.2 |
| 0.80 | 10 | 0.213529 | 3.5704 | 0.40699 | 12.3 s | 1.4 |
| 0.80 | 12 | **0.214104** | 3.5618 | 0.40787 | 13.2 s | **1.5** |
| 0.65 | 4 | 0.208069 | 3.6123 | 0.40805 | 15.7 s | 1.8 |
| 0.50 | 4 | 0.213347 | 3.3515 | 0.46229 | 51.9 s | 5.8 |
| 0.40 | 4 | **0.214287** | -- | -- | 147.1 s | **16.4** |

**Fourth order at `dx = 0.8 W0` is 10 % wrong in `V`** and nothing about the
run looks wrong: drift 0.09 %, conservation 1e-13, a clean parabolic tip.
Order 12 at `dx = 0.8` and order 4 at `dx = 0.4` agree on `V` to 0.09 %, and
the first is 11x cheaper -- halving `dx` multiplies cells by `2^d` and steps
by 4, widening the stencil multiplies neither.

#### How much the tip radius depends on the fit window

`rho` is a least-squares parabola through the `phi = 0` crossings of
`2*half_width + 1` rows centred on the tip. The half-width is a parameter of
the *measurement*, and `measure_tip_scan` reports four of them every sample
so the ambiguity is part of the output rather than a hedge.

Windows in **cells** answer "does the grid resolve the fit". Windows in
**units of `rho`** (`measure_tip_scan_relative`, the literature convention,
0.5/1/1.5/2) answer "is the shape a parabola". Those are different questions:

| | `dx = 0.5` | `dx = 0.4` |
|---|---:|---:|
| `V` | 0.21273 | 0.21429 (converged to 0.7 %) |
| relative-window spread | 48.9 % | 49.0 % (**does not move**) |

A spread that does not shrink under refinement is not a discretisation error.
At `eps4 = 0.04` the tip radius is about 3.5 `W0` and a dendrite stops being
a paraboloid within roughly one radius of the apex, so the 2-rho window is
already on the flank. `sigma*` from this application therefore carries a
systematic uncertainty of order +/-50 % from the radius definition alone
(`sigma* ~ rho^-2`). What it does *not* carry is run-to-run scatter: the
model is deterministic and two runs of a case agree bitwise, which is why a
*ratio* of two `sigma*` measured the same way is good to far better than that.

**Tip velocity is unaffected by the window** -- identical to five significant
figures across every choice -- because it is a level-set crossing on one row
and never touches the fit.

#### Caveats a science run has to deal with

- **`D_th = 2` is Lewis number 1.** A metal's is `10^3`-`10^4`. An explicit
  scheme's step is set by the fastest diffusivity, so a realistic Lewis
  number costs three to four orders of magnitude more. The thermo-solutal
  runs below use `D_th = 20` (Lewis 10), which is enough to make the thermal
  eigenstrain act and is not a quantitative alloy prediction.
- **The step limit is order-aware, and did not used to be.** The von Neumann
  bound for the order-`p` central Laplacian is set by its Nyquist eigenvalue
  -- 4, 5.33, 6.04, 6.42, 6.68, 6.87 for orders 2 to 12, tending to `pi^2` --
  not by the order-2 value of 4. `explicit_dt_limit` now reads it out of the
  same coefficient table the stepper differentiates with.

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

### Stage 4: what the coupling does

`960^2` at `dx = 0.5 W0` (a 480 W0 box), `eps4 = 0.04`, `t_end = 1000`,
64 ranks. The reference is `--elastic=1 --lambda-el=0`, i.e. the same code
path with the feedback switched off, so the comparison isolates the coupling
and not the machinery.

| `lambda_el/lambda` | `V` | `dV/V` | `rho` | `drho/rho` | `sigma*` | `int f_el` |
|---|---:|---:|---:|---:|---:|---:|
| off | 0.212722 | -- | 3.3960 | -- | 0.45158 | 0 |
| 0 (solve, no feedback) | 0.212722 | +0.00 % | 3.3960 | +0.00 % | 0.45158 | 440 |
| 0.25 | 0.202754 | -4.69 % | 3.4962 | +2.95 % | 0.44702 | 443 |
| 0.5 | 0.193325 | -9.12 % | 3.5865 | +5.61 % | 0.44551 | 443 |
| **1 (calibrated)** | **0.177228** | **-16.69 %** | **3.7678** | **+10.95 %** | **0.44032** | **416** |
| 2 | 0.152673 | -28.23 % | 4.3138 | +27.02 % | 0.38995 | 355 |
| 4 | 0.118399 | -44.34 % | 5.5564 | +63.61 % | 0.30308 | 278 |

`sigma*` moves 2.5 % while `V` moves 16.7 %: the elasticity slides the
operating point along the solvability curve rather than changing the
selection, which is what an extra penalty on the driving force should do.

Four controls:

- `--lambda-el=0` with the solve running reproduces the reference to ten
  digits while reporting 440 units of stored energy and 10.5 iterations.
- `--el-soften=1` (the unsoftened 300 K constants, 2x the stiffness) gives
  *exactly* the dynamics of `lambda_el = 2 lambda` with exactly `2.0000x` the
  energy. `f_el` is linear in `C` and the driving force is `lambda_el *
  df_el/dphi`, so those two must coincide; that they do to every digit
  verifies the unit bookkeeping in `material.hpp` end to end.
- `--eps-c=0` in the isothermal case zeroes the effect exactly (`f_el = 0`,
  one iteration) and `--eps-T=0` changes nothing, because `theta` is
  identically zero there. **The isothermal effect is entirely solutal.**
- Lagging: at `--n-el-substep=20` the staleness error is 0.07 % against a
  16.7 % effect, and it is linear in the lag as the quasi-static argument
  requires (-0.024, -0.070, -0.230, -0.507 % at N = 10, 20, 50, 100 relative
  to N = 5).

#### The two eigenstrains partly cancel

Copper contracts the aluminium lattice (`eps_c < 0`) and the latent heat it
releases expands it (`eps_T > 0`), in the same place at the same time.
`600^2`, `dx = 0.8`, `--Mc=0.5 --evolve-theta=1 --Dth=20`, `t_end = 250`:

| | `V` | `dV/V` | `int f_el` |
|---|---:|---:|---:|
| off | 0.121417 | -- | 0 |
| solutal only (`--eps-T=0`) | 0.113783 | -6.29 % | 55.5 |
| thermal only (`--eps-c=0`) | 0.118994 | -2.00 % | 11.9 |
| **both** | 0.120033 | **-1.14 %** | **18.8** |

**Adding a second source of misfit reduces the stored energy by a factor of
three.** For co-located scalar eigenstrains the energies combine as
`(a +/- b)^2`; the single-source runs give `|a| = 7.45`, `|b| = 3.45`, so
same-sign predicts 118.9 and opposite-sign 16.0 against a measured 18.8. The
residual is the two fields having different spatial shapes -- `theta`
diffuses ten times faster than `U`, so they cancel where they overlap and
not elsewhere.

#### Cost

One elastic solve is about **41 finite-difference steps** at `600^2` on 32
ranks (1.52 s against 62.93 s for `t = 20`), rising to roughly 166 at
`1280^2` on 64 ranks as the all-to-all takes a larger share. That is what
makes `n_el_substep` load-bearing rather than an optimisation. The solve's
own 1-to-32-rank scaling is 12.8x (40 % efficiency) -- the honest cost of a
global solve inside a local time loop. What does *not* degrade is the
iteration count: 12.65, 12.82, 12.86 at `128^3`, `256^3`, `512^3`.

#### Sensitivity: the liquid shear modulus is the dominant systematic

| choice | `V` | `dV/V` vs off | `int f_el` |
|---|---:|---:|---:|
| default: free body, `mu_l/mu_s = 0.05` | 0.177228 | -16.69 % | 416 |
| clamped cell (`--el-macro=clamped`) | 0.164579 | -22.63 % | 892 |
| `--el-mu-liquid=0.02` | 0.191819 | -9.83 % | 360 |
| `--el-mu-liquid=0.10` | 0.163671 | -23.06 % | 456 |

A liquid supports no shear and the solve cannot take zero, so `mu_l/mu_s` is
a regularisation. It is not a small one: the effect runs 9.8, 16.7, 23.1 %
across a fivefold range in `mu_l`, roughly logarithmically, with no sign of
settling over the range a solve can afford. **Read the magnitude as "of order
10-20 % at a defensible regularisation", not as `16.7 +/- 0.1`.** What is
insensitive to it: the sign, the monotonicity in `lambda_el`, the
near-invariance of `sigma*`, and the eigenstrain cancellation.

The macroscopic-strain condition is worth another 6 points of the 17: a
clamped cell stores `2.1x` the energy, because a uniformly transforming body
that cannot expand carries a uniform stress on top of the structured one.

### 3-D, at LUMI scale

| case | ranks | `t_end` | `V` | `rho` | `int f_el` | iterations |
|---|---:|---:|---:|---:|---:|---:|
| `128^3` off / on | 128 | 60 | 0.36004 / 0.33332 | 6.290 / 7.428 | 0 / 542 | -- / 12.65 |
| `256^3` off / on | 512 | 120 | 0.47139 / 0.44311 | 4.971 / 4.962 | 0 / 1712 | -- / 12.82 |
| `512^3` on | 2048 | 150 | 0.44313 | 5.462 | 2941 | 12.86 |

`256^3` and `512^3` agree on `V` to `6e-5` across an eightfold change in cell
count. `128^3` disagrees with both by 24 %: a `102 W0` box is too small.
**These runs demonstrate that the coupled machinery works, scales and stays
consistent; they are not a 3-D dendrite science result** -- `t_end = 150` is
a rounded `<100>` cross, not a developed dendrite.

### The solution is four-fold symmetric, and nothing enforces that

The model is invariant under `x <-> y`. The implementation is not obviously
so: stencils are applied axis by axis, the r2c transform treats `x` as the
half-complex axis, and the MPI decomposition is a slab. After 160000 coupled
steps, `max|phi - phi^T| = 9e-14` and the four arms reach 398 cells in every
direction. The anisotropy flux, the anti-trapping current, the Green
operator's Nyquist folding and the halo exchange all have to be right for
that to hold, and it costs nothing to check.

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

## Backends: CPU here, and why the GPU path is not finished

This application is CPU only. That is a current state rather than a
principle, and the reasoning has changed since the first revision of this
file, so it is worth being precise about what exists.

A HIP twin of the finite-difference step -- four kernels, three device halo
exchanges, fourteen device fields, and a CPU-versus-GPU parity test -- is
written and measured, on `standard-g` rather than on a login node. It is not
in this branch; it is a separate reviewable change, because it also needed a
library fix (`DeviceFacesHalo` built its MPI face types before resolving the
active direction set, which rejected any 2-D device application above second
order).

The elastic solve is host-side, and the measurement says that is currently
the right place for it *and* that it is the thing to fix next. On a GPU
build the host round trip -- copying `phi`, `U`, `theta` down and
`df_el/dphi` back -- is **0.3 % of the coupled step**. The host *solve* is
**500 to 3000 times the device step**. So the cost is entirely in the solve
and not at all in the transfer, which means there is nothing to gain from
clever overlapping and everything to gain from a device Green operator over
rocFFT. Half of that exists already in `GPUSpectralStack`. Until it lands, a
GPU coupled run would spend all its time on the host and there is no reason
to do one.

The order matters: porting the finite-difference step first and the elastic
solve later would mean porting one coupled step twice.

## Layout

| File | Contents |
|---|---|
| `include/alloy_dendrite/parameters.hpp` | `ModelParams` and the closed-form thin-interface relations every test compares against. Also the derivation fixing the anti-trapping coefficient. |
| `include/alloy_dendrite/step.hpp` | The four-stage explicit step, the anisotropy of equation (1), and the elastic hook. Explains why the conserved variable is `P(phi) U`. |
| `include/alloy_dendrite/diagnostics.hpp` | Every measurement, defined operationally: conservation, planar front, `k_eff`, tip position and radius, the append-only CSV sink. |
| `include/alloy_dendrite/cases.hpp` | `run_planar` and `run_dendrite<Dim>`, shared by the drivers and the tests. |
| `include/alloy_dendrite/cli.hpp` | `--key=value` parsing that rejects unknown keys. |
| `include/alloy_dendrite/elasticity.hpp` | Equations (5)-(7) wired onto the FD stack: the layout contract between the padded FD field and the HeFFTe inbox, the eigenstrain assembly, the macroscopic-strain condition, and the `lambda_el` calibration. |
| `include/alloy_dendrite/material.hpp` | Al-4.5 wt% Cu in SI with provenance, and the arithmetic that turns it into the dimensionless inputs of (5)-(7). Every value is cited or derived next to its use; values that are representative rather than assessed say so. |
| `include/alloy_dendrite/field_output.hpp` | Raw-brick snapshots plus a JSON manifest, correct at any rank count. |
| `scripts/check_decomposition.py` | Compares two snapshot directories written at different rank counts, reporting the relative max-norm and where it sits. |
| `src/cpu/alloy_dendrite_planar.cpp` | Stage-1 driver. |
| `src/cpu/alloy_dendrite_growth.cpp` | Stage-2/3 driver and the shipped dendrite preset. |
| `tests/test_alloy_dendrite.cpp` | Closed-form relations, the measurements against analytic input, a real Stage-1 run, dimensional consistency, the order-aware step limit, and the coupled elastic path. |

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
