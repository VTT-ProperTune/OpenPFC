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
| Backends | CPU for the science path; HIP twin of the FD step, with elasticity remaining host-side — see [The HIP twin](#the-hip-twin) |

## Binaries

| Binary | Stage | What it does |
|---|---|---|
| `alloy_dendrite_planar` | 1 | Isothermal planar front measured against the thin-interface prediction: velocity, kinetic coefficient, solute boundary layer, **effective partition coefficient**, and the two conservation invariants. |
| `alloy_dendrite_growth` | 2 / 3 / 4 | Deterministic dendrite, 2-D or 3-D (`--nz`), with tip-velocity and tip-radius diagnostics written to CSV. `--elastic=1` is Stage 4. |
| `alloy_dendrite_hip_parity` | — | Runs the CPU stepper and the HIP stepper on one deterministic thermo-solutal case and subtracts them; also dumps and compares the gathered global fields so a 1-rank run can be differenced against an N-rank one. Fails on `--tol`. |
| `alloy_dendrite_coupled_cost` | — | Times one coupled GPU step part by part: GPU phase field, device→host, host prep, host Eyre–Milton, host→device. Answers whether the elastic solve belongs on the device. |

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

# CPU against GPU, and 1 rank against N, on one GCD / eight GCDs
alloy_dendrite_hip_parity --nx=96 --ny=96 --nz=96 --steps=200 --dump=ref.bin
srun -n8 alloy_dendrite_hip_parity --nx=96 --ny=96 --nz=96 --steps=200 \
                                   --compare=ref.bin

# What one coupled GPU step costs, split five ways (elasticity stays host-side)
alloy_dendrite_coupled_cost --nx=128 --ny=128 --nz=128 --steps=30 --warm-start=1
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
`DeviceStepper<Dim>::set_elastic_driving_force` is the same hook on the
device, taking a device-resident field and reading it only on owned cells,
so `dF_el/dphi` never needs its halo exchanged. The HIP path does not port
the elastic solve: `alloy_dendrite_coupled_cost` drives the host adapter,
pushes the result into that field, and lets stage B read it.

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
| `--el-mu-liquid=0.001` | 0.205785 | -3.26 % | 266 |

A liquid supports no shear and the solve cannot take zero, so `mu_l/mu_s` is
a regularisation. A 200-fold scan (0.001 to 0.200) moves the slowdown from
3.26 % to 28.60 %; it is still falling at 0.001 (45 Green applications per
solve). The inviscid limit is a few per cent, not seventeen. **The 16.7 %
is the effect at a conventionally chosen regularisation, not a prediction
about Al–Cu.** What is insensitive to it: the sign, the monotonicity in
`lambda_el`, the near-invariance of `sigma*`, and the eigenstrain cancellation.

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

## The HIP twin

The HIP twin is the **finite-difference step** of equations (1)–(4): four
kernels (`src/hip/alloy_dendrite_hip_kernels.hip`), the same three device
halo exchanges in the same order, sixteen device fields, and the stencil
carried as integer weights plus a final scale so the order stays a run-time
choice in `[2, 14]`. The kernels are transcribed expression by expression
from `step.hpp`, including the parenthesisation and the normalised
Karma–Rappel anisotropy, so that the residual difference between the two
paths measures compiler reassociation and nothing else.

Coupled elasticity stays on the host. That is deliberate: the CPU growth
driver already does Stage 2/4 science through `elasticity.hpp`, and the
measurement below says moving the Green-operator solve to the device is the
thing that would matter, not the field copies. `alloy_dendrite_hip_parity`
is thermo-solutal only. `alloy_dendrite_coupled_cost` is the same GPU step
with the host adapter attached, timed in five parts.

The 2-D device path (`nz = 1`, `Axes2D()`, halo width `> 1`) needed a
library fix: `DeviceFacesHalo` used to build its MPI face types with the
default all-active mask and rejected any one-cell axis at `fd_order > 2`.
The host exchanger has passed the active-direction mask since 2-D slabs
above second order were enabled there; the device one now does the same.

The deliverable of the port is the two measurements below, both from
`standard-g`.

### CPU against GPU

Both steppers run in one process on one decomposition from an initial
condition copied bit-for-bit into the device fields, then the owned cells are
subtracted. Job **21916748**, `96^3`, `dx = 0.8 W0`, order 4,
`eps4 = 0.2` (a deliberately severe 20 % anisotropy so the branch is
unmistakably live), `D_th = 2`, `M_c = 0.5`, anti-trapping on, 200 steps:

| | `phi` | `U` | `theta` |
|---|---:|---:|---:|
| `max abs` difference | **4.9e−13** | 5.8e−14 | 4.2e−14 |
| relative to the field's max | 4.9e−13 | 9.8e−14 | 1.9e−13 |
| relative difference of the domain integral | **0** | **0** | 3.9e−16 |

Identical at 1, 2, 4 and 8 ranks, because the comparison is per rank. The
2-D slab (`256^2`, `nz = 1`) is tighter still at **4.4e−16** in `phi`,
because it has one fewer stencil direction to disagree about. Across FD
orders at `64^3`: **2.2e−15** (order 2), 5.8e−14 (4), 4.8e−13 (6),
1.7e−12 (8) — the drift grows with the stencil width, as a sum of more terms
should.

### 1 rank against N ranks, on the GPU

The 1-rank run writes its gathered global fields; the 2-, 4- and 8-rank runs
read them back and subtract, in *global* index order so the comparison is
about the halo exchange and not about who owns what. Job **21916748**:

| ranks | 2 | 4 | 8 | 4 (2-D slab) |
|---|---:|---:|---:|---:|
| `max abs` difference in `phi`, `U`, `theta` | **0** | **0** | **0** | **0** |

Bitwise, not approximately. That is the expected answer and it is worth
saying why it is expected: the four kernels read only through the padded
brick, the exchange writes ghost cells that are exact copies, and no
reduction enters the step — so a correct decomposition cannot change a
single bit. Any nonzero here would have been a bug, which is what makes the
zero worth measuring.

### The tolerance, and why it needs a step count

`ctest -R alloy-dendrite-hip-parity` runs `32^3` for 50 steps and fails above
`2e−11`. A fixed bound is only meaningful together with a run length, and
this is why — same case, `64^3`, order 4, job **21916748**:

| steps | 50 | 200 | 800 | 3200 |
|---|---:|---:|---:|---:|
| `max abs` difference, `phi` | 1.6e−15 | 5.8e−14 | 1.3e−07 | **0.53** |
| relative difference of `sum phi` | 0 | 0 | 1.5e−12 | 1.5e−05 |

The pointwise difference is not drifting, it is **growing exponentially**,
with an e-folding length of order a hundred steps, and at 3200 steps it is
`O(1)`. That is not a bug and it is not fixable: a solid sphere growing into
a supersaturated melt is morphologically unstable, so the two runs' side
branches pick different phases from a one-ULP difference at the tip, exactly
as two runs of the same code on different hardware would. The bottom row is
the honest long-run statement — domain integrals are amplified four to five
orders of magnitude less, because they do not care where the difference
sits. **Compare pointwise below `~10^3` steps and statistically above it.**
That rule is not specific to this application.

### Throughput, for completeness

One MI250X GCD against one Trento core, `96^3`, order 4, job **21916748**:
2.18 ms against 59.5 ms per step, **27x**. That number is a single core, not
a node, and it is not the point of this section; it is here so the cost
ratios in the next one have a denominator with provenance.

## What coupling elasticity costs

The CPU growth driver already couples equations (5)–(7) on the host through
`elasticity.hpp`. A GPU phase field coupled to that same host solver either
needs a device port of the tensor solve, or pays a host round-trip per
elastic solve. `alloy_dendrite_coupled_cost` runs the actual coupled step —
GPU phase field, `phi/U/theta` down, `h`/`a`/`dh`/`da` built on the host
with the same assembly as `ElasticCoupling`, Eyre–Milton fixed point,
`dF_el/dphi` back up — and times the five parts separately, so the route is
chosen from a ratio. One GCD, order 4, `dx = 0.8 W0`, default liquid
(`mu_l/mu_s = 0.05`), `tol_el = 1e-6`, unit stiffness (`E = 1`,
`nu = 0.3`) rather than the Al–Cu material of the science path. Job
**21916676** (`standard-g`), milliseconds per step:

| grid | `t_pf` (GPU) | round trip | `t_el` warm | `t_el` cold | round trip / `t_pf` | `t_el` / `t_pf` | `t_el` / round trip |
|---|---:|---:|---:|---:|---:|---:|---:|
| `96^3` | 2.02 | 4.14 | 1060 | 2058 | 2.0 | **504** | 256 |
| `128^3` | 2.70 | 8.72 | 2918 | 5517 | 3.1 | **1052** | 335 |
| `192^3` | 5.80 | 28.7 | 18049 | 34631 | 4.9 | **3112** | 629 |

**The round-trip is not the problem.** It is two to five times one GPU
phase-field step in relative terms and four to twenty-nine milliseconds in
absolute terms, and it is a tenth to a third of a percent of the coupled
step. Porting it away would buy 0.3 %. The host FFT solve is 99.7 % of the
step, and it is 250 to 630 times the round-trip it is supposedly being
weighed against — and the gap widens with the grid, because the transform
grows like `N log N` against the phase-field step's `N`.

### Warm start halves it, and does not change the answer

`MicroelasticityParams::warm_start` reuses the previous solution as the
initial iterate. In this time loop, where the interface moves a small
fraction of a cell per step, it halves the iteration count and therefore
halves the cost:

| | iterations | `t_el` at `96^3` | at `128^3` | at `192^3` |
|---|---:|---:|---:|---:|
| cold | 16.5 | 2058 ms | 5517 ms | 34631 ms |
| warm | 8.5 | 1060 ms | 2918 ms | 18049 ms |

Exactly half the iterations at every grid, stable across the run (min 8, max
9, no trend), which is what "the interface moves a fraction of a cell per
step" should buy. A factor of two against a factor of a thousand. It is
worth having on -- it is the default -- but it does not move the decision.

### The floor: one Green application

The interesting question is not how many iterations, it is what one costs.
Job **21916767** (`standard-g`), `128^3`, warm start, `tol_el` swept:

| `tol_el` | 1e−3 | 1e−4 | 1e−5 | 1e−6 |
|---|---:|---:|---:|---:|
| iterations | 1.00 | 2.25 | 5.05 | 8.52 |
| `t_el` (ms) | **362** | 777 | 1732 | 2918 |

and the same job pins the iteration ladder directly, with `n_el_iter`
capped and `tol_el = 0` so the count is exactly what is asked for:

| Green applications | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| `t_el` (ms) | 376 | 709 | 1430 | 2738 |

Straight line: **337 ms per Green application** plus **38 ms** of fixed work
(the closing pass that builds the stress, `f_el` and `dF_el/dphi`). One
application is twelve host transforms of `128^3`, i.e. ~28 ms per transform
on one core. So the floor of the host route -- loosest useful tolerance,
warm-started, a single Green application -- is **362 ms against a 2.78 ms
GPU step, 130x**. No amount of loosening the tolerance closes that, because
the last factor being loosened away is one FFT pass.

### The feedback switched on

Everything above is timed with `lambda_el = 0`, so the solve runs and is
paid for but its answer does not reach equation (2). Turning it on
(`--lambda-el=1`, job **21917122**, `128^3`) costs 3091 ms against 2918 ms
— 6 %, from the phase field moving differently and the warm start being
correspondingly less warm — and the loop runs to completion, which is the
end-to-end statement that the chain solve → `push_owned` → device field →
stage B's `ELASTIC HOOK` is live. It is a cost measurement, not a physics
result: Stage 4 lives on the CPU growth driver.

### How soft the liquid is

`microelasticity.hpp` calls the liquid shear modulus a regularisation
parameter rather than a material constant and prices it on a `32^3` cold
start. In this time loop, warm-started at `128^3` (job **21917122**), the
price is the same shape but the absolute cost is what matters:

| `mu_l / mu_s` | 0.01 | **0.05** (default) | 0.1 |
|---|---:|---:|---:|
| iterations, warm | 15.6 | 8.5 | 6.4 |
| `t_el` (ms) | 5289 | 2918 | 2204 |

(An independent longer replicate, job **21916676**, gives 14.6 / 8.5 / 6.2
iterations and 5037 / 2918 / 2125 ms — the same numbers to about 5 %.)

The soft end of the literature range costs 2.4x the stiff end. It is a knob
worth knowing about, and it is not a knob that changes the conclusion: even
the cheapest liquid leaves the solve at 800x the phase-field step.

### Eight GCDs

`256³` over eight GCDs is `128³` per GCD, so it lines up with the middle row
of the single-GCD table. Job **21917083**, forced `1×1×8` slab, ms/step:
`t_pf` 3.13, round trip 22.9, `t_el` 5437 — the same shape of answer, with
the round-trip at 7.3× the phase-field step and the solve at **1737×**. The
distributed FFT does not rescue the solve: it is eight ranks doing 12
transforms of `256³` between them against eight GCDs doing an FD step, and
the ratio is worse than at one rank, not better.

One aside worth recording because it is easy to get wrong: at eight ranks
`spectral_fft_proc_grid` does *not* return a slab. Its slab threshold is nine
ranks (`kSpectralSlabMinRanks`), so at eight it hands back the same
minimum-surface `2×2×2` brick `decomposition::create` would, and the
phase-field step is then 7.17 ms against 3.13 ms for a forced `1×1×8` — a
factor of 2.3, because a `z`-slab keeps whole `x`–`y` planes contiguous. The
price of *aligning* the FD grid with the FFT's, which is what makes the
round-trip a plain copy, is separately measured at **7 %** on the
phase-field step (7.17 ms aligned against 6.68 ms on the FD-optimal brick,
job 21917039) — small, and much smaller than the redistribution it avoids.

### An intermittent 8-GCD fault

Worth stating plainly rather than leaving for someone else to hit: **at eight
ranks per node, with the GPU-aware device halo path and the host FFT solve
both live on `MPI_COMM_WORLD`, the coupled driver aborts with a GPU memory
fault in about a third of runs.** Ten repeats of an eight-step `64³` run each
way, job **21917129**:

| configuration | runs completed |
|---|---:|
| GPU-aware device halo, elastic solve on | **7 / 10** |
| packed (non-GPU-aware) device halo, elastic solve on | 10 / 10 |
| GPU-aware device halo, `--elastic=0` | 10 / 10 |

and, from job **21917107**, `--n-el-substep=1000` — the solver constructed
and allocated but never called — is also clean over 20 steps, as is
`MPICH_GPU_SUPPORT_ENABLED=0`. So it needs the elastic solve to actually run
*and* the halo exchange to be passing device pointers to MPI; removing
either removes it. It is not rank-deterministic (the aborting rank differs
run to run) and it is not size-dependent (`64³`, `128³` and `256³` all show
it), which is what rules out an indexing bug on this side.

It is **not** root-caused here, and the honest reading is that this is the
interaction between device-pointer point-to-point and HeFFTe's host-buffer
collectives on one communicator under Cray MPICH, not something in the four
kernels — the HIP twin alone runs 200 steps at eight GCDs bitwise-clean (job
21916748). The workaround, `OPENPFC_HIP_FORCE_PACKED_HALO=1`, is set in
`slurm/alloy_dendrite_coupled_cost.sbatch` with a comment saying why, and
costs nothing measurable here. Single-GCD runs — which is where every number
the route decision rests on was taken — are unaffected.

### The route taken, and the one not taken

**Keep the host round-trip.** The measurement says it costs 0.3 % of the
coupled step, so porting it is optimising the wrong term. The host *solve*
is 500 to 3000 times the GPU step. The CPU application already couples that
way; the GPU measurement says there is no reason to do otherwise until a
device Green operator exists.

Two levers make the host route affordable today, both legitimate because the
mechanics are quasi-static — and the first of them does **not** do what the
arithmetic suggests, which is the reason to measure it. Job **21917122**,
`128^3`, one GCD, warm start:

| `n_el_substep` | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| iterations per solve | 8.3 | 9.9 | 10.1 | 13.0 |
| `t_el` per solve (ms) | 2788 | 3317 | 3511 | 4385 |
| **amortised per step (ms)** | **2788** | **1659** | **702** | **439** |
| speed-up against `N = 1` | 1.0 | 1.7 | 4.0 | **6.4** |

Lagging does not amortise as `1/N`. Skipping a solve makes the next one
harder — the phase field has moved `N` cells' worth, so the warm start is
`N` times staler — and the iteration count climbs from 8.3 to 13.0 across
the ladder. At `N = 10` the honest number is 6.4x, not 10x.

How honest: an independent 60-step replicate (job **21916676**, against the
40-step runs above) gives 9.8 / 10.0 / 11.6 iterations and 1652 / 680 /
391 ms amortised at `N` = 2 / 5 / 10, i.e. a speed-up of 1.7 / 4.1 / **7.1**.
The two runs bracket `N = 10` at **6.4–7.1x**, and they differ because the
iteration count depends on how far the interface has travelled by the time
the window is sampled. Quote the range, not either endpoint. Both runs
agree that it is not 10x, which is the part that matters.

Loosening `tol_el` is the cheaper lever, and combining the two is cheaper
still but again not multiplicatively:

| | `tol_el = 1e-6` | `tol_el = 1e-3` |
|---|---:|---:|
| `N = 1` | 8.3 its, 2788 ms/step | 1.0 its, **364 ms/step** |
| `N = 10` | 13.0 its, 439 ms/step | 3.25 its, **112 ms/step** |

So the best measured configuration is `N = 10`, `tol_el = 1e-3`: an elastic
cost of 112 ms amortised per step against 2.7 ms of phase field and 10 ms of
round-trip, i.e. a coupled step about **48 times** the bare GPU step rather
than a thousand. (Arithmetic would have predicted 36 ms and 14x from the
single-lever numbers; the measurement says 112 ms and 48x, because at
`N = 10` one Green application is no longer enough and the count goes to
3.25.)

Whether a solve lagged ten steps at `1e-3` is *accurate* enough is a physics
question this measurement does not answer. The CPU Stage-4 lagging study
above (0.07 % staleness error at `N = 20` against a 16.7 % effect) is the
place that question is answered for science; this table is the cost of
doing it on a GPU phase field.

**What would justify a device elastic solve** -- and the numbers now say what
it would buy -- is the twelve transforms per Green application, not the
round-trip and not the pointwise tensor work. The library already has the
first half: `pfc::sim::stacks::GPUSpectralStack` wraps
`pfc::fft::IDeviceFFT` over HeFFTe's rocFFT backend, so the transforms are a
stack swap rather than new machinery. What is missing is the local half --
the polarisation, the Green contraction and the Eyre-Milton reflection, all
six components with a spatially varying stiffness -- and that is kernels of
exactly the shape of the four in `alloy_dendrite_hip_kernels.hip`. The
obstacle `microelasticity.hpp` documents, that such a contraction does not
fit `SpectralETDOps`, is real, but it is an argument against reusing
`SpectralETDOps`, not against a device path. At 130x on the floor case such
a port would turn a coupled step from FFT-bound back into FD-bound. It is
deliberately not attempted here, because the measurement was the assignment
and porting before measuring is how the wrong term gets optimised.

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
| `include/alloy_dendrite/device_step_hip.hpp` | The HIP launch surface: three trivially copyable descriptors and four launchers. Explains why the stencil weights travel unscaled. |
| `include/alloy_dendrite/device_stepper_hip.hpp` | Device twin of `Stepper`: the sixteen device fields, the three halo groups, the residency bookkeeping. Explains why it takes a decomposition rather than a stack. |
| `scripts/check_decomposition.py` | Compares two snapshot directories written at different rank counts, reporting the relative max-norm and where it sits. |
| `src/cpu/alloy_dendrite_planar.cpp` | Stage-1 driver. |
| `src/cpu/alloy_dendrite_growth.cpp` | Stage-2/3/4 driver and the shipped dendrite preset. |
| `src/hip/alloy_dendrite_hip_kernels.hip` | The four kernels, transcribed expression by expression from `step.hpp`. |
| `src/hip/alloy_dendrite_hip_parity.cpp` | CPU-against-GPU and 1-rank-against-N-rank, with a `--tol` that makes it a test. Thermo-solutal only. |
| `src/hip/alloy_dendrite_coupled_cost.cpp` | The coupled GPU step with equations (5)–(7) attached through the host adapter, timed in five parts. |
| `slurm/*.sbatch` | The jobs that produced the HIP parity and coupled-cost numbers, and the compute-node re-runs of the planar/Stage-2 tables. |
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
