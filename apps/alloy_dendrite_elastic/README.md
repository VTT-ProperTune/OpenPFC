<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# `alloy_dendrite_elastic` — thermo-solutal solidification core

Quantitative dilute-alloy phase field with anti-trapping current, coupled to
solute and to temperature with latent heat: **equations (1)–(4) of the
capstone model spec** (issue #85).

Equations (5)–(7) — the eigenstrain microelasticity and its Fourier
Green-operator solve — live in
`apps/common/include/openpfc_apps/microelasticity.hpp`. This application
provides the attachment point for them (see [Where the elastic solve
attaches](#where-the-elastic-solve-attaches)) and one driver,
`alloy_dendrite_coupled_cost`, that actually attaches them and measures what
the coupling costs (see [What coupling elasticity
costs](#what-coupling-elasticity-costs)). No science-scale coupled run is
claimed: Stage 0 (Eshelby) is covered by `test_microelasticity.cpp`, and
Stage 4 (elastic off versus on) is still untouched.

| | |
|---|---|
| Fields | `phi` (−1 liquid, +1 solid), `U` (supersaturation), `theta` (undercooling) |
| Discretisation | high-order central FD via `pfc::gradient::FDGradient`, orders 2–14 |
| Parallelism | MPI on `pfc::Domain` / `Box3i`, halo width `order/2` via `pfc::comm::HaloExchange` |
| Time integration | explicit, four stages, three halo exchanges per step |
| Dimensions | 2-D (`nz = 1`) and 3-D from one templated stepper |
| Backends | CPU and **HIP**, measured against each other — see [The HIP twin](#the-hip-twin) |

## Binaries

| Binary | Stage | What it does |
|---|---|---|
| `alloy_dendrite_planar` | 1 | Isothermal planar front measured against the thin-interface prediction: velocity, kinetic coefficient, solute boundary layer, **effective partition coefficient**, and the two conservation invariants. |
| `alloy_dendrite_growth` | 2 / 3 | Deterministic dendrite, 2-D or 3-D (`--nz`), with tip-velocity and tip-radius diagnostics written to CSV. |
| `alloy_dendrite_hip_parity` | — | Runs the CPU stepper and the HIP stepper on one deterministic case and subtracts them; also dumps and compares the gathered global fields so a 1-rank run can be differenced against an N-rank one. Fails on `--tol`. |
| `alloy_dendrite_coupled_cost` | — | Times one coupled step part by part: GPU phase field, device→host, host prep, elastic solve, host→device. Answers whether the elastic solve belongs on the device. |

Both take `--key=value` options; `--help` lists them. Unknown keys are an
error rather than a warning, because a typo that silently reverts a parameter
to its default is exactly how a verification app produces a confident wrong
answer.

**Run these on a compute node, not on a login node.** The ready-made jobs in
`slurm/` do that and are what produced every number below;
`sbatch --job-name=... --export=ALL,ALLOY_PLANAR_BIN=<path> slurm/alloy_dendrite_planar_ladder.sbatch`
is the pattern. The command lines below are what those jobs run.

```bash
# Stage 1, about 10 s on one core
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

# CPU against GPU, and 1 rank against N, on one GCD / eight GCDs
alloy_dendrite_hip_parity --nx=96 --ny=96 --nz=96 --steps=200 --dump=ref.bin
srun -n8 alloy_dendrite_hip_parity --nx=96 --ny=96 --nz=96 --steps=200 \
                                   --compare=ref.bin

# What one coupled step costs, split five ways
alloy_dendrite_coupled_cost --nx=128 --ny=128 --nz=128 --steps=30 --warm-start=1
```

CSV output is **appended, never truncated**, and every row carries the
`--run-id`. A resolution study is a sequence of runs whose output belongs in
one file, and clobbering a ten-minute run is a worse failure than a file with
two headers in it.

## What was measured

Every number below comes from a LUMI **compute node** — `standard` for CPU,
`standard-g` for GPU — and carries the Slurm job id that produced it. The
tables in the first revision of this file were produced on a login node; they
have all been re-run, and the [Re-measured on a compute
node](#re-measured-on-a-compute-node) section says which ones moved and why.

Unless a table says otherwise: single rank, `k = 0.15`, `D_l = 2`,
`lambda = 1` (so `beta = 0.6069`, deliberately nonzero — see below),
`eps4 = 0`, fourth-order stencils, `512 W0` periodic box, `t_end = 600`.

### Resolution study

Job **21916463** (`standard`).

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
velocities, three models. Job **21916670** (`standard`), `2048 W0` box
(`1600 W0` at `V = 0.05`) — **the box matters and the first revision of this
table got it wrong for the fastest row; see below**:

| target `V` | `at_scale = 1` (physical) | `at_scale = 0` (no current) | `at_scale = −1` (sign flipped) |
|---:|---:|---:|---:|
| 0.05 | 0.15004 (+0.03 %) | 0.16223 (+8.2 %) | 0.17553 (+17.0 %) |
| 0.10 | 0.15040 (+0.27 %) | 0.17769 (+18.5 %) | 0.21595 (+44.0 %) |
| 0.20 | 0.15132 (+0.88 %) | 0.23689 (+57.9 %) | 0.44640 (+198 %) |
| 0.40 | 0.15260 (+1.73 %) | 0.25634 (+70.9 %) | 0.45567 (+204 %) |

The three columns share a target velocity, not a measured one: switching the
anti-trapping current off changes the kinetics too, so at a target of 0.4 the
measured velocities are 0.385, 0.711 and 0.873 respectively.

**Only the first column is a converged measurement.** With the current on,
the steady-state mass-balance residual `U_inf − (k U_s − 1)` is `3e−5` to
`5e−4` across all four rows, i.e. the front really is in the travelling-wave
state the diagnostic assumes. With the current off or flipped it is `1e−2` to
`5e−2` and the measured velocity misses its target by up to 120 %, so those
two columns are transients: their qualitative content — `k_eff` climbs
monotonically with velocity — is robust, their digits are not, and they move
by up to 7 % when the box is enlarged. They are quoted to show the direction
of the failure, not as reproducible numbers.

#### The box has to outrun the front

The `V = 0.40`, `at_scale = 1` cell is the one place where "which box" is not
a detail, and it is worth the four extra runs (job **21916585**,
`standard`):

| box | measured `k_eff` | error | mass-balance residual |
|---:|---:|---:|---:|
| `512 W0` | 0.13584 | **−9.44 %** | 9.0e−3 |
| `1024 W0` | 0.15260 | +1.73 % | 9.6e−5 |
| `2048 W0` | 0.15260 | +1.73 % | 9.6e−5 |
| `4096 W0` | 0.15260 | +1.73 % | 9.6e−5 |

At `V = 0.385` over `t_end = 600` each of the two fronts travels `231 W0`
into a `256 W0` half-box: they meet, `U_i` goes positive, and the diagnostic
is measuring a collision rather than a steady front. The rule stated two
sections down — the two fronts' tails must not overlap at the periodic seam —
is exactly what is violated, so this is the rule failing to be applied rather
than a new effect. At `V = 0.20` the same check converges already at
`512 W0` (0.151318 against 0.151324), which is why the error only shows up in
the fastest row.

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
`d0/W0 = 0.277`), `Omega = 0.55`, `eps4 = 0.0435` under the corrected
Karma-Rappel `a_s` (see [The anisotropy
convention](#the-anisotropy-convention)), `D_th = 2`, `M_c = 0.5`,
`t_end = 400`. About two minutes on one core. Job **21916714**
(`standard`).

```
t       x_tip     v_tip     rho_tip   fit_rms   sum(theta)
 50.0   114.526   0.12941   7.634     0.037     1207.5
152.0   124.750   0.09037   5.694     0.049     2971.4
254.0   133.149   0.08033   4.811     0.061     4692.4
398.0   144.147   0.07569   4.376     0.057     7183.0
```

The arm is 48 `W0` long against a tip radius of 4.4 `W0`, so it is a dendrite
rather than a growing disc, and both `v_tip` and `rho_tip` are still
decreasing at `t_end`: **this case does not reach a steady tip**, and no
selection constant should be read off it. Latent heat is doing real work —
`sum theta` grows to a mean `theta` of 0.125, so `M_c theta` removes about
11 % of the driving force by the end, and setting `--Mc=0` visibly speeds the
tip up.

Running the same job with `--aniso-form=unnormalised --eps4=0.2`, i.e. the
pre-correction convention at the same effective anisotropy, gives
`x_tip = 144.137` — the first revision of this table to every quoted digit,
on a compute node — against the corrected form's 144.147, with `v_tip` 2.5 %
higher and `rho_tip` 3.4 % lower. The two are the same physical anisotropy;
what changed is that `a_s` no longer carries the `1 + 0.75 eps4 = 1.15`
offset, so `W(n)` and `tau(n)` are smaller by that factor at every
orientation.


#### How much the tip radius depends on the fit window

`rho` is defined by a least-squares parabola through the `phi = 0`
crossings of `2 * half_width + 1` rows centred on the tip row, and the
half-width is a parameter rather than a constant because the parabolic
description is only good within roughly `rho` of the tip: too narrow and
the fit is dominated by the staircase, too wide and it is biased by the
non-parabolic flanks. That is not a hedge — it is measurable, and it was
measured. `t_end = 400`, everything else at the shipped values, percentages
relative to the narrowest window:

Job **21916714** (`standard`), Karma-Rappel `a_s`:

| fit half-width (cells) | 3 | 5 | 8 | 12 |
|---|---:|---:|---:|---:|
| `rho/W0`, `eps4 = 0.0233` | 13.09 (+0 %) | 13.41 (+2 %) | 14.19 (+8 %) | 15.25 (+17 %) |
| `rho/W0`, `eps4 = 0.0435` | 3.85 (+0 %) | 4.51 (+17 %) | 5.28 (+37 %) | 6.21 (+61 %) |

The trend is monotone and in the expected direction — a wider window
reaches the flatter flanks and reports a larger radius — and it is worse
for the sharper tip: at `eps4 = 0.0435` the radius is 4.5 `W0`,
i.e. six cells at `dx = 0.8 W0`, so there is barely a window that is both
wide enough to average the staircase and narrow enough to stay parabolic.
**Tip velocity is unaffected** — identical to ten significant figures
across every window (0.07590134371 at every half-width) — because it is a
level-set crossing on one row and does not involve the fit at all.

Rule of thumb for a science run: choose the half-width so the fit spans
about `rho/2`, and resolve the tip with `rho >= 10 dx`. The shipped
dendrite does not meet the second condition; the `eps4 = 0.0233` case
(`rho = 13 W0` = 16 cells) does, at the cost of a stubbier arm.
`fit_rms` is in every CSV row so the choice can be audited rather than
assumed.

Two caveats that a science run has to deal with:

- **`D_th = 2` means Lewis number 1.** A metal's is `10^3`–`10^4`. An
  explicit scheme's step is set by the fastest diffusivity, so a realistic
  Lewis number costs three to four orders of magnitude more. Getting a
  quantitative thermo-solutal dendrite needs an implicit or spectral thermal
  solve, which is not in this application.
- **The anisotropy convention changed**; see the next section. Cases written
  against the first revision of this application need `eps4` divided by
  roughly four, or `--aniso-form=unnormalised`.

### The anisotropy convention

`MODEL_SPEC.md` was corrected on **2026-09-11** to the normalised
Karma-Rappel form of equation (1), and that correction is now the default
here:

```
   a_s = (1 - 3 eps4) [ 1 + (4 eps4 / (1 - 3 eps4)) sum_i n_i^4 ]   (default)
   a_s = 1 + eps4 sum_i n_i^4                     (--aniso-form=unnormalised)
```

In 2-D the identity `n_x^4 + n_y^4 = (3 + cos 4 theta)/4` reduces the first
to `1 + eps4 cos 4 theta` **exactly**. Two things follow, and both are
asserted in `ctest -R alloy-dendrite` rather than claimed here:

- the orientation average of `a_s` is 1, so `W0` is the interface width of
  an average orientation rather than of no orientation at all, and the
  `d0/W0` and `dx/W0` bookkeeping means what it says;
- the peak amplitude is `eps4` itself, so a published `epsilon_4` transfers.
  Under the old form the effective strength was `(eps4/4)/(1 + 0.75 eps4)`,
  about a quarter of the input, and Karma-Rappel's 0.02 became 0.005 — too
  weak to select a tip, which grows a blob.

The old form is kept, and kept reachable, because the first revision of this
application measured its Stage-2 table with it and a documented measurement
whose code no longer exists is not reproducible.
`alloy_dendrite::effective_anisotropy(form, eps4)` converts between the two,
and every run header prints `aniso=` and `eps_eff=` so a CSV can be traced to
a convention.

## Re-measured on a compute node

The first revision of this file said "all numbers below are from single-rank
runs on a LUMI login node". Every table has been re-run on `standard` with
the same options. The physics is deterministic, pure FD and single-threaded,
so nothing *should* move; that is a prediction, and here is the check.

**Nothing moved.** The resolution ladder reproduces to every quoted digit
across all five rows and six columns (job **21916463**), the FD-order table
reproduces to every quoted digit (same job), and the Stage-2 dendrite run
under the old convention reproduces `x_tip = 144.137` exactly (job
**21916714**). Running an application on a login node produced the right
numbers; it was still the wrong place to produce them, and they are now
reproducible from a job script.

Three things did come out of the re-run, none of them a login-node artefact:

1. **The anti-trapping table's `V = 0.40` row had the wrong box.** It is
   quoted at `512 W0`, where the two fronts meet before `t_end` and the
   measurement is invalid (`k_eff = 0.13584`, mass-balance residual
   `9.0e−3`). At `1024 W0` and above it converges to 0.15260, which is the
   number the table quotes. Fixed above, with the convergence study that
   settles it (job **21916585**).
2. **The `at_scale = 0` and `−1` columns are not converged measurements**
   and their digits should never have been quoted to five places; see the
   note under that table. Their qualitative content stands.
3. **The 2-rank ctest now runs.** It was registered and unrun because `srun`
   cannot allocate from a login node. `ctest -R "alloy-dendrite|microelasticity"`
   on `standard` is 4/4 green including `alloy-dendrite-planar-2rank`
   (job **21916702**), so the conservation claim, which rests on three halo
   exchange groups, is now exercised across a rank boundary.

## Where the elastic solve attaches

Equation (2) ends with `− lambda_el (1−phi^2)^2 dF_el/dphi`. The whole
attachment surface is two things:

```cpp
alloy_dendrite::Stepper<2> st(stack, params, fd_order);
st.set_elastic_driving_force(&dfel_dphi_field);  // field on the same owned box
// and params.lambda_el != 0
```

The hook is a **field**, not a callback, because the spec allows the
quasi-static solve to be lagged by `n_el_substep` phase-field steps: the
stepper has to be able to reuse a solution computed several steps ago. It is
read in stage B of `step.hpp`, at the line marked `ELASTIC HOOK`, and nothing
else in this application changes when equations (5)–(7) land. There is a
ctest (`[elastic-hook]`) that drives it with a constant field and asserts
that installing it with `lambda_el = 0` changes nothing *bitwise*, and that
installing it with `lambda_el != 0` changes the result in the right
direction.

`DeviceStepper<Dim>::set_elastic_driving_force` is the same hook on the
device, taking a device-resident field and reading it only on owned cells,
so `dF_el/dphi` never needs its halo exchanged. `alloy_dendrite_coupled_cost`
drives the whole chain — solve on the host, push the result into that field,
let stage B read it — and the hook has now been exercised against a real
solver rather than against a constant.

## The HIP twin

Four kernels (`src/hip/alloy_dendrite_hip_kernels.hip`), the same three
device halo exchanges in the same order, sixteen device fields, and the
stencil carried as data so the order stays a run-time choice in `[2, 14]`.
The kernels are transcribed expression by expression from `step.hpp`,
including the parenthesisation, so that the residual difference between the
two paths measures compiler reassociation and nothing else.

The deliverable is not the kernels; it is the two measurements below, and
both are from `standard-g`.

### CPU against GPU

Both steppers run in one process on one decomposition from an initial
condition copied bit-for-bit into the device fields, then the owned cells are
subtracted. Job **21916748**, `96^3`, `dx = 0.8 W0`, order 4,
`eps4 = 0.2` (Karma-Rappel convention, i.e. a deliberately severe 20 %
anisotropy so the branch is unmistakably live), `D_th = 2`, `M_c = 0.5`,
anti-trapping on, 200 steps:

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

### Throughput, for completeness

One MI250X GCD against one Trento core, `96^3`, order 4, job **21916748**:
2.18 ms against 59.5 ms per step, **27x**. That number is a single core, not
a node, and it is not the point of this section; it is here so the cost
ratios in the next one have a denominator with provenance.

## What coupling elasticity costs

`apps/common/include/openpfc_apps/microelasticity.hpp` is host-only by
design, so a GPU phase field coupled to it either needs a device port of the
tensor solve or pays a host round-trip per elastic solve.
`alloy_dendrite_coupled_cost` runs the actual coupled step -- GPU phase
field, `phi/U/theta` down, `h`/`a`/`dh`/`da` built on the host, Eyre-Milton
fixed point, `dF_el/dphi` back up -- and times the five parts separately, so
the route is chosen from a ratio. One GCD, order 4, `dx = 0.8 W0`, default
liquid (`mu_l/mu_s = 0.05`), `tol_el = 1e-6`. Job **21916676**
(`standard-g`), milliseconds per step:

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
coupled step, so porting it is optimising the wrong term, and the task this
work was set explicitly said not to port speculatively. The coupling is
therefore wired exactly as the round-trip arrow above: `--elastic=1` on
`alloy_dendrite_coupled_cost` drives the real solve and feeds
`dF_el/dphi` into stage B's `ELASTIC HOOK` through a device field.

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
question this measurement does not answer, and it needs a Stage-4
convergence study against `N = 1`, `tol_el = 1e-6` before it is used for
science.

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
`SpectralETDOps`, not against a device path: none of the four kernels in this
application reuses it either. At 130x on the floor case such a port would
turn a coupled step from FFT-bound back into FD-bound. It is deliberately
not attempted here, because the measurement was the assignment and porting
before measuring is how the wrong term gets optimised.

## Layout

| File | Contents |
|---|---|
| `include/alloy_dendrite/parameters.hpp` | `ModelParams` and the closed-form thin-interface relations every test compares against. Also the derivation fixing the anti-trapping coefficient. |
| `include/alloy_dendrite/step.hpp` | The four-stage explicit step, the anisotropy of equation (1), and the elastic hook. Explains why the conserved variable is `P(phi) U`. |
| `include/alloy_dendrite/diagnostics.hpp` | Every measurement, defined operationally: conservation, planar front, `k_eff`, tip position and radius, the append-only CSV sink. |
| `include/alloy_dendrite/cases.hpp` | `run_planar` and `run_dendrite<Dim>`, shared by the drivers and the tests. |
| `include/alloy_dendrite/cli.hpp` | `--key=value` parsing that rejects unknown keys. |
| `include/alloy_dendrite/device_step_hip.hpp` | The HIP launch surface: three trivially copyable descriptors and four launchers. Explains why the stencil weights travel unscaled. |
| `include/alloy_dendrite/device_stepper_hip.hpp` | Device twin of `Stepper`: the sixteen device fields, the three halo groups, the residency bookkeeping. Explains why it takes a decomposition rather than a stack. |
| `src/cpu/alloy_dendrite_planar.cpp` | Stage-1 driver. |
| `src/cpu/alloy_dendrite_growth.cpp` | Stage-2/3 driver and the shipped dendrite preset. |
| `src/hip/alloy_dendrite_hip_kernels.hip` | The four kernels, transcribed expression by expression from `step.hpp`. |
| `src/hip/alloy_dendrite_hip_parity.cpp` | CPU-against-GPU and 1-rank-against-N-rank, with a `--tol` that makes it a test. |
| `src/hip/alloy_dendrite_coupled_cost.cpp` | The coupled step with equations (5)–(7) attached through the host, timed in five parts. |
| `slurm/*.sbatch` | The jobs that produced every number in this file. Each one says what question it answers. |
| `tests/test_alloy_dendrite.cpp` | Closed-form relations, the measurements against analytic input, a real Stage-1 run, dimensional consistency, both anisotropy conventions, and the elastic hook. |

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
