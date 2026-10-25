<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# EHD film (`apps/ehd_film`)

Elastohydrodynamic **thin film under a flexible plate**. The linear
constant-mobility model is the 0.2 application for GitHub issue `#81`; the
nonlinear compliant-lubrication model below with a localized load and
adhesion is the science upgrade for `#116`.

Binaries: `ehd_film` (CPU, linear verifier/demo), `ehd_film_nonlinear` (CPU,
nonlinear science driver), `ehd_film_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on (linear model only — see "Nonlinear model" below for why).

## Problem setup: localized load on a compliant lubricated plate

The `Problem setup` block the report contract (`#112`) asks every
application to carry, for the `ehd_film_nonlinear` **science preset**
(`inputs_json/load_relaxation_stiff.json` /
`load_relaxation_compliant.json`). The compact single-mode `ehd_film`
**verification preset** is described separately below.

| Item | Description |
|---|---|
| **Use case** | A viscous film squeezed between a substrate and a flexible plate (e.g. a compliant coating, a soft lubrication bearing, or a stamped/rolled film) receives a localized press, then the load lifts off and the film redistributes |
| **Question** | Under the same localized load, how does cubic lubrication mobility change the forced and post-release response relative to the exact constant-mobility bending+tension solution, and does plate stiffness alter that nonlinear departure? |
| **Domain** | 2-D periodic patch, 256² cells, `dx = 1` (code length units) |
| **Boundary conditions** | Periodic on both axes — see "Why periodic" below |
| **Initial condition** | `h(x,y,0) = h0` (uniform gap); the disturbance comes entirely from the applied load, not the IC |
| **Key parameters** | `h0=1`, `M0=1`, `gamma=10`, `A=0` (adhesion off in the science preset — see "Deferred" below); load `p0=0.5`, `a=8`, `t_load=60`; two bending stiffnesses `B=640` (stiff) and `B=100` (compliant) |
| **Observable** | Central gap `h_center(t)` / deflection, pressure extrema, RMS spreading radius, displaced volume, total volume (conservation check) — all in the diagnostics CSV |
| **Model maturity** | numerical verification: **analytical** (unforced $k^6$ decay, and the `#610` forced/switch-off linear oracle on the same sampled load; both are reproduced by the constant-mobility flux path, see Tests) · physical completeness: **reduced** (2-D lubrication + Kirchhoff bending + tension + optional adhesion; no plate inertia, no cavitation, no contact) · calibration: **none** — `B`, `gamma`, `M0`, the load shape and its amplitude are illustrative nondimensional grid-unit parameters, not fit to a material |

### Why periodic

A periodic patch represents a small window of a much larger plate/film far
from any physical edge or support — the right idealisation for asking how a
*local* disturbance decays and spreads, as opposed to how the plate responds
globally to its actual boundary conditions (clamped edge, simply supported
edge, etc.), which this app does not model.

### Domain-adequacy check

The patch must stay large compared with the disturbance over the reported
interval, or the periodic images interact and the "spreading" measurement
is really measuring the box size. Checked directly, not assumed: the
domain half-width is `128` grid units; the RMS spreading radius (see
`ehd_film::sample_ehd_film`) measured over the full `t in [0, 600]` run
peaks at `27.4` (stiff plate) and `23.1` (compliant plate) — under 22% of
the half-width in both cases.

## Nonlinear model (`#116`)

$$
p = B\nabla^4h - \gamma\nabla^2h - \Pi(h) + p_{\mathrm{ext}}(x,y,t),
\qquad
\partial_t h = \nabla\cdot\bigl[M(h)\nabla p\bigr],
\qquad
M(h) = M_0\left(\frac{h}{h_0}\right)^{3}.
$$

`M(h)` is the cubic no-slip lubrication mobility (same law and
normalisation as `thin_film`'s `#114` upgrade): exact at `h=h0`, so setting
`A=gamma=0` and starting from a small perturbation must reproduce the exact
$k^6$ bending-relaxation verifier below — asserted in `test_ehd_film.cpp`,
not assumed. `p_ext` is a localized applied load,

$$
p_{\mathrm{ext}}(x,y,t) =
  \begin{cases}
    p_0\exp(-r^2/2a^2) & 0\le t<t_{\mathrm{load}}\\
    0 & t\ge t_{\mathrm{load}}
  \end{cases}
$$

applied and then removed, so a run shows loading followed by
recovery/redistribution. `Pi(h)` is the same disjoining pressure as
`ehd_film`'s linear model, now with an optional `h_star` precursor form
(ported from `thin_film`) so a plate held down by adhesion settles on a
stable thin gap instead of the pressure diverging as `h -> 0`.

### Sign conventions

* `p` follows the existing linear model: flux down the pressure gradient,
  so with OpenPFC's $k_{\mathrm{lap}}=-|k|^2$ pure bending decays,
  $\lambda(k)=-M_0Bk^6<0$ (the verifier).
* `p_ext > 0` is **a load pressing the plate down onto the film** (increased
  applied pressure). A localized positive `p_ext` has
  $\nabla^2p_{\mathrm{ext}}<0$ at its centre, so
  $\partial_th=\nabla\cdot[M\nabla p]\approx M\nabla^2p_{\mathrm{ext}}<0$
  there: the gap thins under the load and thickens in the surrounding
  annulus where the fluid is displaced to (squeeze flow). Asserted directly
  in `test_ehd_film.cpp` ("localized load thins the gap...").
* Adhesion enters with the same sign as bending and tension: `-Pi(h)` in
  `p`. `A>0` is attractive at `h0`.

### Numerics

The nonlinear flux $\nabla\cdot[M(h)\nabla p]$ is evaluated with the
shared `pfc::sim::FluxETD` stepper
(`include/openpfc/kernel/simulation/spectral_flux.hpp`): a state-dependent
mobility cannot be written as a reciprocal-space symbol, so it is applied
in real space every step with Orszag 2/3 dealiasing on the flux transform.
`ehd_film_nonlinear` stays on the host; it does not instantiate a device
mobility. The existing constant-mobility `ehd_film_hip` binary is
untouched.

## Measured results

`load_relaxation_stiff.json` (`B=640`) and `load_relaxation_compliant.json`
(`B=100`), both `256²`, `gamma=10`, `A=0`, load `p0=0.5, a=8, t_load=60`,
run to `t1=600` on 4 ranks (LUMI CPU, shared allocation):

| Quantity | Stiff (`B=640`) | Compliant (`B=100`) |
|---|---|---|
| Deflection at end of loading (`t=60`) | `0.2525` | `0.3414` (35% more) |
| Spreading radius at end of loading | `14.15` | `12.30` (more localized) |
| Displaced volume at end of loading | `106.18` | `110.56` |
| Deflection at `t=600` (540 after load-off) | `0.0409` (83.8% recovered) | `0.0608` (82.2% recovered) |
| Spreading radius at `t=600` | `27.41` | `23.08` |
| Volume relative drift, whole run (max over all reported steps) | `7.0e-15` | `8.0e-15` |

The bending–tension crossover length $\ell^*=\sqrt{B/\gamma}$ is `8.0`
for the stiff case (equal to the load width `a`) and `3.16` for the
compliant case (well below it). At the load scale $k\sim 1/a$ the stiff
plate sits at the bending–tension crossover, so both terms contribute;
the compliant plate is tension-dominated. The two nonlinear cases show
similar *normalized endpoint* recovery (about 84% and 82% of the
load-release deflection by $t=600$). That is an empirical observation
for these two runs, not a demonstration that recovery is set only by
`gamma`. Compare the full trajectories with the exact linear forced
oracle (`scripts/forced_linear_oracle.py`, research `#610`) before making
a mechanism claim.

The nonlinear $h^3$ mobility reduces the load-release deflection
relative to that linear oracle (deeper thinning lowers mobility). **The
physically interpretable stiffness trend requested by `#116` remains:**
the stiffer plate deflects less and spreads the same load over a wider
area; the more compliant plate deflects more and keeps the disturbance
more localized. Both cases conserve total fluid volume to round-off,
confirming the divergence-form flux is exact regardless of the nonlinear
mobility, tension, or the time-dependent load.

## Run

```bash
mkdir -p results/ehd_film
mpirun -n 1 ./apps/ehd_film/ehd_film \
  ../apps/ehd_film/inputs_json/relaxation.json

mkdir -p results/ehd_film_nonlinear
mpirun -n 4 ./apps/ehd_film/ehd_film_nonlinear \
  ../apps/ehd_film/inputs_json/load_relaxation_stiff.json
mpirun -n 4 ./apps/ehd_film/ehd_film_nonlinear \
  ../apps/ehd_film/inputs_json/load_relaxation_compliant.json
```

`ehd_film` (verifier/demo): the shipped input is a 128² corrugation under a
stiff plate; VTK of `h` goes to `results/ehd_film/`. JSON `model.params`:
`h0`, `B`, `M0`, `gamma`, `A`, `h_star`. Initial condition
`"type": "cosine_mode"` with `h0` / `amplitude` / `nx` / `ny` / `nz`.

`ehd_film_nonlinear` (science driver): JSON `model.params` as above, plus
`load.p0` / `load.a` / `load.t_load` (Gaussian load, centred, see above)
and `diagnostics.csv` (one row per `saveat` with `time`, `h_center`,
`deflection_center`, `p_max`, `p_min`, `spreading_radius`,
`displaced_volume`, `volume`, `volume_rel_drift`). It also accepts the
session's `fields` key —
`fields: [{"name": "h", "data": "results/ehd_film_nonlinear/<case>_%04d.vti"}]`
— which writes a VTK snapshot of the gap at every `saveat`, indexed by save
and not by step (`_0000` is `t=0`). Both `load_relaxation_*.json` presets set
it; the applications catalog
renders the compliant-against-stiff
comparison from those snapshots. Omit the key and the driver writes
diagnostics only, as it did before.

## Tests

`ctest -R ehd-film` (single binary `test_ehd_film`) checks, for the linear
model: $L(k)=M_0 B k_{\mathrm{lap}}^3$, exact $\exp(-M_0 B k^6 t)$,
two-mode rate ratio 64, mean-gap conservation, and that $A>0$ grows at
low $k$; and for the nonlinear model (`#116`): the cubic mobility reduces
to `M0` at `h0`; the flux stepper reproduces the exact $k^6$ decay at
constant mobility (the point of keeping the linear verifier); volume
conservation under bending + tension + adhesion together; the load's sign
convention (thins the gap centre, conserves volume); and the load's
on/off gating at `t_load`. `#610` adds the exact forced/switch-off
bending+tension oracle (`linear_oracle.hpp`) and asserts that the
constant-mobility flux path reproduces it for a cosine load, including
mean-gap conservation and field-level agreement at load release and
during recovery. HIP builds add `HIP_EhdFilmETD` and
`ehd-film-hip-smoke` (linear model only). LUMI-G smoke: job 21799835
(`small-g`, 16², mean $h=1$).

The paper comparator is the same closed form on the sampled Gaussian
load:

```bash
python3 apps/ehd_film/scripts/forced_linear_oracle.py \
  --B 100 --out results/ehd_film_nonlinear/linear_B100.csv
python3 apps/ehd_film/scripts/forced_linear_oracle.py \
  --B 640 --out results/ehd_film_nonlinear/linear_B640.csv
```

## Deferred (not in this PR)

Per `#116`'s stretch goals and the parts of its acceptance criteria this
PR does not attempt:

* **Case C (adhesive/unstable film under a compliant plate)** — comparing a
  rigid/very-stiff and a compliant plate under a disjoining-pressure
  instability. `A>0` and the `h_star` precursor form are implemented and
  unit-tested (see Tests), but no dynamic science run drives the adhesive
  instability to a measured rupture time or wavelength; the base branch's
  experience with the `(9,3)` precursor power pair (stiff, easy to NaN once
  `h` approaches `h_star`) made this a separate, riskier piece of work than
  the load/relaxation case.
* **Energy diagnostics** (bending/capillary/wetting contributions) — not
  in this PR's measured CSV; only the mass/volume balance is reported.
* **Time-periodic loading / frequency response** and **anisotropic plate
  bending** — issue-listed stretch goals, not implemented.
* **CPU/HIP agreement for the nonlinear model** — moot for now since the
  nonlinear flux path has no HIP kernels (see "Numerics" above); the
  existing CPU/HIP agreement is for the unchanged linear model only.

## Layout

| Path | Role |
|------|------|
| `include/ehd_film/ehd_film_physics.hpp` | Linear model: `L(k)` with $k_{\mathrm{lap}}^3$, `h_star` schema |
| `include/ehd_film/ehd_film_pointwise.hpp` | $\Pi$ remainder, device-capable, now with the `h_star` precursor form |
| `include/ehd_film/ehd_film_session.hpp` | JSON session, field `h`, 2/3 dealias (linear model) |
| `include/ehd_film/nonlinear.hpp` | `#116`: cubic mobility, Gaussian load, `EhdFilmSample` diagnostics |
| `include/ehd_film/linear_oracle.hpp` | `#610`: exact forced/switch-off Fourier solution |
| `scripts/forced_linear_oracle.py` | Grid-matched Gaussian DFT trajectory for the paper |
| `src/ehd_film.cpp` / `src/hip/` | CPU / HIP `main` (linear model) |
| `src/ehd_film_nonlinear.cpp` | CPU `main`, nonlinear science driver (`#116`) |
| `inputs_json/relaxation.json` | Linear verifier/demo: bending-driven leveling |
| `inputs_json/load_relaxation_stiff.json` | Nonlinear science preset, `B=640`, `p0=0.5` |
| `inputs_json/load_relaxation_compliant.json` | Nonlinear science preset, `B=100`, `p0=0.5` |
| `inputs_json/load_relaxation_*_p005.json` | Same pair at `p0=0.05` (linear-oracle approach) |
| `inputs_json/load_relaxation_compliant_dt2.json` | `B=100`, `p0=0.5`, `dt=0.25` stability control |
