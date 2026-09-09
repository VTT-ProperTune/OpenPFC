<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Thin film (`apps/thin_film`)

Lubrication **dewetting / coating** of a periodic liquid film, integrated
with spectral ETD. This is the 0.2 application for GitHub issue `#78`.

Binaries: `thin_film` (CPU, linear verifier); `thin_film_nonlinear` (CPU, full
`h^3` lubrication, spectral ETD, `#114`); `thin_film_fd` (CPU, the same `h^3`
lubrication, conservative face-flux FD -- integrates *through* rupture where
the two spectral solvers cannot, see "Two methods, one problem" below);
`thin_film_hip` / `thin_film_nonlinear_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on. The nonlinear flux path (`pfc::apps::SpectralFlux` / `FluxETD`) runs on
host and device -- see
[`spectral_flux.hpp`](../common/include/openpfc_apps/spectral_flux.hpp).

## Dewetting and rupture: problem setup

The science case (`#114`), as opposed to the linear verifier below. This is the
`Problem setup` block the report contract asks every application to carry.

| Item | Description |
|---|---|
| **Use case** | A thin liquid coating on a solid substrate: does it level into a uniform layer, or break up into droplets? |
| **Question** | What sets the wavelength, the rupture time and the failure location — and does a substrate defect change the answer? |
| **Domain** | 2-D periodic patch, 512² cells at `dx = 0.5`, i.e. 256 × 256 in units of the mean thickness — about 16 fastest-growing wavelengths per side |
| **Boundary conditions** | Periodic on both axes, representing an interior patch of a much larger uniform coating |
| **Initial condition** | `h = h0[1 + ε ξ(x,y) + g(x,y)]` with deterministic hashed noise `ξ`; `g` is an optional Gaussian depression for the defect case |
| **Key parameters** | `h0 = 1`, `γ = 1`, `M0 = 1`, `A = 8.6022`, precursor `h* = 0.15`; nondimensional throughout |
| **Observable** | Rupture time, minimum thickness, hole area fraction, dominant spacing, liquid volume |
| **Model maturity** | numerical verification: **analytical** (the nonlinear solver reproduces exact `k⁴` decay at constant mobility) · physical completeness: **reduced** (isothermal, no evaporation, no slip, small-slope lubrication) · calibration: **nondimensional** — the parameters are illustrative, not a specific liquid |

### Model

The verifier freezes the mobility at `M0`. The science case does not:

```text
dh/dt = div( M(h) grad p ),   p = -gamma lap h - Pi(h),   M(h) = M0 (h/h0)^3
```

The cubic factor is the physics that matters. A thinning region loses mobility
as `h³`, so drainage stalls and a depression sharpens into a hole instead of
relaxing. Constant mobility cannot produce that.

Because `M(h)` sits *inside* a divergence it cannot be written as a
reciprocal-space symbol, so this case uses the shared conservative flux stepper
[`openpfc_apps/spectral_flux.hpp`](../common/include/openpfc_apps/spectral_flux.hpp)
rather than the pointwise ETD path — four extra transforms per step in 2-D.
`SpectralFlux` / `FluxETD` are templated on `MemorySpace`, so `thin_film_nonlinear`
(host) and `thin_film_nonlinear_hip` (device) run the identical physics; the
mobility \(M(h)\) is evaluated on device with the same pointwise mechanism the
\(\Pi\) remainder uses (`OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE`, see
`src/gpu/thin_film_pointwise.inc`).

### The disjoining pressure needs a precursor

Setting `h_star > 0` switches from the bulk two-term potential to

```text
Pi = A[ (h*/h)^3 - (h*/h)^2 ]
```

which is repulsive below `h*` and attractive above it, so the film still
destabilises at `h0` but a rupturing hole drains to a stable precursor instead
of to zero. The exponents are 3 and 2, not the 9 and 3 of the bulk form, and
that is a numerical choice as much as a physical one: a ninth-power repulsion
calibrated to the same `Pi'(h0)` reaches `Pi ~ 1e6` by `h = 0.05 h0`, so one
cell dipping there produces a NaN on the next step **regardless of timestep**.
The cubic pair is some four thousand times softer and integrates through hole
formation.

### Measured results

512², `dx = 0.5`, `dt = 0.002`, 16 ranks. "Reaches precursor" is the first time
`min h` falls below `h* = 0.15`.

| Case | Reaches precursor | Final `min h` | Hole area fraction | Dominant spacing | Volume drift |
|---|---|---|---|---|---|
| Spontaneous (noise only) | **t = 140** | 0.135 | 1.82 % | 14.2 | 1.3e-15 |
| Defect-triggered | **t = 85** | 0.145 | 0.33 % | 25.6 | 3.3e-16 |

The linear theory predicts a fastest-growing wavelength of 16.2; the measured
spacing of 14.2 for the spontaneous case is consistent with it once finite
amplitude and the structure-factor bin width are allowed for.

The engineering result is the comparison: a single 30 % deep depression brings
failure forward by about 40 %, and the film fails **at the defect** rather than
at the wavelength the instability would have chosen — the defect case's 25.6
spacing reflects one isolated hole, not a pattern.

Volume is conserved to round-off in both, which is the check that the
divergence form is being integrated honestly. The defect case starts from a
lower volume because the depression removes liquid; that is its initial
condition, not a loss.

### Known limitation: integrating *through* rupture

These runs stop where they do for a reason. Once `min h` falls to roughly
`0.6 h*` the solver diverges, and **neither refining the timestep by 5× nor the
grid by 2× moves the point of breakdown**. That is the signature of a
structural limitation rather than an accuracy one:

* the mobility is *degenerate* — `M -> 0` as `h -> 0`, so the equation loses
  parabolicity exactly where the interesting event happens;
* the scheme has no positivity preservation, so a cell can overshoot to
  `h <= 0`, at which point the potential is evaluated at a clamp;
* an explicit remainder cannot cope with the stiff repulsion in the precursor
  once the two effects combine.

Integrating through rupture and into late-stage droplet coarsening needs a
positivity-preserving or entropy-dissipating scheme, which a spectral ETD1
method is not. The shipped spectral presets therefore stop while the solution
is still trustworthy, and hole *formation* is reported rather than droplet
statistics -- **that scheme** is `thin_film_nonlinear`. A different scheme,
`thin_film_fd`, does not stop here; see the next section.

## Two methods, one problem (`#124`)

The insight this application is built to demonstrate: **the same degenerate
mobility that breaks the spectral scheme is what makes a conservative
finite-volume scheme well behaved.** `M(h) -> 0` as `h -> 0` is fatal for a
pointwise/spectral treatment (the equation loses parabolicity exactly where
the interesting event happens) but is exactly the property a face-centred
flux needs: if the mobility at a face is formed from the mobilities of the
two cells it separates, the flux out of a vanishing cell vanishes with it,
and the cell cannot be driven negative. Two solvers, same equation:

| | `thin_film_nonlinear` (spectral ETD1) | `thin_film_fd` (conservative FD) |
|---|---|---|
| Flux | pointwise `M(h) grad p` in real space, transformed as a whole | formed *at cell faces*, `M_face (p_R - p_L)/dx` |
| Mass conservation | exact in the continuum (`k=0` mode untouched); not face-by-face | **exact to round-off, for any `dt`**, by telescoping (Test: `FD flux divergence sums to zero...`) |
| Positivity | none -- a cell can be driven to `h <= 0` | preserved in practice with a harmonic-mean face mobility (measured below) |
| Good for | the smooth pre-rupture regime: rupture *time*, dominant wavelength, defect-vs-spontaneous comparison, at spectral accuracy and one FFT-pencil decomposition | rupture *through* to hole formation and rim/droplet coarsening; anywhere the film gets thin |
| Stops at | `min h ~ 0.6 h*` (diverges) | does not stop; pins at the precursor and keeps integrating |
| Parallelism | HeFFTe pencil decomposition | `pfc::decomposition` Cartesian grid + `pfc::comm::SparseExchange` halo exchange (same pattern as `apps/allen_cahn`) |

### Problem setup: the FD case

Same equation, same domain, same initial-condition family as the spectral
science case above -- the *only* things that change are the discretization
and (necessarily) the timestep and run length, since this solver is the one
that does not have to stop at `min h ~ 0.6 h*`.

| Item | Description |
|---|---|
| **Use case** | Same dewetting film as above, continued *through* rupture: hole formation, rim growth, and coarsening -- the regime `thin_film_nonlinear` cannot reach |
| **Domain** | 2-D periodic, 512² cells at `dx = 0.5`, identical to the spectral case |
| **Boundary conditions** | Periodic on both axes, via `pfc::comm::SparseExchange` (separated halo layout, `apps/allen_cahn`-style) |
| **Initial condition** | Same hashed-noise / Gaussian-defect family as `thin_film_nonlinear`, byte-identical when the same seed is used (verified: t=0 `min_h`/`max_h`/volume match to 15 significant digits between the two binaries) |
| **Key parameters** | Same physical parameters, plus `"fd": {"face_mobility": "harmonic"\|"arithmetic", "order": 2\|4}` |
| **Observable** | Everything `thin_film_nonlinear` reports, plus a connected-component hole count (`n_holes`, rank-0 gather + flood fill) |
| **Model maturity** | numerical verification: **analytical** (reproduces the exact `k^4` capillary decay and the linearized unstable growth rate `M0 k^2(Pi'(h0) - gamma k^2)`, both to 5 %) · physical completeness: same reduced lubrication model as the spectral case · calibration: nondimensional, same as the spectral case; face-mobility choice verified empirically (below), not assumed |

### Face mobility: harmonic beats arithmetic, measured

Two symmetric averages of the two cells' mobilities at a face are provided.
Driving the same precursor case (`A = 8.6022`, `h* = 0.15`, a deep localized
defect) with each, on identical grids:

| Face average | Behaviour |
|---|---|
| **Harmonic**, `2 M_L M_R / (M_L + M_R)` | `min h` settles at `0.151`-`0.157`, just above `h*`, and **stays there** for as long as the run continues (checked to `t = 500`, `5*10^5` steps) |
| **Arithmetic**, `(M_L + M_R)/2` | identical up to `min h ~ 0.33`, then overflows to `inf`/`NaN` within about 5 time units, right as the thinnest cell approaches `h*` |

The mechanism is exactly what the harmonic mean is for: it is zero whenever
*either* side of a face is dry, so the flux out of a thinning cell shuts off
as that cell empties. The arithmetic mean only halves the flux in the same
situation -- not enough to stop a cell from being pulled through zero once
the destabilizing `Pi'(h)` term is strong enough, which is exactly the
regime a rupture study needs. `thin_film_fd` defaults to harmonic;
`"face_mobility": "arithmetic"` is kept only to reproduce this comparison
(`FD face mobility: harmonic mean...` test).

### Stable timestep

The curvature operator inside `p` is a discrete Laplacian, so it inherits a
`dt <~ dx^4` explicit stability limit -- a real cost, not a rounding
footnote. Measured on the 512² / `dx = 0.5` case (order-2 curvature
operator, harmonic face mobility): **`dt = 0.001` runs the whole way through
rupture with no sign of instability; `dt = 0.002`-`0.0025` already show
spurious excess thinning inside 5 time units that is not present at smaller
`dt` (a numerical, not physical, effect); `dt >= 0.003` overflows within a
few hundred steps.** `thin_film_fd`'s shipped cases use `dt = 0.001`.

### Curvature-operator order: 2 vs 4

`"fd": {"order": 4}` swaps the 5-point curvature Laplacian for the 9-point
one (`pfc::field::fd::laplacian2d_xy_periodic_separated<4>`); the face flux
itself is unchanged (still the natural 2-point difference -- that is what
"flux at a face" means). On the 64² / `dx = 0.5` probe case, same `dt`
(`5*10^-4`, stable for both), same 250 time units: order 2 gives
`min h = 0.7171`, order 4 gives `min h = 0.7248`, about a 1 % difference at
this resolution -- consistent with the 4th-order operator's smaller
truncation error, not a change in the physics. The shipped cases use order 2
because it is cheaper and the accuracy difference at this `dx` is minor;
order 4 is there for anyone who wants to check that the answer is not
resolution-dependent in the curvature operator specifically.

### Measured agreement: the smooth pre-rupture regime

**Controlled, single-mode comparison** (the rigorous check: a deterministic
cosine perturbation near `k_peak`, not noise, both solvers built from the
same physics module, `FD and spectral agree in the smooth pre-rupture
regime` test): after 100 steps of exponential growth from the same initial
condition, `thin_film_fd`'s `min h` agrees with the spectral run's to 2 %,
and volume agrees to `1e-6` relative -- both still far from the precursor.
This is the comparison that isolates discretization error from everything
else, and it is where "the two methods agree" is unambiguously true.

**The actual 512², broadband-noise flagship case** is a different, harder
comparison, and it is reported here honestly rather than only where it is
flattering. The two binaries start from a *bit-identical* initial condition
(`thin_film_dewetting.json` and `thin_film_fd_dewetting.json` differ only in
`t1`/`dt`/the `"fd"` block; `t = 0` `min_h`/`max_h`/volume match to 15
significant digits). They do **not** track each other closely afterwards:

| | Spectral (`thin_film_nonlinear`) | FD (`thin_film_fd`) |
|---|---|---|
| Crosses/approaches `h*` at | `t = 140` (crosses, keeps falling) | `t ~ 340`-`400` (approaches asymptotically, does not cross) |
| `min h` at `t = 100` | `0.705` | `0.990` |

That is a real, reproducible timing offset, not noise -- rerunning either
side changes nothing (deterministic hashed noise). The likely mechanism:
spectral ETD treats every wavenumber exactly and applies Orszag 2/3
dealiasing, which removes the top third of wavenumbers outright; the FD
scheme's 2nd-order discrete Laplacian damps near-Nyquist content at a
*finite* rate that is only `(2 - 2cos(k dx))/(dx^2 k^2)` of the continuum
value -- at this `dx = 0.5`, that ratio is `0.41` at the Nyquist wavenumber,
so the biharmonic damping of single-cell noise is about `6x` weaker than the
continuum/spectral value. The broadband hashed-noise initial condition is
dominated by exactly that near-Nyquist content (it is spatially white), so
the two schemes clear the noise floor at genuinely different rates before
the coherent dewetting pattern takes over -- a real, explainable structural
difference in how each method treats initial noise, not a bug in either
one. The single-mode comparison above is the one to trust for "do the two
methods agree"; the broadband comparison is the one to trust for "how long
until this specific noisy run rewards patience."

### Measured results: through rupture and into coarsening

512², `dx = 0.5`, `dt = 0.001`, harmonic face mobility, order-2 curvature,
single rank (see "Run distributed" below for the MPI check). "Reaches
precursor" here means "approaches to within 1 % of `h* = 0.15` and stays
there" -- the FD solver does not cross below `h*`, it is *pinned* there, in
contrast to the spectral runs above, which cross it and then diverge.

| Case | Approaches precursor | Min `h` reached | Hole area fraction (final) | `n_holes` peak -> final | Volume drift |
|---|---|---|---|---|---|
| Spontaneous (`t1=400`) | `t ~ 340` | 0.1515 | 19.6 % | 227 (`t=345`) -> 204 (`t=400`) | `4.3e-11` abs (`1.3e-15` rel) |
| Defect-triggered (`t1=300`) | not yet reached (`min h = 0.164` at `t=300`) | 0.164 | 1.3 % | 1 -> 1 | `1.5e-10` abs (`4.6e-15` rel) |

The spontaneous case is the headline result: **it forms holes, keeps them
positive, conserves volume to round-off, and its hole count falls from 227
to 204 between `t=345` and `t=400`** -- coalescence, i.e. coarsening, is
directly observed, not just hole *formation*. The rim collects the drained
liquid (`max h` rises past `1.4`, up from `1.01` at `t=0`). The
defect-triggered case is slower to reach the precursor here than in the
spectral run (which reached it by `t=85`) for the same reason the broadband
comparison above diverges in timing; it has a single, isolated hole
throughout, matching the spectral run's qualitative finding that a defect
produces one hole rather than a pattern.

Both cases were run as a single `sbatch` job on `standard-g`
(`apps/thin_film/slurm/fd_flagship.sbatch`, 16 MPI ranks, one node,
CPU-only); the spectral comparisons above were run interactively (16 ranks,
shared allocation) since they are the same, previously-characterized
512²/`dt=0.002` cases the existing presets already ship.

### Run distributed: MPI correctness check

`thin_film_fd` decomposes the domain with `pfc::decomposition::create` and
exchanges halos with `pfc::comm::SparseExchange`, the same building blocks
as `apps/allen_cahn`. Checked directly: the same case
(`thin_film_dewetting.json`-style, 5 time units) run at 1 rank and 4 ranks
gives **bit-identical `min h` and `max h`** at every save point; `volume`
and `mean h` differ only at the `1e-13` relative level expected from a
different `MPI_Allreduce` summation order. The physics does not depend on
the decomposition; only floating-point summation order does.

## Physics

One field `h` (film thickness) with constant mobility \(M_0\):

\[
\partial_t h=\nabla\cdot\bigl[M_0\nabla p\bigr],\qquad
p=-\gamma\nabla^2 h-\Pi(h).
\]

Flow is down the pressure gradient, so capillary \(\nabla^4\) *damps* short
waves. (A leading minus on the divergence with this \(p\) would reverse that
and is not used.) Cubic mobility \(M\propto h^3\) is a flux nonlinearity; this
slice uses \(M_0=M(h_0)\), which is exact for the linear band.

Disjoining pressure (mean thickness \(h_0\)):

\[
\Pi(h)=A\bigl((h_0/h)^3-(h_0/h)^9\bigr)
\]

(van der Waals attraction plus short-range repulsion). \(A=0\) is a leveling
coating.

In Fourier space, with OpenPFC \(k_{\mathrm{lap}}=-|k|^2\),

\[
L(k)=-M_0\gamma\,k_{\mathrm{lap}}^2-M_0\Pi'(h_0)\,k_{\mathrm{lap}}.
\]

Linear growth:

\[
\lambda(k)=M_0 k^2\bigl(\Pi'(h_0)-\gamma k^2\bigr).
\]

The fastest-growing mode is \(k_{\mathrm{peak}}^2=\Pi'(h_0)/(2\gamma)\) when
\(\Pi'(h_0)>0\). Capillary \(k^4\) is a multiply.

## Run

```bash
mkdir -p results/thin_film
mpirun -n 1 ./apps/thin_film/thin_film \
  ../apps/thin_film/inputs_json/dewetting.json
mpirun -n 1 ./apps/thin_film/thin_film \
  ../apps/thin_film/inputs_json/leveling.json
```

Dewetting: 128², \(A=0.05\), cosine near \(k_{\mathrm{peak}}\). Leveling:
\(A=0\), roughness decays. VTK of `h` under `results/thin_film/`.

JSON `model.params`: `h0`, `gamma`, `M0`, `A`. Initial condition
`"type": "cosine_mode"` with `h0` / `amplitude` / `nx` / `ny` / `nz`.

The `h^3` lubrication science cases (`#114`, `#124`) take a different JSON
shape (`domain.Lx`/`Ly`/`Lz`/`dx`, `model.params` adds `h_star`,
`timestepping.t1`/`dt`/`saveat`, `initial_conditions.amplitude`/`seed`/
`defect_amplitude`/`defect_sigma`, `diagnostics.csv`) and the same case file
drives *any* of the three nonlinear binaries:

```bash
mpirun -n 16 ./apps/thin_film/thin_film_nonlinear \
  ../apps/thin_film/inputs_json/thin_film_dewetting.json   # spectral, stops before rupture
./apps/thin_film/thin_film_nonlinear_hip \
  ../apps/thin_film/inputs_json/thin_film_dewetting.json   # same JSON, same observables, on device
mpirun -n 16 ./apps/thin_film/thin_film_fd \
  ../apps/thin_film/inputs_json/thin_film_fd_dewetting.json  # FD, continues through it
```

`thin_film_fd`'s cases add an optional `"fd": {"face_mobility":
"harmonic"|"arithmetic", "order": 2|4}` block (defaults: harmonic, 2). See
"Two methods, one problem" above for what those knobs do and the measured
comparison. `apps/thin_film/slurm/fd_flagship.sbatch` reproduces the
512²-through-rupture runs on LUMI's `standard-g`.

## Tests

`ctest -R thin-film` checks \(\Pi'(h_0)\), the \(L(k)\) formula, volume
conservation, an unstable mode vs \(\exp(\lambda t)\), a stable high-\(k\)
mode decaying, and \(A=0\) leveling, plus the nonlinear/flux and FD suites
(`[nonlinear]`, `[fd]` tags): cubic mobility, the flux stepper's exact
\(k^4\) decay, a Gaussian defect, and for the FD solver -- exact mass
conservation via `compute_rhs`, the linearized \(k^4\) decay and unstable
growth rate, harmonic-vs-arithmetic positivity through a driven rupture, and
FD-vs-spectral agreement on a controlled single-mode case. HIP builds add
`HIP_ThinFilmETD` (constant-mobility session parity) and a nonlinear-flux
HIP-vs-host parity case (`[thin_film][hip][nonlinear]`), plus
`thin-film-hip-smoke`. LUMI-G smoke: job 21781449 (`small-g`, 16², mean
\(h=1\)).

## Layout

| Path | Role |
|------|------|
| `include/thin_film/thin_film_physics.hpp` | `SpectralETDPhysics` + schema |
| `include/thin_film/thin_film_pointwise.hpp` | Device-capable \(\Pi\) remainder |
| `include/thin_film/thin_film_session.hpp` | JSON session, field `h`, 2/3 dealias |
| `include/thin_film/nonlinear.hpp` | `CubicMobility`, `PotentialPointwise`, `sample_film`, `GaussianDefect` (shared by all three nonlinear solvers) |
| `include/thin_film/nonlinear_driver.hpp` | `MemorySpace`-templated `run_thin_film_nonlinear` (host and device) |
| `include/thin_film/fd_flux.hpp` | Conservative face-flux FD solver: `FDFluxSolver`, `FaceMobility`, `sample_film_fd`, `count_dry_regions_rank0` |
| `src/thin_film.cpp` / `src/hip/thin_film.cpp` | CPU / HIP `main` (linear verifier) |
| `src/thin_film_nonlinear.cpp` / `src/hip/thin_film_nonlinear.cpp` | CPU / HIP `main`, spectral `h^3` lubrication |
| `src/thin_film_fd.cpp` | CPU `main`, conservative FD `h^3` lubrication |
| `src/gpu/thin_film_pointwise.{hip,inc}` | Device instantiations: \(\Pi\) remainder, \(\Pi(h)\), \(M(h)\cdot\nabla p\) |
| `inputs_json/dewetting.json` | Unstable coating (linear verifier) |
| `inputs_json/leveling.json` | \(A=0\) capillary smoothing (linear verifier) |
| `inputs_json/thin_film_dewetting.json` / `thin_film_fd_dewetting.json` | Spontaneous dewetting, spectral / FD |
| `inputs_json/thin_film_defect.json` / `thin_film_fd_defect.json` | Defect-triggered dewetting, spectral / FD |
| `inputs_json/thin_film_dewetting_heroic.json` | Heroic-scale device run |
| `slurm/fd_flagship.sbatch` | Reproduces the 512² FD through-rupture runs on `standard-g` |
