<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Thin film (`apps/thin_film`)

Lubrication **dewetting / coating** of a periodic liquid film, integrated
with spectral ETD. This is the 0.2 application for GitHub issue `#78`.

Binaries: `thin_film` (CPU); `thin_film_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on.

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
method is not. The shipped presets therefore stop while the solution is still
trustworthy, and hole *formation* is reported rather than droplet statistics.

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

## Tests

`ctest -R thin-film` checks \(\Pi'(h_0)\), the \(L(k)\) formula, volume
conservation, an unstable mode vs \(\exp(\lambda t)\), a stable high-\(k\)
mode decaying, and \(A=0\) leveling. HIP builds add `HIP_ThinFilmETD` and
`thin-film-hip-smoke`. LUMI-G smoke: job 21781449 (`small-g`, 16², mean
\(h=1\)).

## Layout

| Path | Role |
|------|------|
| `include/thin_film/thin_film_physics.hpp` | `SpectralETDPhysics` + schema |
| `include/thin_film/thin_film_pointwise.hpp` | Device-capable \(\Pi\) remainder |
| `include/thin_film/thin_film_session.hpp` | JSON session, field `h`, 2/3 dealias |
| `src/thin_film.cpp` / `src/hip/thin_film.cpp` | CPU / HIP `main` |
| `inputs_json/dewetting.json` | Unstable coating |
| `inputs_json/leveling.json` | \(A=0\) capillary smoothing |
