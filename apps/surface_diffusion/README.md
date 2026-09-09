<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Surface diffusion (`apps/surface_diffusion`)

Mullins **small-slope surface diffusion**: thermal smoothing of nanoscale
roughness, and (`#115`) orientation-dependent relaxation of a patterned
nanosurface under an anisotropic surface stiffness. This is the 0.2
application for GitHub issues `#79` and `#115`.

Binaries: `surface_diffusion` (CPU, isotropic verifier);
`surface_diffusion_anisotropic` (CPU, anisotropic science driver);
`surface_diffusion_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL` is on (isotropic
only -- see [Anisotropic model](#anisotropic-model-115) for why the
anisotropic driver is CPU-only).

## Problem setup

| Item | Verification preset (`surface_diffusion`) | Science preset (`surface_diffusion_anisotropic`) |
|---|---|---|
| Use case | Numerical oracle: exact single/two-mode `k^4` decay | Annealing/relaxation of a patterned nanoscale surface (e.g. a lithographically ruled or deposited crossed ridge pattern) under isotropic vs anisotropic surface diffusion |
| Domain | 2D periodic slab, e.g. \(128\times128\) grid units | 2D periodic slab, \(128\times128\) grid units, \(dx=1\) |
| Grid | 16²-32² (tests), 128² (demo), single Fourier mode(s) | 128², resolving 16 corrugation periods per axis (8 grid points/period) |
| Boundary conditions | Periodic on x/y (spectral); represents an infinite/repeated flat surface patch | Periodic on x/y; represents an interior patch of a repeated nanoscale surface pattern, not a finite chip/facet edge |
| Initial condition | One or two cosine modes, amplitude \(\le 0.1\) | Crossed sinusoidal corrugation \(h=h_0+A[\cos(2\pi n_x x/L_x)+\cos(2\pi n_y y/L_y)]\), \(A=0.05\), \(n_x=n_y=16\); same field run isotropically (`eps_a=0`) and anisotropically (`eps_a=0.5`, `m=6`) |
| Key physical parameters | `B` (Mullins coefficient, grid units, unfitted) | `B0` (grid units, unfitted), `eps_a` (anisotropy strength, illustrative -- see `anisotropy.hpp`), `m` (stiffness symmetry order, 4 or 6) |
| Observable | Mode amplitude vs \(\exp(-Bk^4t)\); two-mode rate ratio \((k_2/k_1)^4\) | RMS roughness, structure-factor dominant wavelength, directional spectral energy (kx- vs ky-dominated modes), max \(|\nabla h|\), mean height (conserved) |
| Model maturity | Numerical verification: **analytical** (exact `k^4` decay). Physical completeness: **reduced** (isotropic small-slope Mullins only). Calibration: **none** (grid units) | Numerical verification: **regression** against the isotropic verifier at `eps_a=0` and against an exact single-orientation decay rate (see Tests). Physical completeness: **reduced** (small-slope height-function model; orientation-dependent *kinetic* coefficient imposed directly, not a Herring `gamma+gamma''` stiffness derived from a fitted `gamma(theta)`). Calibration: **none** -- `B0`, `eps_a`, `m` are illustrative, not fitted to a measured material `gamma(theta)` |

## Physics

For a small-slope, isotropic free surface the height \(h\) obeys Mullins
surface diffusion. The chain is:

1. Mean curvature is \(\kappa\approx-\nabla^2 h\).
2. Chemical potential is \(\mu\propto\gamma\kappa\), so \(\mu\propto-\nabla^2 h\).
3. Atoms hop from high to low \(\mu\); the surface flux is
   \(j\propto-\nabla\mu\propto\nabla(\nabla^2 h)\).
4. Mass conservation on the height field is \(\partial_t h=-\nabla\cdot j\),
   which produces the biharmonic

\[
\partial_t h = -B\nabla^4 h.
\]

\(B\) lumps surface diffusivity, surface energy, atomic volume and temperature
(grid units here). Every Fourier mode decays independently:

\[
h_k(t)=h_k(0)\exp(-B|k|^4 t).
\]

Short wavelengths therefore disappear much faster than long ones — the
annealing / roughness-stability story. With OpenPFC
\(k_{\mathrm{lap}}=-|k|^2\),

\[
L(k)=-B\,k_{\mathrm{lap}}^2.
\]

There is no real-space nonlinearity.

## Anisotropic model (`#115`)

Real crystalline surfaces do not relax isotropically: the surface free energy
\(\gamma\) depends on the local surface-normal orientation, and by the
Herring relation the *kinetic* stiffness that controls diffusive smoothing is
\(\tilde\gamma(\theta)=\gamma(\theta)+\gamma''(\theta)\) (Mullins 1957;
Rettori & Villain, *J. Phys. France* 49, 257 (1988); Bonzel & Preuss, *Surf.
Sci.* 336, 209 (1995)). This code does **not** differentiate a specific
literature \(\gamma(\theta)\) to obtain \(\tilde\gamma\); it imposes the same
periodic *functional form* directly on the kinetic coefficient,

\[
B(\theta)=B_0\bigl[1+\epsilon_a\cos(m\theta)\bigr],\qquad
\theta=\operatorname{atan2}(h_y,h_x),
\]

with \(\theta\) the local surface-*gradient* orientation and \(m\in\{4,6\}\)
the crystal symmetry order, evaluated spectrally (\(h_x,h_y\) are computed
from `i*k_x*h_hat` / `i*k_y*h_hat`, not a finite-difference stencil). The
governing equation generalises the isotropic divergence form:

\[
\partial_t h=\nabla\cdot\bigl[B(\theta)\,\nabla(\nabla^2 h)\bigr]
\quad\xrightarrow{\ B(\theta)\to B_0\ }\quad
\partial_t h=-B_0\nabla^4 h.
\]

**This is still a small-slope, height-function model.** \(\theta\) is the
orientation of the local surface gradient of a single-valued height field; it
does **not** claim arbitrary-slope, overhanging, or faceted-plane crystalline
surface evolution. \((B_0,\epsilon_a,m)\) are illustrative parameters, not
fitted to any specific material's measured \(\gamma(\theta)\) -- do not read
\(\epsilon_a\) as a calibrated anisotropy strength.

Because \(B(\theta)\) depends on the field's own gradient, the operator is
not a reciprocal-space symbol; `surface_diffusion_anisotropic` evaluates it
with a self-contained spectral-flux stepper
(`include/surface_diffusion/anisotropic_flux.hpp`) that integrates the
constant-\(B_0\) part exactly (ETD) and treats the orientation-dependent
remainder explicitly, the same splitting `openpfc_apps/spectral_flux.hpp`
uses for `thin_film`'s \(h^3\) mobility. At `eps_a=0` the remainder is zero
(up to FFT round-off) and the stepper reproduces the isotropic verifier's
exact \(\exp(-B_0k^4t)\) trajectory -- checked in `tests/test_surface_diffusion.cpp`,
not assumed.

**Why `m=6` for the shipped science preset, not `m=4`.** The crossed
corrugation's two ridge sets sit at \(\theta=0\) (x-ridges) and
\(\theta=\pi/2\) (y-ridges). A fourfold stiffness treats those as *the same*
orientation (\(\cos(4\cdot0)=\cos(4\cdot\pi/2)\)), so it would not show a
directional effect on this particular initial condition; a sixfold stiffness
does not (\(\cos(6\cdot0)=1\neq\cos(6\cdot\pi/2)=-1\)). `m=4` is fully
implemented and unit-tested (the fourfold symmetry of \(B(\theta)\) itself),
it is simply not the orientation-selection demonstration case here.

## Run

```bash
mkdir -p results/surface_diffusion
mpirun -n 1 ./apps/surface_diffusion/surface_diffusion \
  ../apps/surface_diffusion/inputs_json/smoothing.json

# Anisotropic science preset (#115): same crossed-corrugation initial
# surface, run once isotropically and once anisotropically.
mkdir -p results/surface_diffusion
mpirun -n 1 ./apps/surface_diffusion/surface_diffusion_anisotropic \
  ../apps/surface_diffusion/inputs_json/nanosurface_isotropic.json
mpirun -n 1 ./apps/surface_diffusion/surface_diffusion_anisotropic \
  ../apps/surface_diffusion/inputs_json/nanosurface_anisotropic.json
```

The shipped input is a 128² slab with three cosine wavelengths. VTK of `h`
goes to `results/surface_diffusion/`. Short modes flatten first.

JSON `model.params`: `B` (default 1). Initial condition `"type":
"cosine_mode"` with `h0` and either a single `amplitude`/`nx`/`ny`/`nz` or a
`modes` array.

`surface_diffusion_anisotropic` reads a different, simpler JSON shape (it is
a standalone driver, not a `SpectralETDSession`): `model.params` is `B0`,
`eps_a`, `m`; `domain` requires `Lz: 1`; `initial_conditions` is `{h0,
modes:[{nx,ny,amplitude}, ...]}` (each mode a plane cosine, summed); optional
`diagnostics.csv` writes the observables table below every `saveat`.

## Measured orientation selection (LUMI, CPU, 2026-09-09)

Both presets start from the identical crossed corrugation (\(n_x=n_y=16\),
\(A=0.05\), RMS roughness `0.05`, `max|grad h|=0.0555`, `energy_kx_frac =
energy_ky_frac = 0.5`) and run to `t1=8` (`dt=0.01`, 800 steps).

| Observable at `t=8` | Isotropic (`eps_a=0`) | Anisotropic (`eps_a=0.5`, `m=6`) |
|---|---|---|
| RMS roughness | 0.002382 | 0.001549 |
| max\(\vert\nabla h\vert\) | 0.002646 | 0.001704 |
| `energy_kx_frac` / `energy_ky_frac` | 0.500 / 0.500 | 0.249 / 0.751 |
| mean height | \(-1.4\times10^{-17}\) | \(-1.2\times10^{-18}\) |

The isotropic run holds `energy_kx_frac = energy_ky_frac = 0.500` for every
sample in the run (linear dynamics: `|k|` sets the rate, orientation does
not) -- a useful internal check as well as a physical statement. The
anisotropic run's split moves monotonically away from 1:1 at every sample
(`0.500 -> 0.486 -> ... -> 0.249/0.751` at `t=0,0.5,...,8`; see
`results/surface_diffusion/nanosurface_anisotropic.csv`), i.e. the softer
(`theta=pi/2`, y) orientation increasingly dominates the surviving spectral
content as the stiffer (`theta=0`, x) orientation is preferentially damped --
the orientation-selection effect `#115` asks for, reproduced from a
deterministic initial condition. `dominant_wavelength` from the (orientation-
blind) structure factor is unchanged between the two runs (`7.758`), which is
expected and is exactly why the directional split, not the structure factor
alone, is the observable that shows the anisotropy.

Caveat measured directly from this run: a short single-orientation unit test
(`tests/test_surface_diffusion.cpp`) shows the *isolated* stiff/soft decay
rates differing by a factor of 3 for these `(B0,eps_a,m)`, but the *combined*
crossed-corrugation run above shows a smaller (order-of-magnitude smaller at
short times) net effect, because each orientation's local `B(theta)` is set
by the combined gradient of both ridge sets, not by either mode alone -- the
two orientations are nonlinearly coupled once both are present. The 3:1
final energy split above is the honest, measured, coupled-system number; do
not read the isolated single-mode ratio as a prediction for a mixed pattern.

## Mode amplitude versus time

`smoothing.json` writes `h_%04d.vti`. Open the series in ParaView as a time
sequence: the \((n_x,n_y)=(16,8)\) ripple is gone well before \((2,1)\).

To plot amplitude versus time, take the same inner product as
`tests/test_surface_diffusion.cpp` on each snapshot: for a mode
\((n_x,n_y)\) with wavevector \(k=2\pi(n_x/L_x,n_y/L_y)\),

\[
A(t)=\frac{\sum h\,w}{\sum w^2},\qquad
w=\cos(k_x x+k_y y),
\]

and compare to \(A(0)\exp(-B|k|^4 t)\). Catch2 already checks a single mode
against that exponential and a two-mode rate ratio of 16
(\((k_2/k_1)^4\) with \(n_x=2\) vs \(n_x=1\)). Mean height is conserved.

## Tests

`ctest -R surface-diffusion` checks:

- (verifier, unchanged by `#115`) \(L(k)=-B k_{\mathrm{lap}}^2\), exact
  single-mode \(\exp(-B k^4 t)\), two-mode decay rates in the ratio
  \((k_2/k_1)^4=16\), and mean-height conservation;
- (`#115` anisotropic model) `B(theta)` reduces exactly to `B0` at
  `eps_a=0`; `B(theta)` has the fourfold/sixfold symmetry it claims (and that
  `m=4` makes `theta=0`/`theta=pi/2` equivalent while `m=6` does not);
  `AnisotropicSurfaceDiffusionETD` reproduces the isotropic `exp(-B0k^4t)`
  decay at `eps_a=0`; a single-orientation run's decay rate matches
  `B(theta)*k^4` exactly for both the stiff and soft orientations under a
  sixfold anisotropy; mean height is conserved during an anisotropic run;
  a crossed-corrugation run shows equal x/y amplitude decay isotropically
  and unequal decay anisotropically (the faceting/orientation-selection
  claim, made quantitative).

HIP builds add `HIP_SurfaceDiffusionETD` and `surface-diffusion-hip-smoke`
(isotropic verifier only -- the anisotropic driver is CPU-only). LUMI-G
smoke: job 21791182 (`small-g`, 16², mean \(h=0\)).

## Layout

| Path | Role |
|------|------|
| `include/surface_diffusion/surface_diffusion_physics.hpp` | Isotropic `L(k)=-B k_{\mathrm{lap}}^2` (verifier) |
| `include/surface_diffusion/surface_diffusion_pointwise.hpp` | Zero remainder (verifier) |
| `include/surface_diffusion/surface_diffusion_session.hpp` | JSON session, field `h` (verifier) |
| `include/surface_diffusion/anisotropy.hpp` | `B(theta) = B0[1+eps_a cos(m theta)]` + JSON schema (`#115`) |
| `include/surface_diffusion/anisotropic_flux.hpp` | Self-contained spectral-flux ETD1 stepper for `div[B(theta) grad(lap h)]` (`#115`) |
| `src/surface_diffusion.cpp` / `src/hip/` | CPU / HIP `main` (verifier) |
| `src/surface_diffusion_anisotropic.cpp` | CPU science driver: crossed corrugation, CSV diagnostics (`#115`) |
| `inputs_json/smoothing.json` | Multi-wavelength annealing demo (verifier) |
| `inputs_json/nanosurface_isotropic.json` / `nanosurface_anisotropic.json` | Same crossed-corrugation IC, `eps_a=0` vs `eps_a=0.5` (`#115`) |
