<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Cahn–Hilliard (`apps/cahn_hilliard`)

Conserved spinodal decomposition of a **Fe–Cr-like** alloy on a periodic
grid, integrated with spectral ETD. This is the 0.2 application for GitHub
issue `#77`. The older teaching program `examples/12_cahn_hilliard` is a
hard-coded math demo on the historical stack; use this app for JSON/TOML,
Catch2 checks, and a HIP twin when rocFFT HeFFTe is on.

Binaries: `cahn_hilliard` (CPU); `cahn_hilliard_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL` is on. Built when `OpenPFC_ENABLE_HEFFTE=ON`.

## Fe–Cr ageing: problem setup

The science case, as opposed to the compact verifiers below. This is the
`Problem setup` block the report contract asks every application to carry.

| Item | Description |
|---|---|
| **Use case** | Thermal ageing of a ferritic Fe–Cr alloy inside the 475 °C miscibility gap — the mechanism behind "475 °C embrittlement" |
| **Question** | How does the α/α′ domain length scale evolve from early spinodal amplification to late-stage coarsening, and at what rate? |
| **Domain** | 2-D periodic representative volume, 256² (early stage) or 512² (coarsening) cells at `dx = 1` code unit = **1.07 nm**, i.e. 273 nm and 547 nm per side |
| **Boundary conditions** | Periodic on both active axes |
| **Initial condition** | `seeded_noise`, mean Cr fraction ±0.005, deterministic and decomposition-independent |
| **Key parameters** | `T = 748.15 K` (475 °C); Redlich–Kister `L0 = 20500 − 9.68 T` J/mol; `Vm = 7.09e-6` m³/mol; `κ = 1e-9` J/m; `D = D0 exp(−Q/RT)` with `D0 = 2e-5` m²/s, `Q = 241` kJ/mol |
| **Observable** | Characteristic domain length `L(t) = 2π/k₁` from the azimuthally averaged structure factor, plus its growth exponent |
| **Simulated ageing** | ~1100 h at 475 °C (4000 code time units × 0.278 h) |
| **Model maturity** | numerical verification: **analytical** (linear growth rate, exact `k⁴` symbol) · physical completeness: **reduced** (no elastic misfit, no magnetic Gibbs-energy term) · calibration: **representative** — see the warning below |

### Why a periodic box

A periodic cell represents an interior volume of bulk material far from any
free surface or grain boundary. That is the right idealisation for spinodal
decomposition, which is a bulk instability with no nucleation barrier and no
preferred site. It excludes exactly what its name implies: surfaces, grain
boundaries, and any long-range stress field that would couple to the
composition. Coherent elastic misfit in particular is known to bias α/α′
morphology and is **not** in this model.

The box must be large compared with the selected wavelength or the statistics
are meaningless. At `c₀ = 0.5` the model predicts `λ_max ≈ 17.3` code units, so
512² holds about 30 wavelengths per side.

### Physical scales

The solver integrates in code units with `κ = M = 1`. Their physical meaning
follows from the energy-density scale `f₀ = RT/Vm`:

```text
length  l_c = sqrt(kappa/f0)                = 1.068 nm
time    t_c = l_c^2 |f''(c0)| / D           = 1003 s = 0.278 h   (at c0 = 0.5)
```

The time scale depends on the free-energy curvature at the alloy composition,
so it is evaluated per composition rather than once per material.

> **Warning**
> The thermodynamic and kinetic constants are **representative, not verified
> against the primary sources digit by digit**. The interaction parameter is
> the classical bcc Cr–Fe assessment of Andersson & Sundman, CALPHAD **11**
> (1987) 83–92. Treat the results as semi-quantitative and check the numbers
> before publishing anything calibrated. Every constant is a JSON parameter so
> a better assessment can be substituted without touching code.

### One consequence worth knowing

With the assessed interaction, `Fe-32Cr` at 475 °C is **outside** the spinodal —
it sits in the nucleation-and-growth regime. The app's original representative
`Omega = 20100 J/mol` is deeper and does place it inside. The science presets
therefore use `c₀ = 0.45` and `c₀ = 0.50`, which the model's own spinodal
calculation confirms are unstable. `CahnHilliardParams::spinodal` reports the
band for whatever coefficients are supplied, and a test asserts this.

### Measured results

`fe_cr_early_stage.json`, 256², 8 ranks:

| Quantity | Predicted | Measured |
|---|---|---|
| Fastest-growing wave number | `k_max = 0.3624` | `k_peak = 0.3682` (within one shell) |
| Mean Cr fraction drift | 0 | `2.1e-15` |
| Total free energy | decreasing | monotone |

`fe_cr_coarsening.json`, 512², 16 ranks, to 4000 code units ≈ 1114 h:

| `c₀` | Exponent `n` in `L ∝ tⁿ`, fitted over `t ∈ [2000, 4000]` | Final `L` |
|---|---|---|
| 0.50 | **0.353** | 39.8 nm after 1114 h |
| 0.45 | **0.371** | 36.6 nm after 943 h |

The literature comparison is the Lifshitz–Slyozov / Cahn–Hilliard result that
conserved coarsening follows `L ∝ t^{1/3}`. Both compositions agree within
about 5–11% over the late window. Fitted over the whole run the exponent drops
to 0.31, because the early interval is still wavelength selection rather than
coarsening — which is why the window is stated rather than hidden.

The three regimes the run is meant to show are all visible in the CSV: `L`
jumps from 4.0 to ~19 code units while the unstable band is selected, holds
near the selected wavelength, then grows as a power law.

## Physics

One field `c` (Cr mole fraction) with constant mobility \(M\):

\[
\partial_t c = M\nabla^2\bigl(f'(c)-\kappa\nabla^2 c\bigr).
\]

The bulk density is a regular-solution model in units of \(RT\):

\[
f(c)=\omega\,c(1-c)+c\ln c+(1-c)\ln(1-c),\qquad \omega=\Omega/(RT).
\]

Defaults \(T=748.15\,\mathrm{K}\) (475 °C) and
\(\Omega=20.1\,\mathrm{kJ\,mol^{-1}}\) put Fe–32Cr (\(c_0=0.32\)) inside the
chemical spinodal. They are a **representative miscibility-gap model**, not a
CALPHAD assessment. \(\kappa\) and \(M\) are in grid units.

In Fourier space, with OpenPFC's \(k_{\mathrm{lap}}=-|k|^2\),

\[
L(k)=M\bigl(f''(c_0)\,k_{\mathrm{lap}}-\kappa\,k_{\mathrm{lap}}^2\bigr),
\qquad
M_{\mathrm{nl}}(k)=M\,k_{\mathrm{lap}}.
\]

The real-space remainder \(n(c)=f'(c)-f'(c_0)-f''(c_0)(c-c_0)\) is the
pointwise nonlinearity. High-order stiffness is a multiply; that is the
spectral point of this app.

Linear growth of a small mode \(e^{ikx}\) is \(\lambda(k)=L(-k^2)\). Modes
with \(\lambda>0\) sit in the spinodal band
\(0<k^2<-f''(c_0)/\kappa\).

## Run

```bash
mkdir -p results/cahn_hilliard
mpirun -n 1 ./apps/cahn_hilliard/cahn_hilliard \
  ../apps/cahn_hilliard/inputs_json/fe_cr_spinodal.json
# HIP (LUMI-G / rocFFT HeFFTe): same JSON
srun ./apps/cahn_hilliard/cahn_hilliard_hip \
  ../apps/cahn_hilliard/inputs_json/fe_cr_spinodal.json
```

VTK of `c` goes to `results/cahn_hilliard/c_%04d.vti`. Open the series in
ParaView. The shipped input is a 128² slab, 20 time units, one cosine seed
in the unstable band.

JSON `model.params`: `c0`, `T` (K), `Omega` (J/mol), `R`, `kappa`, `M`.
All have defaults. Initial condition `"type": "cosine_mode"` sets
\(c=c_0+A\cos(2\pi n_x x/L_x+\cdots)\).

## Coarsening preset and diagnostics

Keep `fe_cr_spinodal.json` for single-mode growth verification. The separate
[`coarsening.json`](inputs_json/coarsening.json) seeds broadband perturbations
on a 128² grid and runs to 100 time units with `dt=0.01`. It is an illustrative
spinodal/coarsening case, not a calibrated Fe–Cr timescale. The CLI exposes it as:

```bash
./scripts/openpfc init results/spinodal --app=cahn_hilliard --preset=spinodal
```

`seeded_noise` takes `c0`, `amplitude`, and a nonnegative 64-bit integer `seed`.
It hashes global grid indices and removes the global mean using an exact
integer reduction. Initial cells are identical across MPI decompositions;
floating-point timestepping/reductions may subsequently differ by roundoff.
`amplitude` bounds `|c-c0|` (not RMS); require
`0 <= amplitude < min(c0, 1-c0)`. Noise is grid-scale: changing the grid changes
the initial condition, so use a fixed smooth IC for spatial convergence studies.

Enable time-series diagnostics with:

```json
"diagnostics": {"csv": "results/cahn_hilliard/diagnostics.csv"}
```

At `saveat`, rank zero writes
`step,time,mean,mass,min,max,bulk_energy,gradient_energy,total_energy,invalid_cells`.
Sampling includes the initial scheduled save and works without VTK writers.
The energy is evaluated from the current accepted field:

\[
F=\int [f(c)+(\kappa/2)|\nabla c|^2]\,dV
 =\int [f(c)-(\kappa/2)c\nabla^2c]\,dV.
\]

The Laplacian uses the solver's spectral wavenumbers and backend; periodicity
makes the two integral expressions equivalent. Integrals include `dx*dy*dz`,
including the slab depth. Bulk density is in units of `RT`, while lengths and
mobility remain grid units. The system `last_free_energy()` accessor remains a
bulk-only RHS diagnostic; use this CSV for current-state total energy. Optional
diagnostics cost an extra forward and inverse FFT plus host reductions per
sample (including host transfers on HIP).

Existing CSV files are never overwritten or appended. On restart choose a new
CSV path; its first row records the restored state, followed by scheduled saves.
Nonfinite cells or composition outside `0<c<1` produce an invalid count and NaN
energies, then stop the run collectively. This checks output times only; it is
not a bound-preserving integrator or an every-step stability guard. Without
diagnostics the existing logarithm evaluator's clamp remains unchanged.

Check mass drift and total-energy evolution, and repeat with smaller `dt`.
ETD here is not unconditionally energy-stable; the CSV is evidence to inspect,
not a guarantee of monotonicity at arbitrary steps or resolutions.

## Tests

`ctest -R cahn-hilliard` (or `test_cahn_hilliard`) checks the spinodal
interval, the \(L(k)\) formula, mean-\(c\) conservation, linear-mode growth
against \(\exp(\lambda t)\), and that composition variance grows while total
free energy falls for a resolved small-step case. One- and two-rank tests check
analytical gradient energy and bit-identical initial noise across decompositions.
HIP builds add `HIP_CahnHilliardETD` (CPU vs device field
to \(10^{-10}\)) and `cahn-hilliard-hip-smoke` (`SPECTRAL_CHECKSUM`).
LUMI-G smoke: job 21780712 (`small-g`, 16², mean \(c=0.32\)).

## Layout

| Path | Role |
|------|------|
| `include/cahn_hilliard/cahn_hilliard_physics.hpp` | `SpectralETDPhysics` + schema |
| `include/cahn_hilliard/cahn_hilliard_pointwise.hpp` | Device-capable \(n(c)\) and bulk \(f\) |
| `include/cahn_hilliard/cahn_hilliard_session.hpp` | JSON session, field name `c`, 2/3 dealias |
| `include/cahn_hilliard/cosine_mode.hpp` | Reproducible Fourier-mode IC |
| `src/cahn_hilliard.cpp` | CPU `main` |
| `src/hip/cahn_hilliard.cpp` | HIP `main` (`CahnHilliardHIPSession`) |
| `src/gpu/cahn_hilliard_pointwise.hip` | Device instantiation of `n(c)` |
| `inputs_json/fe_cr_spinodal.json` | Workstation 2D demo |

The physics header is templated on `MemorySpace`. A CUDA stamp of the same
`.inc` can follow the aluminum pattern if needed.
