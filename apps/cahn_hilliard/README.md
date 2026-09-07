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
