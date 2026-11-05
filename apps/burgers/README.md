<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Burgers (`apps/burgers`)

One periodic viscous Burgers problem, advanced by three selectable
methods. The flux and the Riemann solver stay in this application.
There is no new generic conservation-law framework.
The CPU FDM2/FVM implementation and its tests remain buildable without
optional FFTW3. Spectral methods require FFTW3 and otherwise fail explicitly.

## Contract

The PDE, the initial field and the comparison rules below were fixed
before reading a campaign.

```text
∂t u + ∂x(u²/2) = ν ∂xx u,    x ∈ [0, 2π),    u(x,0) = sin(x).
```

For `ν = 0` the first shock forms at `t = 1`, `x = π`. Before that,
the characteristic solution is `u = sin(ξ)` with `x = ξ + t sin(ξ)`.
After that, oddness about `x = π` keeps the shock there. The offset
`δ = t sin(δ)` cuts the multi-valued interval, and the retained
pre-image on `[0, π]` is `[0, π - δ]`. The point value on the shock
is recorded as `0`.

A viscous front is a smooth field. The reference is Cole–Hopf,
independent of the stepper:

```text
φ(x,0) = exp(cos(x) / (2ν)),
φ(x,t) = I_0(z) + 2 Σ I_n(z) exp(-ν n² t) cos(n x),    z = 1/(2ν),
u = -2ν φ_x / φ.
```

The smooth energy balance is `dE/dt = -ν ∫ (u_x)² dx` with
`E = ½ ∫ u² dx`. The test observes `u_x` with `fd::evaluate<2>` on
the evolved field. It does not ask the stepper to report its own
residual. In the finite-volume representation, the reported discrete
`E_h = ½ Δx Σ_i (cell_average_i)²` is **not** the exact cell-integrated
quadratic energy: Jensen's inequality makes it lower already at t=0.
Do not interpret its offset from nodal energy as physical/numerical
dissipation without first accounting for the observation map.

### Representations

Nodal methods collocate at `x_i = i Δx`, `Δx = 2π/N`. The
finite-volume unknown is the average on `[i Δx, (i+1) Δx)`. The sine
wave's cell average is `(cos(x_i) - cos(x_{i+1})) / Δx`. Errors use
the matching observation of the reference. A nodal value is not
compared with a cell average.

### Methods

All three baselines use the same SSP-RK3 step. No spectral viscosity
and no limiter are added.

1. **`spectral`.** `ComplexLineFFT` on the full line. The product is
   the 2/3-rule: modes with `|m| ≥ N/3` are removed from each factor
   and again from the product. The derivative multiplier is `i m` on
   `[0, 2π)`, and the Nyquist multiplier is `0`. Viscosity is
   `-ν m²` on the undealiased linear term, including the Nyquist
   second derivative. The backward transform is divided by `N`.
2. **`fdm2`.** `-D_h(u²/2) + ν D_{xx,h} u` with
   `pfc::field::fd::evaluate<2>`. The caller wraps the periodic index.
   The stencil sum of a derivative is zero, so this form keeps the
   discrete mass.
3. **`fvm`.** Piecewise-constant Rusanov flux
   `½(f_L + f_R) - ½ s (u_R - u_L)` with `s = max(|u_L|, |u_R|)`.
   The viscous interface flux is `-ν (u_R - u_L) / Δx`. The update is
   the difference of those interface fluxes. Under the CFL
   restriction the first-order scheme does not increase the discrete
   maximum.

`spectral_aliased` skips both truncations and is only a diagnostic.
`fdm2_advective` replaces the flux derivative with `-u D_h u`. For
the order-2 central stencil, `Σ u_i (u_{i+1} - u_{i-1}) = 0`, so this
diagnostic also keeps the discrete mass. It is still not the
conservative update: the two tendencies differ pointwise, and the
evolved fields separate. Mass is not the witness.

The explicit step has to resolve the method's own stability limit.
Finite volume has a combined viscosity/flow screening criterion
`dt ≤ 1/(max|u|/dx + 2ν/dx²)`. The code rejects steps above the method's
current SSP-RK3 screening bound and nonfinite intermediate stages, before
publishing any candidate field. This is a conservative screening policy,
not a proof of post-shock spectral stability. The dealiased Fourier slope
keeps modes out to about `N/3`, so its explicit step is tighter than
the mesh CFL. An equal `Δt` is used in the tests only where every
method is inside its limit. The CLI reports `cfl_dt = Δx / max|u|`
and does not rewrite the requested step. It additionally reports the
initial conservative `stability_limit_initial` and `status=ok`.
CTest runs the actual CLI for FD2, FVM and (when available) spectral.
The file-and-failure integration check additionally verifies the 64 data rows,
cell-centre versus node coordinates, nonfinite values, unstable time-step
rejection and failed output-directory handling. The program emits `status=ok`
only after a requested `u.dat` write has succeeded.

Zero-mode conservation does not select the entropy solution after the
shock. The conservative Fourier product is not the skew-symmetric
form, so a flat discrete energy is not assumed. The entropy solution
dissipates `E` once the shock exists. The monotone update is the one
expected to follow that decay without an overshoot. Central FD2 and
dealiased Fourier may ring; that ring is reported, not filtered.

Rendering a figure does not launch a run. `--outdir` writes the final
samples only. The quantitative checks are `burgers-all-tests`.

## Run

```bash
./apps/burgers/burgers --method spectral --N 128 --nu 0 --dt 0.001 --time 0.5
./apps/burgers/burgers --method fdm2 --N 128 --nu 0 --dt 0.001 --time 0.5
./apps/burgers/burgers --method fvm --N 128 --nu 0 --dt 0.001 --time 1.5
```

The process prints method, SSP-RK3, `ν`, `N`, `Δx`, `Δt`, time, step
count, `cfl_dt`, mass, energy, maximum and a checksum. It is one MPI
rank and does not call HeFFTe. Further reconstruction, WENO, GPU and
MPI runs are outside this comparison.

## What the tests call

Each numerical check builds `burgers::Simulation`, calls
`set_initial_sine` or writes a Riemann state, then calls `rate` or
`advance`. The characteristic, Bessel and entropy functions live in
the test, not in the stepper.

- The sine wave's dealiased tendency equals `-sin(x)cos(x) - ν sin(x)`.
- A mode with `|m| ≥ N/3` is removed before the product; the aliased
  diagnostic keeps a nonzero tendency.
- FD2 on the initial sine wave is order 2. The Rusanov tendency,
  against the cell average of the same rate, is order 1.
- At `t = 0.5` and `ν = 0`, nodal or cell-average errors follow the
  characteristic solution. Halving the Fourier step reduces the error.
- At `ν = 0.5`, the fields match Cole–Hopf. The spectral energy drop
  matches the trapezoidal integral of `ν ∫ (u_x)² dx`.
- A periodic jump `+1 | -1` keeps the Rusanov shock at `x = π`.
- At `t = 1.5` and `ν = 0`, the finite-volume maximum stays within the
  initial cell-average bound, the Fourier maximum overshoots, the
  shock stays at `π`, and the entropy energy has decreased.

## Measured

The numbers below are the `BURGERS_MEASURE` lines from
`burgers-all-tests` on this revision. They are not fitted floors.
The pass bounds in the test are wider than these values and were not
used to change a stencil.

`burgers-all-tests` on this revision, Release, one CPU rank: 83 assertions, 10 cases.

| Check | Result |
| --- | --- |
| Dealiased sine tendency, `ν=0` and `ν=0.5`, `N=64` | peak error `2.4e-15` and `8.5e-14` |
| Mode `m=11` on `N=32` | dealiased tendency is roundoff; aliased peak is `2.5` |
| FD2 tendency, `t=0`, `ν=0` | `L2(32)=2.26e-2`, `L2(64)=5.68e-3`, ratio `3.98` |
| Rusanov tendency against the cell-average rate | `L2(32)=0.173`, `L2(64)=0.0869`, ratio `1.99` |
| Advective versus flux form | point defect `1.36e-2`; at `t=0.5`, `N=32` the fields differ by `1.56e-2`. Both masses stay near `1e-15` |
| Characteristic solution, `ν=0`, `t=0.5`, `Δt=0.001` | Fourier `L2` `7.05e-4` → `1.84e-6` from `N=32` to `64`. FD2 ratio `3.84`. Finite-volume ratio `1.75` |
| SSP-RK3, Fourier `N=64`, `t=0.5` | `L2(Δt=0.05)=2.76e-5`, `L2(Δt=0.025)=3.91e-6`, ratio `7.1` |
| Cole–Hopf, `ν=0.5`, `t=0.2` | Fourier `N=64` `L2=1.90e-11`. FD2 ratio `3.98`. Finite-volume `L2(64)=1.02e-2`, ratio `2.01` |
| Energy balance, Fourier `N=128`, same viscosity | `E` falls by `0.287`. Trapezoid of `ν∫(u_x)²` with `fd::evaluate<2>` misses that drop by `2.4e-4` |
| Rusanov jump `+1 \| -1`, `N=64`, `t=0.3` | steepest interface center `3.093`, half a cell off `π` because the face sits on `π` |
| Shock, `ν=0`, `t=1.5`, `N=128`, `Δt=0.001` | finite-volume max `0.966` stays under the initial cell max `1.000`. Fourier max `1.323`. FD2 max `1.549`. Finite-volume `L1` `0.101`; Fourier `L1` `1.061` |

The entropy solution's energy at that shock time is `1.356`. The initial energy is `π/2 ≈ 1.5708`. Finite volume ends at `1.298`, so the first-order flux dissipates more than the shock. FD2 ends at `1.410` and still overshoots. The dealiased Fourier energy is `1.570796325`, about `2e-9` below `π/2`. On this run the conservative product did keep the quadratic energy. That conserved energy is not the entropy solution: the Fourier field rings, and its `L1` error is about ten times the finite-volume error. The largest finite-volume jump is centered at `3.117`, which is `Δx/2` to the left of `π`. The shared face of that pair is the shock.

The evolved finite-volume ratio `1.75` at `t=0.5` is the steepening field, not a fitted rate. The initial Rusanov ratio `1.99` is the clean first-order check. No stencil coefficient was changed after these numbers.
