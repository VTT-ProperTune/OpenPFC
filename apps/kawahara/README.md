<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Kawahara (`apps/kawahara`)

Capillary-gravity **dispersive waves** with competing third- and fifth-order
dispersion. This is the 0.2 application for GitHub issue `#80`; `#119` adds a
documented physical-parameter mapping and a wave-packet/dispersive-radiation
science case on top of the existing arbitrary-coefficient verification.

Binaries: `kawahara` (CPU); `kawahara_hip` when `OpenPFC_ENABLE_HIP_SPECTRAL`
is on.

## Problem setup

| Item | Verification preset (`pulse.json`, mode tests) | Science preset (wave packet / solitary wave, `#119`) |
|---|---|---|
| Use case | Exercise \(L(k)=-i(\beta k^3+\gamma k^5)\) against independently derived phase | Capillary-gravity wave packet dispersion (Case A) and the loss of KdV integrability to fifth-order dispersion (Case B), both just below the critical Bond number |
| Domain | 1D periodic line, \(N_x=64\)-\(256\) | 1D periodic line, \(N_x=2560\), \(L_x=800\) (Case A) / \(N_x=512\), \(L_x=128\) (Case B) |
| Grid | \(dx=\pi/4\) to \(\pi/2\) | \(dx=0.3125\) (Case A) / \(dx=0.25\) (Case B, i.e. 13 points across the solitary wave's width and 16 across the radiated wavelength) |
| Boundary conditions | Periodic (both ends) | Periodic (both ends). Case A's domain is sized so the packet never reaches the boundary, and an `edge_fraction` sentinel checks it every `saveat`. Case B is measured *after* its radiation has wrapped: the observables there (tail RMS, dominant tail wavenumber) are whole-domain quantities for which wrapping is harmless, and no no-wrap claim is made |
| Initial condition | Single cosine mode / arbitrary-amplitude Gaussian pulse | `wave_packet`: Gaussian-envelope carrier at `k0` (Case A); `kdv_soliton`, the exact solitary wave of the `gamma=0` limit with its width derived from `alpha`/`beta` (Case B) |
| Key parameters | `alpha=1,beta=1,gamma=-1` (nondimensional, hand-picked) | `alpha,beta,gamma` from the documented \((h,g,\tau)\) mapping below; \(\tau=0.30\) |
| Observable | Phase, amplitude vs analytical cosine solution; mean \(u\) | Group velocity (envelope centroid), phase velocity (carrier mode), packet width, Fourier spectrum, edge/no-wrap sentinel (Case A); solitary-wave amplitude decay, propagation speed, trailing-radiation RMS and its dominant wavenumber against the resonance \(c_p(k)=c\) (Case B) |
| Model maturity | numerical verification: **analytical**; physical completeness: **canonical Kawahara ODE test**; calibration: **none** (arbitrary coefficients) | numerical verification: **analytical** (carrier phase velocity; solitary-wave amplitude, width and speed; radiation wavenumber from the dispersion relation) + **regression** (group velocity, tail RMS); physical completeness: **reduced** (1D, weakly nonlinear, long-wave asymptotics); calibration: **representative**, not quantitative (see honesty note below) |

## Physics

Long weakly nonlinear free-surface waves can need both \(\partial_x^3\) and
\(\partial_x^5\) when the leading dispersive term is small or changes sign.
The Kawahara equation (1972) is

\[
\partial_t u + \alpha u\partial_x u - \beta\partial_x^3 u + \gamma\partial_x^5 u = 0.
\]

The app defines `beta` with a minus sign in the PDE. For a reference equation
with `+b u_xxx`, supply `beta=-b`. Earlier documentation incorrectly printed
`+beta u_xxx`; the implementation and existing input results are unchanged.
Defaults \(\alpha=1\), \(\beta=1\), \(\gamma=-1\) give competing
dispersion. This is a 1D equation on a periodic line
(\(N_y=N_z=1\)).

Linear Fourier modes rotate rather than decay:

\[
\omega(k)=\beta k^3+\gamma k^5,\qquad
L(k)=-i\omega(k),\qquad
u_k(t)=u_k(0)\,e^{-i\omega(k)t}.
\]

Phase velocity \(c_p=\omega/k=\beta k^2+\gamma k^4\). With the default
signs, \(c_p>0\) for \(|k|<1\) (third-order wins) and \(c_p<0\) for
\(|k|>1\) (fifth-order wins). That is the opposite of even-order
dissipative operators such as Cahn-Hilliard or Mullins surface diffusion,
where \(k^4\) is a real negative multiplier and short waves simply damp.

The quadratic term is evaluated pseudospectrally as \(N=u^2\) with
\(M(k)=-i(\alpha/2)k_x\), under the Orszag 2/3-rule.

## Capillary-gravity parameter mapping (`#119`)

`include/kawahara/capillary_gravity_mapping.hpp` maps depth `h`, gravity `g`
and Bond number `tau = T/(rho g h^2)` (surface tension `T`, density `rho`) to
`(alpha, beta, gamma)`, in a frame moving at the linear long-wave speed
\(c_0=\sqrt{gh}\):

\[
\alpha = \frac{3c_0}{2h}, \qquad
\beta  = \frac{c_0h^2}{6}(3\tau-1), \qquad
\gamma = \frac{c_0h^4}{90}.
\]

`gamma` is always positive; `beta` is negative (competing dispersion, real
crossover \(k_c=\sqrt{-\beta/\gamma}\)) below the critical Bond number
\(\tau=1/3\), zero at it, and positive (no real crossover) above it. This
reduction is standard for capillary-gravity water waves near critical Bond
number and is attributed in the secondary literature to Hasimoto (1970) and
Kawahara (1972), with the near-critical analysis to Hunter & Vanden-Broeck
(1983).

**Honesty note (required by `#119`):** this formula was assembled from
secondary/tertiary literature summaries found by web search on 2026-09-09,
not from reading the primary Hasimoto (1970) / Kawahara (1972) papers, which
were not accessible from this machine (no general internet access at
build/run time; the original journal issues are not open-access and not
mirrored in this repository). The numeric prefactors recur consistently
across the independent secondary sources found, but have not been
independently re-derived from the Euler water-wave equations or checked
against a primary source here. Treat this mapping as **representative** of
the documented capillary-gravity regime, not as a quantitative,
primary-source-verified calibration. See the header's doc comment for the
full caveat.

## Science case A: linear wave-packet dispersion (`#119`)

`wave_packet_below_crossover.json` / `wave_packet_above_crossover.json` use
`h=g=1`, `tau=0.30` (near, but below, the critical Bond number), giving
\(\alpha=1.5\), \(\beta=-1/60\), \(\gamma=1/90\), crossover
\(k_c=\sqrt{1.5}\approx1.2247\). **`alpha` is set to `0` for this case**,
deliberately, to isolate the *linear* dispersion relation from nonlinear
self-steepening -- Case B below restores the full \(\alpha\).

Initial condition `"type": "wave_packet"`: a Gaussian-envelope carrier
\(u=A\exp(-(x-x_0)^2/2\sigma^2)\cos(k_0(x-x_0))\), \(\sigma=15\),
\(x_0=150\), on a \(N_x=2560\), \(L_x=800\) periodic line
(\(dx=0.3125\)). Two `k0` values straddle the crossover:

- **below**: `k0=0.6990` (\(nx=89\) exactly, an exact grid Fourier mode)
- **above**: `k0=2.0028` (\(nx=255\))

Run to \(t_1=250\); `diagnostics.csv` samples every `saveat=5`. Domain size
was chosen from the analytical group velocity
\(d\omega/dk=3\beta k^2+5\gamma k^4\) (largest magnitude \(\approx0.69\) for
the above-crossover packet) so the packet cannot drift more than a small
fraction of \(L_x\) in \(t_1\); this is checked automatically at run time via
`edge_fraction` (see below), not just asserted here.

```bash
mkdir -p results/kawahara
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/wave_packet_below_crossover.json
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/wave_packet_above_crossover.json
```

`diagnostics.csv` columns:
`step,time,mean,mode_amplitude,mode_phase,centroid,width,peak_envelope,edge_fraction`.

- `mode_amplitude`/`mode_phase` are the discrete Fourier coefficient at
  exactly `k0` (an exact grid mode): for the linear (`alpha=0`) run this
  coefficient evolves *exactly* as \(e^{-i\omega(k_0)t}\), so
  `mode_phase(t)-mode_phase(0)` gives the phase velocity
  \(c_p=\omega(k_0)/k_0\) essentially to machine/ETD precision, independent
  of the packet's bandwidth.
- `centroid`/`width` come from the envelope intensity, estimated as
  \(A(x)^2\approx2\times\text{lowpass}(u^2)\) (a standard narrowband
  quadrature-demodulation/Hilbert-transform-style technique: squaring
  removes the sign, and a spectral low-pass with quartic Gaussian rolloff
  `exp(-(k/(0.5*k0))^4)` removes the \(k_0\) and \(2k_0\) oscillations while
  passing the envelope's own (much lower) spectral content largely intact,
  leaving the slowly varying envelope).
  `centroid(t)` traces the group-velocity-driven drift of the packet;
  `d(centroid)/dt` is compared against \(d\omega/dk\) in the tests below.
- `edge_fraction` is the largest envelope value in the outer 5% of the domain
  on either side, relative to the peak envelope -- the automated "has the
  packet reached the periodic boundary" sentinel. Small (see `Tests`) at
  every sample in both cases at the parameters above.

**Measured on LUMI** (`SLURM_JOB_ID` shared allocation, `srun -n 1`), from
the actual `diagnostics.csv` of both runs above:

| `k0` | side | \(d\omega/dk\) analytic | \(v_g\) measured (centroid drift, \(t_1=250\)) | agreement | \(c_p=\omega/k\) analytic | phase velocity from `mode_phase` | agreement | max `edge_fraction` |
|---|---|---|---|---|---|---|---|---|
| 0.6990 | below \(k_c\) | \(-0.011167\) | \(-0.010915\) | 2.3% | \(-0.005491\) | \(-0.005491\) | \(\sim10^{-14}\) (exact) | \(9.3\times10^{-9}\) |
| 2.0028 | above \(k_c\) | \(0.693258\) | \(0.696118\) | 0.4% | \(0.111911\) | \(0.111911\) | \(\sim10^{-14}\) (exact) | \(1.0\times10^{-8}\) |

The phase-velocity agreement is to machine/ETD precision, as expected (the
discrete mode's evolution is exact for the linear step); the group-velocity
agreement is the narrowband-envelope-extraction accuracy, well within a few
percent. `edge_fraction` stays \(<10^{-7}\) throughout both runs: the packet
never approaches the periodic boundary over \(t_1=250\).

## Science case B: a KdV solitary wave forced to radiate (`#119`)

The question is what the fifth-order term *does*, and answering it needs a
control whose behaviour without that term is known exactly rather than merely
observed.

Setting `gamma=0` reduces this app's equation

$$u_t + \alpha u u_x - \beta u_{xxx} + \gamma u_{xxxxx} = 0$$

to KdV with dispersion coefficient \(\delta=-\beta\), and KdV has an exact
travelling solution:

$$u(x,t) = A\,\mathrm{sech}^2\!\Bigl(\frac{x-x_0-ct}{W}\Bigr),
\qquad c=\frac{\alpha A}{3},
\qquad W=\sqrt{\frac{-12\beta}{\alpha A}}.$$

That is the control. It propagates unchanged indefinitely, so *every*
departure from a constant peak amplitude and an empty tail in the
\(\gamma\neq0\) run is attributable to \(\gamma\).

`nonlinear_pulse_kdv_only.json` (`gamma=0`) and
`nonlinear_pulse_kawahara.json` (`gamma=1/90`) share `alpha=1.5`,
`beta=-1/60` (the \(\tau=0.30\) capillary–gravity mapping, i.e. just below
the critical Bond number), a `kdv_soliton` initial condition with
`amplitude=0.05` at `x0=32` on an \(N_x=512\), \(L_x=128\) line
(\(dx=0.25\)), and run to \(t_1=100\) with `dt=0.005`. The initial condition
takes `alpha` and `beta` rather than a width, and derives
\(W=1.63299\) from them, so an input cannot quietly stop being a solution of
the equation it is run against.

### What the fifth-order term should do

The solitary wave is resonant with the linear waves whose phase velocity
equals its own,

$$c_p(k) = \beta k^2 + \gamma k^4 = c,$$

which for \(\beta<0<\gamma\) has exactly one positive root — here
\(k_{\mathrm{res}}=1.5579\), a wavelength of \(4.03\), some 16 grid points and
well inside the \(2/3\) dealiasing cut at \(k=8.38\). Nothing in the solver is
told this number; it comes from the dispersion relation alone, which is what
makes it worth measuring.

```bash
mkdir -p results/kawahara
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/nonlinear_pulse_kdv_only.json
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/nonlinear_pulse_kawahara.json
```

Both are 1-D lines, so they run on a single rank; HeFFTe cannot split
\(N_y=N_z=1\) across several.

`diagnostics.csv` columns: `step,time,mean,peak_amplitude,peak_x,tail_rms`.
`tail_rms` is the RMS of `u` outside a window of \(\pm3W\) around the
instantaneous peak — the shed dispersive-radiation amplitude. A sech\(^2\)
holds over 99.9% of its area inside that window, and its remaining skirt is
what the control's small nonzero `tail_rms` measures.

### Measured on LUMI, at \(t=100\)

| Quantity | KdV control (`gamma=0`) | Full Kawahara (`gamma=1/90`) |
|---|---|---|
| `mean` | `0.0012757759076995707` | `0.0012757759076995744` |
| `peak_amplitude` | `0.05000699988643334` | `0.03598628436280036` |
| `peak_x` | `34.5` | `35.25` |
| `tail_rms` | `4.1875e-05` | `3.1327e-03` |

Reading that across:

- **The control is steady.** Its peak amplitude is unchanged to 1.4 parts in
  \(10^4\) after 20000 steps, and its `tail_rms` is the same
  \(4.19\times10^{-5}\) it started at — the sech\(^2\) skirt, not radiation.
  It has moved \(2.50\) code lengths, against \(cT = 0.025\times100 = 2.50\)
  from the closed form. This is the control behaving as an exact solution
  should, which is the whole reason the comparison below means anything.
- **The full run radiates.** `tail_rms` is 75× the control's, and the pulse
  has given up 28% of its amplitude to pay for it.
- **The radiation is at the predicted wavenumber.** The test measures the
  dominant wavenumber of the field outside the pulse window (with a
  raised-cosine mask, so the window's own edges contribute nothing at the
  scale of interest) and finds \(k=1.669\) against
  \(k_{\mathrm{res}}=1.5579\) — a 7% overshoot, consistent with the pulse
  radiating while its own amplitude, and therefore its speed, is still
  changing. The control's masked field peaks at \(k=0.049\), the lowest mode
  in the box: no wave train at all.
- **Mean is conserved** to a relative \(2\times10^{-15}\) in both, as it must
  be — both terms in the PDE are \(x\)-derivatives, so the \(k=0\) mode is
  untouched by construction.

**What this is and is not.** The mapping from \((h,g,\tau)\) to
\((\alpha,\beta,\gamma)\) is *representative*, not primary-source-verified;
see the provenance note in `capillary_gravity_mapping.hpp`. The numbers above
are exact statements about this solver on this input, and the resonance
agreement is a genuine physical check of it, but the run is not calibrated
against a laboratory water-wave experiment.

**Superseded approach.** Earlier revisions of this case used a Gaussian bump
of `amplitude=0.15`, `sigma=6` run to \(t=40\). A Gaussian solves neither
equation: it steepens under \(\alpha uu_x\), and this close to the critical
Bond number the dispersion available to arrest the steepening is weak
(\(|\beta|=0.017\)), so the "control" was integrating the numerical approach
to a gradient singularity — measurably flat only out to \(t\approx27\), with
the estimated Burgers breaking time at \(t\approx44\). Whether it survived to
\(t=40\) depended on the platform's rounding: it did on LUMI/Cray and
produced `NaN` on `ubuntu-24.04`/`gcc-13`. The difference it reported between
the two runs (a 6.8% change in trailing RMS) was also an order of magnitude
smaller than what the solitary wave shows. The solitary wave takes the
singularity out of the problem instead of timing the run to stop just short
of it.

## Run (original verification preset)

```bash
mkdir -p results/kawahara
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/pulse.json
```

The shipped input is a 256-point line (\(L_x=64\pi\)) with a Gaussian pulse.
VTK of `u` goes to `results/kawahara/`. The packet travels and radiates
dispersive ripples; it does not flatten the way a \(k^4\) smoother would.

JSON `model.params`: `alpha`, `beta`, `gamma`. Initial conditions:
`"type": "cosine_mode"` (`u0`/`amplitude`/`nx`), `"gaussian_pulse"`
(`amplitude`/`sigma`/`x0`), or `"wave_packet"`
(`amplitude`/`sigma`/`k0`/`x0`/`phase0`, `#119`).

## Tests

`ctest -R kawahara` checks \(\omega(k)=\beta k^3+\gamma k^5\), a linear
cosine against independently derived third-only and fifth-only solutions
with no amplitude loss, opposite phase
velocities on either side of \(|k|=1\), and mean-\(u\) conservation with
the quadratic term on -- unchanged, still the numerical-verification core.
`#119` adds, in the same binary/ctest target:

- the capillary-gravity mapping's `alpha`/`beta`/`gamma`/crossover formulas
  against hand-computed values;
- a self-test of the wave-packet diagnostics against a known static
  Gaussian-envelope carrier (recovers centroid/width/amplitude);
- measured group velocity (envelope centroid drift) and phase velocity
  (carrier mode phase) against \(d\omega/dk\) and \(\omega/k\) for `k0` on
  both sides of the crossover, plus the automated `edge_fraction` no-wrap
  check;
- that the `gamma=0` control is an exact KdV solitary wave (steady peak
  amplitude, closed-form propagation speed, no wave train behind it), that
  switching `gamma` on costs it 28% of its amplitude and raises the trailing
  RMS 75-fold, and that the shed wave train sits within 7% of the
  wavenumber where the linear phase velocity equals the pulse's own speed,
  mean-\(u\) conservation for both.

HIP builds add `HIP_KawaharaETD` and `kawahara-hip-smoke`; the `#119`
diagnostics/wave-packet path is CPU-only (`KawaharaSession`, not
`KawaharaHIPSession`) -- not wired onto the HIP session in this PR.
LUMI-G smoke: job 21792406 (`small-g`, 32-point line, mean \(u=0\)).

## Layout

| Path | Role |
|------|------|
| `include/kawahara/kawahara_physics.hpp` | Complex \(L(k)=-i\omega(k)\) |
| `include/kawahara/kawahara_pointwise.hpp` | \(N=u^2\) |
| `include/kawahara/kawahara_session.hpp` | JSON session, field `u`, 2/3 dealias, optional diagnostics CSV |
| `include/kawahara/capillary_gravity_mapping.hpp` | Physical \((h,g,\tau)\to(\alpha,\beta,\gamma)\) mapping (`#119`) |
| `include/kawahara/wave_packet.hpp` | Gaussian-envelope carrier IC (`#119`) |
| `include/kawahara/wave_packet_diagnostics.hpp` | Envelope/phase/tail-RMS diagnostics + CSV writers (`#119`) |
| `src/kawahara.cpp` / `src/hip/` | CPU / HIP `main` |
| `inputs_json/pulse.json` | Localized long-wave pulse (verification preset) |
| `inputs_json/wave_packet_{below,above}_crossover.json` | Science case A |
| `include/kawahara/kdv_soliton.hpp` | Exact KdV solitary-wave IC, width derived from `alpha`/`beta` (`#119`) |
| `inputs_json/nonlinear_pulse_{kdv_only,kawahara}.json` | Science case B |
