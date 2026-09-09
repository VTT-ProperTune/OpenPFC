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

| Item | Verification preset (`pulse.json`, mode tests) | Science preset (wave packet / nonlinear pulse, `#119`) |
|---|---|---|
| Use case | Exercise \(L(k)=-i(\beta k^3+\gamma k^5)\) against independently derived phase | Capillary-gravity wave packet dispersion and dispersive radiation near the critical Bond number |
| Domain | 1D periodic line, \(N_x=64\)-\(256\) | 1D periodic line, \(N_x=2560\), \(L_x=800\) (Case A) / \(N_x=512\), \(L_x=128\) (Case B) |
| Grid | \(dx=\pi/4\) to \(\pi/2\) | \(dx=0.3125\) (Case A) / \(dx=0.25\) (Case B) |
| Boundary conditions | Periodic (both ends) | Periodic (both ends); domain sized so the packet/radiation do not reach the boundary over the reported interval (checked automatically, see below) |
| Initial condition | Single cosine mode / arbitrary-amplitude Gaussian pulse | `wave_packet`: Gaussian-envelope carrier at `k0` (Case A); `gaussian_pulse`, representative (not literature-calibrated) amplitude (Case B) |
| Key parameters | `alpha=1,beta=1,gamma=-1` (nondimensional, hand-picked) | `alpha,beta,gamma` from the documented \((h,g,\tau)\) mapping below; \(\tau=0.30\) |
| Observable | Phase, amplitude vs analytical cosine solution; mean \(u\) | Group velocity (envelope centroid), phase velocity (carrier mode), packet width, Fourier spectrum, edge/no-wrap sentinel, dispersive-tail RMS |
| Model maturity | numerical verification: **analytical**; physical completeness: **canonical Kawahara ODE test**; calibration: **none** (arbitrary coefficients) | numerical verification: **analytical** (phase) + **regression** (group velocity, tail RMS); physical completeness: **reduced** (1D, weakly nonlinear, long-wave asymptotics); calibration: **representative**, not quantitative (see honesty note below) |

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

## Science case B: nonlinear localized pulse (`#119`)

`nonlinear_pulse_kdv_only.json` (`gamma=0`, third-order-only control) and
`nonlinear_pulse_kawahara.json` (`gamma=1/90`, full Kawahara) share
`alpha=1.5`, `beta=-1/60`, a `gaussian_pulse` initial condition
(`amplitude=0.15`, `sigma=6`, on a \(N_x=512\), \(L_x=128\) line,
\(dx=0.25\)), and run to \(t_1=40\) with `dt=0.005`. **Honesty note:** the
amplitude/width are a *representative* localized-pulse initial condition,
not a pre-computed exact Kawahara solitary-wave profile -- deriving that
profile (a nonlinear boundary-value problem in its own right) was out of
scope here, so the pulse disperses somewhat rather than propagating as a
clean coherent structure in either run. What is compared is the *difference
the fifth-order term makes* to the same initial pulse, not an exact soliton.
A larger amplitude (`0.3`) and coarser grid/timestep were tried first and
went unstable (`NaN` by `t=63`, confirmed by a real run, not assumed); the
shipped parameters were tuned down until both runs stayed finite to `t1`.

```bash
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/nonlinear_pulse_kdv_only.json
mpirun -n 1 ./apps/kawahara/kawahara \
  ../apps/kawahara/inputs_json/nonlinear_pulse_kawahara.json
```

`diagnostics.csv` columns: `step,time,mean,peak_amplitude,peak_x,tail_rms`.
`tail_rms` is the RMS of `u` outside a fixed window (`4*sigma` on each side)
around the instantaneous peak -- the shed dispersive-radiation amplitude.

**Measured on LUMI**, at `t=40`: `mean` is identical between the two runs to
17 significant digits (\(0.017624730055999\ldots\), conserved, as expected
from `M(k)` at `k=0`), i.e. the two runs differ only through `gamma`, not
through any drift in total \(u\). `peak_amplitude`: third-order-only
`0.16220`, full `0.15274` (full run's peak is **5.8% lower**).
`tail_rms`: third-order-only `0.014240`, full `0.013266` (full run's
trailing RMS is **6.8% lower** at `t=40`, `10.4%` lower at `t=38`) --
a reproducible, if modest, effect: at this amplitude/duration the fifth-order
term measurably softens the pulse (lower peak) while the two runs' peak
locations diverge slightly (`73.75` vs `73.25` at `t=40`), consistent with
the fifth-order term changing the effective nonlinear propagation, not
simply adding a separate radiating tail on top of an otherwise-identical
core.

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
- a reproducible difference in trailing-radiation RMS between a
  third-order-only and a full third+fifth-order nonlinear pulse run, and
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
| `inputs_json/nonlinear_pulse_{kdv_only,kawahara}.json` | Science case B |
