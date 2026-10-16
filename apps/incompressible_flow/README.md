<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Periodic incompressible flow

Maintained application for the periodic Fourier velocity. It is built
when HeFFTe is enabled. It is not a copy of
`examples/ns2d_vorticity/`. That directory remains the 2-D
vorticity–streamfunction prototype and the owner of its spatial,
temporal, and closed-box checks. This binary calls the installed Leray
projector, the 2/3-dealiased rotational term, and integrating-factor
RK4.

With the corresponding spectral device library enabled, the same maintained
application is also built as `incompressible_flow_cuda` or
`incompressible_flow_hip`. Both support Taylor--Green and seeded decaying HIT
on the existing distributed device FFTs. Initialization and sampled
diagnostics remain on the host; the IFRK4 stages remain resident between
samples. Metadata and help identify the actual selected backend. Forced HIT
and the locked convergence series remain host-only. One device spectral
backend is selected per build; neither command implements another FFT stack.

Device commands accept `--fft distributed|local`. The default remains
distributed HeFFTe, including at one rank. `local` explicitly selects the
existing vendor 3-D FFT on one rank with identical boxes, scaling and IFRK4
operations. It rejects multiple ranks and does not construct an unused
distributed device plan. Metadata identifies cuFFT or rocFFT for this choice.
This is not automatic policy selection; MPI decomposition is unchanged.

## Equation

The cube is \([0,2\pi]^3\). The HeFFTe pencil may be shared across
ranks. One rank remains a complete run. The advance is

\[
\partial_t\hat{\mathbf u}=P(\mathbf u\times\boldsymbol\omega)
-\nu|k|^2\hat{\mathbf u}.
\]

\(P\) is the Fourier Leray projector. The 2/3 mask stays on for this
quadratic term. There is no body force. Viscosity is the diagonal
symbol \(-\nu|k_{\mathrm{odd}}|^2\), so an unforced mean velocity is
invariant. CPU only.

## Taylor–Green

One shipped case is

\[
u=\sin x\cos y\cos z,\qquad
v=-\cos x\sin y\cos z,\qquad
w=0.
\]

Kinematic viscosity defaults to \(0.05\). The binary does not assign a
Reynolds number. The refinement tables for this field live in
[`examples/taylor_green3d/README.md`](../../examples/taylor_green3d/README.md).
At this viscosity the \(N=256\) series is the spatial reference, and
the timestep pairs that clear \(10^{-11}\) sit near order 4. This
application does not rerun that ladder.

## Command

```bash
./apps/incompressible_flow/incompressible_flow \
  --case taylor-green --n 32 --nu 0.05 --dt 0.01 --time 0.1 \
  --outdir results/taylor-green
```

`--time` must be an integer number of steps. \(N\) is even and at
least 8. Speed CFL is checked on every recorded sample. Above 2 the
run stops with `status=cfl`. A non-finite field leaves
`status=nonfinite`. The box length is \(2\pi\) and is not a flag.
`diagnostics.csv` records the initial and final samples.
`--sample-dt` adds a row at that interval, and the interval must be
an integer number of steps that divides the run. `--volume-dt`
writes \(|\omega|\) as one x-fastest float32 cube per interval,
including \(t = 0\), under `vorticity/`. Both flags belong to
Taylor–Green. The other cases reject them.

The same samples also write `spectrum.csv` (`time`, `shell`,
`shell_ke`) and `scales.csv` (`time`, `k_max`, `eta`,
`k_max_eta`). Shell \(s\) holds the modes in \([s-1/2, s+1/2)\).
\(k_{\max}\) is the largest integer strictly below \(N/3\), and
\(\eta=(\nu^3/\varepsilon)^{1/4}\). `metadata.txt` records the
sample trapezoid \(E(T)-E(0)+\int\varepsilon\,\mathrm{d}t\). The
initial field is not isotropic, so this case does not report a
Taylor microscale. `--profile` times the step and, on the device,
synchronizes between the spectral work, the FFTs, and the product.
Leave it off for a production trajectory.

## Decaying homogeneous isotropic turbulence

```bash
./apps/incompressible_flow/incompressible_flow \
  --case decaying-hit --n 32 --nu 0.02 --dt 0.015625 --time 0.0625 \
  --seed 1 --outdir results/decaying-hit
```

The target density is the curve used by Yoffe and McComb for freely
decaying isotropic turbulence (arXiv:1805.01238, 2018, equation 10):

\[
E(k)=0.266\left(\frac{k}{3.536}\right)^4
\exp\left[-\left(\frac{k}{3.536}\right)^2\right].
\]

On \([0,2\pi]^3\) the integer wave index is the physical wavenumber.
Their paper draws a Gaussian ensemble with this expectation. This
application keeps one seed and sets each retained mode to the isotropic
share \(E(|k|)/(4\pi|k|^2)\), including the Hermitian weight of the
real-to-complex buffer. The continuous integral of \(E\) is
\(0.266\times 3.536\times 3\sqrt{\pi}/8 \approx 0.625\). Modes removed
by the 2/3 mask are omitted, so the realized kinetic energy is the sum
over retained modes and is not rescaled afterward. The mean mode is
zero. A projected draw whose squared amplitude is below \(10^{-24}\) is
left at zero. The seed is hashed with the signed wave index, so the
same mode does not depend on the grid size or on the buffer order.
Default seed is 1. Viscosity defaults to \(0.02\), the value in their
Table 2 for the \(128^3\) Gaussian runs. This single realization is not
assigned their Taylor-Reynolds number.

Definitions used in `diagnostics.csv`:

- kinetic energy \(E=\tfrac12\langle\mathbf u\cdot\mathbf u\rangle\);
- enstrophy \(\Omega=\langle\boldsymbol\omega\cdot\boldsymbol\omega\rangle\),
  and \(\omega_{\mathrm{rms}}=\sqrt{\Omega}\);
- dissipation \(\varepsilon=\nu\Omega\), so the periodic balance is
  \(dE/dt=-\varepsilon\);
- \(u_{\mathrm{rms}}=\sqrt{2E/3}\);
- Taylor microscale \(\lambda=u_{\mathrm{rms}}\sqrt{15\nu/\varepsilon}\)
  and \(\mathrm{Re}_\lambda=u_{\mathrm{rms}}\lambda/\nu\) when
  \(\varepsilon>0\);
- Kolmogorov scale \(\eta=(\nu^3/\varepsilon)^{1/4}\);
- \(k_{\max}\) is the largest integer strictly below \(N/3\), and the
  resolution column is \(k_{\max}\eta\);
- the integral scale is
  \((3\pi/(4E_{\mathrm{modal}}))\sum_s E_s/s\), with shell \(s\) the
  modes in \([s-1/2,s+1/2)\) and \(E_s\) their kinetic energy;
- the outer fraction is the modal kinetic energy in modes with a
  component above \(N/6\), divided by the modal kinetic energy.

`spectrum.csv` stores \(E_s\) at the samples. `metadata.txt` stores the
seed, the spectrum, and the trapezoidal energy residual
\(E(T)-E(0)+\tfrac12(\varepsilon(0)+\varepsilon(T))T\).

### Resolution ladder

The ladder is fixed in the source before the campaign. It is not a
continuous-integration test. Samples are \(t=0\), \(0.5\), and \(1\).
Viscosity is \(0.02\) and the seed is 1. Spatial grids are
\(N=128,64,32\) at \(\Delta t=1/256\), finest first, plus one \(N=64\)
control at \(\Delta t=1/512\). Temporal steps at \(N=64\) are
\(\Delta t=1/256,1/128,1/64,1/32\), finest first. Field error is the
common-band velocity and vorticity on the coarser grid. The reference
row has a blank error. An order is
\(\log_2(e(\Delta t)/e(\Delta t/2))\) only when both absolute errors
exceed \(10^{-11}\) and the finer step is not the reference. Otherwise
the order is `roundoff`, or blank when the finer step is the
reference. Speed CFL is evaluated at the sample times. Above 2 that resolution
stops and the row is kept. The coefficients are checked after every
step, and a non-finite field stops the same way. Neither constant is
changed after the rows exist.

```bash
./apps/incompressible_flow/incompressible_flow \
  --case decaying-hit --series spatial --outdir results/hit-spatial
./apps/incompressible_flow/incompressible_flow \
  --case decaying-hit --series temporal --outdir results/hit-temporal
```

Both series finished with `status=ok`. Every resolution is in the
tables below. The binary is the build of this case on top of
`87d93d71`, and these numbers were copied in after the run. The
campaign files are
`/scratch/project_462001519/juaho/flow3d-230/`.

The continuous integral evaluates to \(0.625173\). On \(N=128\) the
realized kinetic energy at \(t=0\) is \(0.625173\), matching that
integral to a relative \(2\times 10^{-14}\). The tail past the 2/3
mask is below double precision on this spectrum, and the field is
not rescaled. \(N=64\) matches the same energy. \(N=32\) starts at
\(0.624847\), a relative gap of \(5.2\times 10^{-4}\), where the
mask removes a visible tail. The initial Taylor-Reynolds number on
\(N=128\), seed 1, is \(12.910\). Table 2 of Yoffe and McComb lists
\(12.91\) for the \(128^3\) runs at this viscosity. Seed 1 meets
that table at \(t=0\) because the spectrum and the viscosity fix
\(\mathrm{Re}_\lambda\). The later samples are this one realization.

At \(t=0\) on \(N=128\), \(u_{\mathrm{rms}}=0.6456\),
\(\lambda=0.3999\), \(L=0.5151\), \(\eta=0.05656\), and
\(\varepsilon=0.7817\). At \(t=1\) those are \(0.3392\), \(0.4221\),
\(0.6529\), \(0.08016\), and \(0.1937\), with
\(\mathrm{Re}_\lambda=7.160\).

Scalars on \(\Delta t=1/256\). The reference grid is \(N=128\).

| \(N\) | \(E(0)\) | \(E(1)\) | \(\Omega(1)\) | \(\mathrm{Re}_\lambda(1)\) | \(k_{\max}\eta(0)\) | \(k_{\max}\eta(1)\) |
|------:|---------:|---------:|--------------:|---------------------------:|--------------------:|--------------------:|
| 128 | 0.625173 | 0.172601 | 9.68567 | 7.15982 | 2.376 | 3.367 |
| 64 | 0.625173 | 0.172589 | 9.68498 | 7.15960 | 1.188 | 1.683 |
| 32 | 0.624847 | 0.173533 | 9.94457 | 7.10419 | 0.5659 | 0.7964 |

From \(N=64\) to \(N=128\), kinetic energy, enstrophy, and
\(\mathrm{Re}_\lambda\) at \(t=1\) agree to relative
\(6.6\times 10^{-5}\), \(7.1\times 10^{-5}\), and
\(3.0\times 10^{-5}\). The same three gaps at \(N=32\) are
\(5.4\times 10^{-3}\), \(2.7\times 10^{-2}\), and
\(7.8\times 10^{-3}\). \(k_{\max}\eta\) stays above 1 on \(N=64\)
and \(N=128\), and above 2 on \(N=128\). On \(N=32\) it is 0.566
at \(t=0\) and 0.796 at \(t=1\).

Relative common-band \(L^2\) against \(N=128\). The reference error
is blank. The outer fraction is the modal kinetic energy in modes
with a component above \(N/6\).

| \(N\) | velocity, \(t=0.5\) | velocity, \(t=1\) | vorticity, \(t=0.5\) | vorticity, \(t=1\) | outer fraction, \(t=1\) |
|------:|--------------------:|------------------:|---------------------:|-------------------:|------------------------:|
| 32 | \(1.72\times 10^{-1}\) | \(1.82\times 10^{-1}\) | \(3.69\times 10^{-1}\) | \(3.72\times 10^{-1}\) | \(1.99\times 10^{-1}\) |
| 64 | \(1.17\times 10^{-2}\) | \(6.44\times 10^{-3}\) | \(4.94\times 10^{-2}\) | \(2.85\times 10^{-2}\) | \(9.88\times 10^{-3}\) |
| 128 |  |  |  |  | \(2.68\times 10^{-5}\) |

At \(t=0\) the \(N=64\) velocity gap is \(2.01\times 10^{-8}\). The
\(N=32\) velocity gap is already \(2.28\times 10^{-2}\), the part
of the initial tail the mask removes. At later times the \(N=64\)
vorticity field stays a few percent from \(N=128\), and the \(N=32\)
vorticity error stays near \(0.37\). The scalar table is smoother
than the field.

The \(N=64\) control at \(\Delta t=1/512\), compared with the
\(N=64\) spatial field, moves the velocity by an absolute \(L^2\) of
\(2.78\times 10^{-9}\) and the vorticity by \(2.94\times 10^{-8}\)
at \(t=1\). The \(N=64\) versus \(N=128\) vorticity gap at that time
is \(8.87\times 10^{-2}\). That grid gap is \(3\times 10^{6}\) times
the halved-step gap. The spatial differences above are the grid.

The ladder residual is the three-point trapezoid of \(\varepsilon\)
on \(t=0\), \(0.5\), and \(1\). At \(t=1\) it is
\(1.115\times 10^{-2}\) on \(N=128\) and \(1.116\times 10^{-2}\) on
\(N=64\), a relative difference of \(1.0\times 10^{-3}\). \(N=32\)
gives \(1.318\times 10^{-2}\). The kinetic-energy drop on \(N=128\)
is \(0.453\), so the residual is \(2.5\%\) of that drop. The same
residual on the two finer grids is the high bias of a trapezoid on
a convex decaying dissipation. `metadata.txt` of a short user run
stores the two-point residual instead.

Absolute \(L^2\) at \(t=1\) on \(N=64\) against \(\Delta t=1/256\).
The order on a row compares that row with the next finer step.

| \(\Delta t\) | velocity \(L^2\) | \(p_u\) | vorticity \(L^2\) | \(p_\omega\) |
|---:|---:|---:|---:|---:|
| \(1/32\) | \(1.27\times 10^{-5}\) | 4.038 | \(1.32\times 10^{-4}\) | 4.027 |
| \(1/64\) | \(7.73\times 10^{-7}\) | 4.106 | \(8.10\times 10^{-6}\) | 4.100 |
| \(1/128\) | \(4.49\times 10^{-8}\) |  | \(4.72\times 10^{-7}\) |  |
| \(1/256\) |  |  |  |  |

The pairs that clear \(10^{-11}\) sit near 4: 4.106 and 4.100 at
\(\Delta t=1/64\), and 4.038 and 4.027 at \(\Delta t=1/32\). At
\(t=0.5\) those pairs are 4.100 and 4.092, then 4.028 and 4.008.
Every \(t=0\) order is the word `roundoff`, because the initial
field matches. The blank orders on \(\Delta t=1/128\) are the
reference pair. Speed CFL stays at most 0.954, on \(\Delta t=1/32\)
at \(t=0\). The largest modal divergence in the ladder is
\(1.33\times 10^{-10}\), on \(N=128\) at \(t=0\). Nodal divergence
\(L^2\) stays near \(10^{-16}\).

The prescribed curve peaks near wavenumber 5. On \(N=128\) the shell
energy peaks at shell 5 at \(t=0\) (\(0.154\)) and at shell 4 at
\(t=1\) (\(0.0367\)). Shells at and above 16 hold
\(2.0\times 10^{-3}\) of the energy at \(t=1\). Shell 42 holds
\(5.6\times 10^{-10}\). \(\mathrm{Re}_\lambda\) falls from 12.910
to 7.160 while the peak stays at those shells. \(N=128\) is the
grid on which \(k_{\max}\eta\) stays above 2. \(N=64\) carries the
scalars and stays a few percent off in vorticity. \(N=32\) stays
below \(k_{\max}\eta=1\).

\(N=128\) took 601 s, \(N=64\) took 33 s, and \(N=32\) took 2.7 s.

## Forced homogeneous isotropic turbulence

```bash
./apps/incompressible_flow/incompressible_flow \
  --case forced-hit --n 32 --nu 0.02 --power 0.25 --dt 0.015625 \
  --time 0.0625 --seed 1 --outdir results/forced-hit
```

The force is the constant-power scheme of Doering and Petrov
(arXiv:physics/0404049, 2004, equation 3). On \([0,2\pi]^3\) the
forced shell is the integer wavevectors with \(|k|=1\):

\[
\mathbf f=\varepsilon\frac{P\mathbf u}{2E_{\mathrm{band}}}.
\]

\(P\) keeps that shell, \(\varepsilon\) is the injected power, and
\(E_{\mathrm{band}}\) is the kinetic energy in the shell. The mean
mode is not forced. If \(E_{\mathrm{band}}\le 10^{-24}\) the force is
zero for that evaluation. The initial field is the decaying-hit seed.
Default power is \(0.25\) and default viscosity is \(0.02\). This is
the only forcing scheme in the binary.

`diagnostics.csv` adds the instantaneous injection and the band
energy. The two-point residual in `metadata.txt` is
\(E(T)-E(0)-\int(P-\varepsilon_{\mathrm{diss}})\,dt\) with the
trapezoid on the two samples.

### Stationary ladder

The ladder is fixed before the campaign. It is not a
continuous-integration test. Viscosity is \(0.02\), power is
\(0.25\), and the seed is 1. Samples are every \(0.5\) through
\(t=20\). Block averages use \([10,20]\), five blocks of length 2.
The uncertainty of a mean is the sample standard deviation of those
block means, divided by \(\sqrt{5}\). Spatial grids are
\(N=128,64,32\) at \(\Delta t=1/256\), finest first. \(N=64\) is
repeated at \(\Delta t=1/512\) and at \(\Delta t=1/128\). Speed CFL
above 2, or a non-finite field, stops that resolution and keeps the
row. These constants are not changed after the rows exist.

```bash
./apps/incompressible_flow/incompressible_flow \
  --case forced-hit --series stationary --outdir results/forced-hit
```

The one-rank ladder finished with `status=ok` on every row. The
numbers below were copied from
`/scratch/project_462001519/juaho/flow3d-231/summary.csv`. Full
precision stays in that file. The constants above were not changed.

Averages use \([10,20]\). The budget residual is
\(E(20)-E(10)-\int_{10}^{20}(P-\varepsilon)\,dt\), with the
trapezoid on the half-unit samples. It checks that identity. The
window can still drift while that residual stays small.

| run | \(N\) | \(\Delta t\) | \(\langle E\rangle\) | stderr | \(\langle\varepsilon\rangle\) | \(\mathrm{Re}_\lambda\) | \(k_{\max}\eta\) | budget residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 128 | \(1/256\) | 1.254104 | 0.024317 | 0.256418 | 45.3983 | 3.14496 | \(-6.906\times 10^{-5}\) |
| spatial | 64 | \(1/256\) | 1.254029 | 0.024300 | 0.256448 | 45.3910 | 1.57241 | \(-7.152\times 10^{-5}\) |
| spatial | 32 | \(1/256\) | 1.256298 | 0.027497 | 0.261146 | 44.9999 | 0.74505 | \(-2.850\times 10^{-4}\) |
| control | 64 | \(1/512\) | 1.254030 | 0.024301 | 0.256443 | 45.3916 | 1.57242 | \(-6.711\times 10^{-5}\) |
| coarse | 64 | \(1/128\) | 1.254008 | 0.024294 | 0.256496 | 45.3845 | 1.57231 | \(-1.074\times 10^{-4}\) |

Injection is \(0.25\) on every row. Its block uncertainty is at most
\(1.9\times 10^{-17}\).

On \(N=128\), kinetic energy is \(1.350101\) at \(t=10\) and
\(1.285856\) at \(t=20\), a fall of \(0.064245\). The first block
mean is \(1.336675\) and the last is \(1.241558\). Mean dissipation
is \(0.256418\), which exceeds the injected power by
\(6.42\times 10^{-3}\). The budget residual on this row,
\(-6.906\times 10^{-5}\), is how far that fall misses the trapezoid
of \(P-\varepsilon\). \(N=64\) has the same pattern. Kinetic energy
is still falling at the end of the window.

From \(N=64\) to \(N=128\), the kinetic-energy, dissipation, and
\(\mathrm{Re}_\lambda\) means agree to relative \(6.0\times 10^{-5}\),
\(1.2\times 10^{-4}\), and \(1.6\times 10^{-4}\). On \(N=32\),
\(k_{\max}\eta=0.745\). The dissipation mean there differs from
\(N=128\) by a relative \(1.8\times 10^{-2}\).

At \(N=64\), halving the step moves the kinetic-energy mean by a
relative \(1.0\times 10^{-6}\) and the dissipation mean by
\(1.8\times 10^{-5}\). Doubling the step moves them by
\(1.7\times 10^{-5}\) and \(1.9\times 10^{-4}\). The largest speed
CFL in the samples is \(0.283\), on \(N=128\). The largest modal
divergence is \(1.33\times 10^{-10}\), on \(N=128\) at \(t=0\).
Nodal divergence \(L^2\) stays near \(10^{-16}\).

\(N=128\) took 11143 s. The \(N=64\) spatial run took 862 s,
\(N=32\) took 58 s, the halved step took 1528 s, and the doubled
step took 360 s.

## Device step

`incompressible_flow_hip` advances Taylor–Green and decaying HIT with
the device integrating-factor step. HeFFTe uses rocFFT. The velocity
and the Runge–Kutta stages stay on the device. Diagnostics copy the
hats back at a sample. `--sample-dt` copies back at each of those
samples and leaves the step on the device. `--volume-dt` writes the
vorticity frames from the same copies. Forced HIT and the locked
series stay on `incompressible_flow`. `incompressible_device_parity`
compares one host step with one device step on the same pencil.

## Channel wall

The periodic binary stays periodic. A separate steady balance
checks an impermeable wall on one Fourier × Fourier × Chebyshev
rank. The pressure correction is the Neumann Poisson solve.
\(\mathrm{d}p/\mathrm{d}z\) matches the normal tendency at each
Chebyshev end, and the corrected acceleration is
divergence-free. No-slip is the Dirichlet value of the velocity
at those ends.

Plane Couette is \(u=z\), with wall values \(\pm 1\), zero body
force, zero flow rate, and unit wall shear. Plane Poiseuille is
\(u=1-z^2\), with zero wall values, flow rate \(4/3\), and wall
shear \(\mp 2\). A constant streamwise body force \(2\nu\)
balances that profile. The same force is a constant pressure
gradient for this profile. Both stay spectrally exact from
degree 2. One explicit viscous step uses that acceleration.
Couette and Poiseuille stay fixed under the step. The mode
\(\sin(\pi z)\) advances at \(-\nu\pi^{2}\), and
\(\sin(\pi z)\cos(2x)\) advances at \(-\nu(4+\pi^{2})\).
These profiles are parallel, so that step leaves the convective
term out. The term itself is −(u·∇)u on the same grid. It is zero
on those profiles. The streamfunction (1−z²)² sin(x) matches the
hand-derived product, and the impermeable projection of that
product is the acceleration. One explicit step adds that
acceleration to the viscous step and restores the walls. A
parallel profile agrees with the viscous step. Turbulent
channel statistics are not this case.

## Limits

The run is periodic and three-dimensional. Taylor–Green and
decaying-hit are unforced. forced-hit adds the one low-mode scheme
above. The locked ladders are one rank. A user run may use the
HeFFTe pencil. The periodic cases use one viscosity on a cube.
Pipe flow is outside this application. The channel section
records the steady laminar balance, one explicit viscous
step, the convective term, and one step that adds the two, on
the Chebyshev ends. A 2-D double shear is not
reimplemented here. The decaying-hit case
is one prescribed spectrum, not an ensemble, and it does not
claim an inertial range. On the locked decaying ladder,
kinetic energy, enstrophy, and \(\mathrm{Re}_\lambda\) agree between
\(N=64\) and \(N=128\) to better than \(10^{-4}\) relative at the
sample times. The vorticity field at \(N=64\) remains a few percent
from \(N=128\). \(N=128\) keeps \(k_{\max}\eta>2\). \(N=32\) stays
below \(k_{\max}\eta=1\). On the locked forced ladder, mean
dissipation on \([10,20]\) exceeds the injected power, so kinetic
energy still falls through that window. \(N=64\) and \(N=128\)
agree on those window means to about \(10^{-4}\) relative.
\(N=32\) again stays below \(k_{\max}\eta=1\).
