<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 3-D Taylor–Green on the Fourier velocity (issue #229)

Verification driver for the periodic Fourier velocity. It lives under
`examples/`, not in `apps/`. The step is the installed Leray projector,
the 2/3-dealiased rotational term, and integrating-factor RK4. This
directory does not contain a second projector.

The box is \([0,2\pi]^3\). The initial velocity is

\[
u=\sin x\cos y\cos z,\qquad
v=-\cos x\sin y\cos z,\qquad
w=0.
\]

Kinematic viscosity is \(0.05\). No Reynolds number is assigned.
Samples are \(t=0\), \(1/2\), and \(1\). Every step divides those times
in exact binary, so a sample is an integer step. One MPI rank. The
reference of each ladder is the finest series that reaches every sample
with a finite field and speed CFL at most 2. Every series in both
ladders finished. Field error is the common-band difference: matching
integer wavenumbers, scaled by
\(N_{\mathrm{coarse}}^3/N_{\mathrm{fine}}^3\), then inverted on the
coarse grid. Vorticity error is the curl of those hats. Norms are
discrete RMS values. Order is
\(p=\log_2\bigl(e(\Delta t)/e(\Delta t/2)\bigr)\) on absolute \(L^2\),
and only when both errors exceed \(10^{-11}\). Parent `8a6a257c1085`,
driver content `4144bd1da758`. Rows are under
`/scratch/project_462001519/juaho/flow3d-229/` and are not committed.

The initial field is the shell \(k^2=3\). Its kinetic energy is \(1/8\)
and its enstrophy is \(3/4\). On every grid those integrals match to
about \(10^{-13}\). The vertical velocity starts near \(10^{-17}\). A
viscous field that stayed on that shell would decay as
\(0.125\exp(-6\nu t)\). By \(t=1\) the nonlinear term has moved energy.
On the \(N=256\) series the \(k^2=3\) shell holds 96.0% of the kinetic
energy, \(k^2=8\) holds 3.74%, and \(k^2=11\) holds 0.219%. Measured
kinetic energy is 0.091872, which sits \(7.30\times 10^{-4}\) below the
exponential (about 0.8% relative). Enstrophy is 0.591, and \(6\) times
the kinetic energy is 0.551, so the decay is slightly faster than the
one-shell law. The exponential is a column in the output, not a target.
The vertical velocity at \(t=1\) has \(L^2\) \(6.02\times 10^{-2}\) and
maximum \(0.178\). That component is produced by vortex stretching.

Nodal divergence \(L^2\) at the later samples stays near \(10^{-18}\).
The largest modal divergence at \(t=1\) on \(N=256\) is
\(1.46\times 10^{-11}\). Mean velocity stays near \(10^{-18}\). Speed
CFL is at most 0.32, on the coarsest temporal step at \(t=0\).

`test_taylor_green3d` passed, 22 assertions in 2 cases. The smoke case
is \(N=16\), \(\Delta t=1/64\), through \(t=1/32\).

## Fixed-dt field error

One step, \(\Delta t=1/256\), on \(N=32\), \(64\), \(128\), and \(256\).
The reference is \(N=256\). A second \(N=64\) series uses \(\Delta t/2\)
and is compared with the \(N=64\) series at the spatial step.

```bash
./examples/taylor_green3d/taylor_green3d \
  --case spatial \
  --revision 8a6a257c1085+4144bd1da758 \
  --outdir /scratch/project_462001519/juaho/flow3d-229/spatial
```

Relative vorticity \(L^2\) against \(N=256\). The \(t=0\) column is the
sampling of one trigonometric field. The outer fraction is the share of
kinetic energy in modes with \(\max(|k_i|,|k_j|,|k_k|) > N/6\).

| \(N\) | \(t=0\) | \(t=0.5\) | \(t=1\) | outer fraction, \(t=1\) |
|------:|--------:|----------:|--------:|------------------------:|
| 32 | \(1.17\times 10^{-15}\) | \(3.25\times 10^{-7}\) | \(4.49\times 10^{-5}\) | \(1.82\times 10^{-6}\) |
| 64 | \(3.53\times 10^{-15}\) | \(8.00\times 10^{-14}\) | \(1.15\times 10^{-9}\) | \(4.54\times 10^{-11}\) |
| 128 | \(5.61\times 10^{-15}\) | \(2.17\times 10^{-16}\) | \(2.10\times 10^{-16}\) | \(8.09\times 10^{-21}\) |

At \(t=1\), \(N=32\) has relative velocity \(L^2\) \(7.00\times 10^{-6}\)
and \(L^\infty(\omega)\) \(1.19\times 10^{-4}\). \(N=64\) has relative
velocity \(L^2\) \(9.21\times 10^{-11}\) and \(L^\infty(\omega)\)
\(4.90\times 10^{-9}\). \(N=128\) matches \(N=256\) to roundoff, near
\(10^{-16}\). Doubling \(N\) from 32 to 64 cuts the \(t=1\) vorticity
\(L^2\) by about \(4\times 10^{4}\). The outer band at \(N=32\) holds
\(1.82\times 10^{-6}\) of the energy. The field at this viscosity is
smooth.

Kinetic energy at \(t=1\) agrees from \(N=32\) upward to about
\(10^{-13}\). Enstrophy agrees to about \(6\times 10^{-11}\). Those
scalars hide the \(4.49\times 10^{-5}\) vorticity error still present
at \(N=32\).

Repeating \(N=64\) at \(\Delta t/2\) changes the vorticity by an
absolute \(L^2\) of \(2.11\times 10^{-12}\) at \(t=1\). The \(N=64\)
versus \(N=256\) gap at that time is \(8.87\times 10^{-10}\), about
420 times larger. The spatial table is not timestep-limited.

## Fixed-grid temporal order

The spatial table changes \(N\) at one \(\Delta t\). This run holds
\(N=64\) and uses
\(\Delta t=1/32\), \(1/64\), \(1/128\), \(1/256\), and \(1/512\).
The reference is \(1/512\). The same-grid comparison has scale 1.
Order is withheld at or below \(10^{-11}\), and the reference row has
no order: its error against itself is zero.

```bash
./examples/taylor_green3d/taylor_green3d \
  --case temporal \
  --revision 8a6a257c1085+4144bd1da758 \
  --outdir /scratch/project_462001519/juaho/flow3d-229/temporal
```

Absolute \(L^2\) at \(t=1\) against \(\Delta t=1/512\). The order on a
row is the comparison with the next finer step:

| \(\Delta t\) | velocity \(L^2\) | \(p_u\) | vorticity \(L^2\) | \(p_\omega\) |
|---:|---:|---:|---:|---:|
| \(1/32\) | \(1.66\times 10^{-9}\) | 3.997 | \(9.15\times 10^{-9}\) | 3.995 |
| \(1/64\) | \(1.04\times 10^{-10}\) | roundoff | \(5.74\times 10^{-10}\) | 4.003 |
| \(1/128\) | \(6.47\times 10^{-12}\) | roundoff | \(3.58\times 10^{-11}\) | roundoff |
| \(1/256\) | \(3.81\times 10^{-13}\) |  | \(2.11\times 10^{-12}\) |  |
| \(1/512\) | 0 |  | 0 |  |

The completed pairs above the floor sit near 4. Velocity has one such
pair, at \(\Delta t=1/32\). Vorticity has two, 3.995 and 4.003. Finer
velocity errors are at the floor, so those orders are withheld. At
\(t=0.5\) the coarse pair is 3.996 in velocity and 3.994 in vorticity,
and the next vorticity pair is 4.002. The \(t=0\) errors are zero on
every step. The blank cells on \(\Delta t=1/256\) are the reference
pair, not a missing run.

The same \(N=64\) spectrum at \(t=1\) matches the \(N=256\) shells
quoted above, and the outer-band fraction there is \(4.54\times 10^{-11}\).
Modal divergence on the reference step stays at most about
\(2\times 10^{-13}\).

These runs show a divergence-free 3-D vortex whose vertical velocity
grows, a smooth spatial ladder through \(N=256\), and fourth-order
gaps on the timestep pairs that clear \(10^{-11}\). They do not assign
a Reynolds number, compare with a published decay curve, or measure an
order inside that floor. The reference is \(N=256\), not a finer grid.
