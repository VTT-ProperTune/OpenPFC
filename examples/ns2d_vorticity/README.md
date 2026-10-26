<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 2-D vorticity–streamfunction Navier–Stokes (issue #21)

Bounded research prototype under `examples/`, not a shipped application.
It is a verification experiment for the spectral stack. Moving it to
`apps/` is a separate decision under the
[admission criteria](../../apps/README.md#what-belongs-here); a fixed
application count is not why it stays here. Shared spectral pieces are
`include/ns2d/spectral.hpp`. Navier–Stokes is
`vorticity_stream.hpp` (#21/#22). Two-dimensional incompressible
visco-resistive MHD is `mhd.hpp` (#23). This is **not** Strauss reduced
MHD and it is **not** Cahn–Hilliard–Navier–Stokes.

## Equations

Periodic 2-D incompressible Navier–Stokes in vorticity–streamfunction
form:

$$
\partial_t\omega+\mathbf u\cdot\nabla\omega=\nu\nabla^2\omega,
\qquad
\nabla^2\psi=-\omega,
\qquad
\mathbf u=(\partial_y\psi,-\partial_x\psi).
$$

OpenPFC's FFT stack is three-dimensional; the grid is an
$N\times N\times 1$ slab and $k_z=0$ for every mode.

Taylor–Green uses $[0,2\pi]^2$. The Minion–Brown / Bell double shear
uses the **unit square**. Do not put $\rho=30$ on $[0,2\pi]^2$: that
makes the layer $2\pi$ times thinner than the benchmark. The same
physical layer on $[0,2\pi]^2$ would need $\rho'=\rho/(2\pi)$ and
$\nu'=2\pi\nu$. This code does not do that transformation.

### Zero mode

$\hat\psi(\mathbf 0)=0$. Mean vorticity is conserved: $L(0)=0$ and
$\hat N(\mathbf 0)=0$. The code does not project
$\hat\omega(\mathbf 0)$ to zero.

### Discretization

- FFT Poisson: $\hat\psi_{\mathbf k}=\hat\omega_{\mathbf k}/|\mathbf k|^2$
  for $\mathbf k\neq 0$.
- Spectral derivatives use $i k$ with the Nyquist component zeroed
  (`k_component_odd`).
- **2/3 dealiasing of the state and of $\hat N$.** The initial
  condition and every IFRK4 stage are projected onto the retained band
  *before* real-space products. Masking only $\hat N$ after the product
  is not enough if the state still holds modes above the cutoff.
- **Integrating-factor RK4** on $L=-\nu|\mathbf k|^2$. The viscous
  piece is $e^{L\Delta t}$; the Jacobian is classical RK4 in the IF
  variable. When $N\equiv 0$ (Taylor–Green) the step is exact. When
  $L\equiv 0$ it is RK4, whose stability region includes the imaginary
  axis up to $|z|\approx 2\sqrt{2}$. ETDRK2/Heun does not
  ($|1+i\omega+(i\omega)^2/2|>1$ for every $\omega\neq 0$).

## Diagnostics

2-D means over the $N^2$ cells ($N_z=1$):

| Name | Normalization |
|------|----------------|
| `ke` | $\frac12\langle u^2+v^2\rangle$ |
| `enstrophy` | $\frac12\langle\omega^2\rangle$ |
| `max_abs_omega` | $\max\Vert \omega\Vert $ |
| `div_linf` / `div_l2` | $L^\infty$ and RMS of spectral $\partial_x u+\partial_y v$ |
| `cfl` | $\Delta t\,\max(\Vert u\Vert ,\Vert v\Vert )/\min(\Delta x,\Delta y)$ |
| `mean_omega` | $\langle\omega\rangle$ (conserved) |
| `wall_step_s` | wall time of the last timed step |

Shear resolution (unit square, thickness $1/\rho$):

$$
\text{cells/thickness}=N/\rho,\qquad
\text{2/3-effective}=(2N/3)/\rho.
$$

For $\rho=30$: $N=128$ has 4.27 cells (2.84 effective); $N=256$
has 8.53 (5.69); $N=512$ has 17.07 (11.38).

## HIP / GPU

Not implemented. `GPUSpectralStack` / `SpectralETDOps` expose pointwise
$N(\psi)$ plus a diagonal $L(k)$, not Poisson plus four spectral
derivatives.

## Commands

```bash
./scripts/build.sh --machine=lumi --cpu --no-submit \
  --build-dir=/flash/project_462001519/juaho/build/openpfc-ns2d-21-cpu
ctest --test-dir <build> -R ns2d-vorticity --output-on-failure
```

Taylor–Green (correctness, not a movie):

```bash
./examples/ns2d_vorticity/ns2d_vorticity --verify --N 32 --steps 20 \
  --dt 0.05 --nu 0.1 --outdir results/ns2d/tg
```

Minion–Brown unit-square shear, CFL $0.4$ so
$\Delta t=0.4/N$, $\nu=10^{-4}$, $\rho=30$, $\varepsilon=0.05$,
to the standard $t=1.2$:

```bash
# N=256: 8.5 cells/thickness; animation-ready VTK
./examples/ns2d_vorticity/ns2d_vorticity --case shear --N 256 \
  --steps 768 --cfl 0.4 --nu 1e-4 --rho 30 --eps 0.05 --dump 64 \
  --outdir /scratch/project_462001519/juaho/ns2d-21-ifrk4/shear256
```

On LUMI write large series under scratch, not flash. ParaView loads
`omega_*.vti`. Rank 0 writes `diagnostics.csv` and `run.json`.

The driver aborts on non-finite diagnostics or CFL $> 2$.

## Evidence (corrected integrator / scaling / dealias)

CPU HeFFTe 2.3.0 FFTW/Milan, 1 rank. Catch2: 10 cases, 341 assertions.
CLI `--verify` PASS. Source: this branch after the PR #22 review.

### Taylor–Green $N=32$, $\nu=0.1$, $\Delta t=0.05$, $t=1$

- $L^\infty(\omega)=2.76\times10^{-15}$
- KE / enstrophy match $e^{-4\nu t}/4$ and $e^{-4\nu t}/2$ to
  $\sim10^{-15}$
- `div_linf` $\sim10^{-15}$

### Nonlinear tests (Taylor–Green cannot satisfy these)

- Two-mode IFRK4: error ratio $>8$ when $\Delta t$ is halved
  (first-order ETD1-Euler would be $\approx 2$).
- Inviscid two-mode, 80 steps, $\Delta t=0.02$: finite, KE relative
  drift $<10^{-3}$, mean $\omega$ conserved.

### Aliasing

On $N=32$, $\sin(12x)\sin(13x)$ aliases into the retained $k=1$
cosine. Projecting each factor onto the 2/3 band before the product
removes that contamination.

### Minion–Brown shear, $\rho=30$, $\nu=10^{-4}$, CFL $0.4$

All three resolutions finished $t=1.2$ finite. Mean $\omega$ is
constant to printer precision. Spectral divergence stays $\le10^{-12}$.

| Grid size $N$ | cells / $1/\rho$ | KE($t=1.2$) | Enstrophy $Z(t=1.2)$ | Peak vorticity $\max\Vert \omega\Vert $ | $\max\Vert u\Vert $ peak |
|------:|-------------------:|--------------:|-------------:|-------------------:|-------------------:|
| 128 | 4.27 (2.84 eff.) | 0.425162 | 32.1945 | 28.297 | 1.438 at $t=0.8$ |
| 256 | 8.53 (5.69) | 0.425162 | 32.1923 | 28.206 | 1.439 at $t=0.8$ |
| 512 | 17.07 (11.38) | 0.425162 | 32.1923 | 28.206 | 1.439 at $t=0.8$ |

256 and 512 agree to five digits on KE, enstrophy and
$\max\|\omega\|$ at $t=1.2$. $\max\|u\|$ grows $1.00\to 1.44$
by $t=0.8$ (KH roll-up) then sits near $1.28$–$1.33$ through
$t=1.6$ on $N=256$ with no blow-up.

Animation-ready series: `shear256_t16` (17 VTK frames, $t=0$–$1.6$)
under `/scratch/project_462001519/juaho/ns2d-21-ifrk4/`. Frames were
not rendered in this session; the claim of roll-up is from the
converged diagnostics, not from a screenshot.

### Fixed-dt field error (issue #224)

The CFL $0.4$ table changes $\Delta t$ with $N$. The run below
holds one step,
$\Delta t = 0.4/512 = 7.8125\times 10^{-4}$, on the same shear
($\nu=10^{-4}$, $\rho=30$, $\varepsilon=0.05$, unit square).
Output times are $t=0.4$, $0.8$, and $1.2$ (512, 1024, and 1536
steps). The reference is $N=512$. Error is the common-band
difference: `restrict_hat_by_k` copies matching modes, scaled by
$N_{\mathrm{coarse}}^2/N_{\mathrm{fine}}^2$, and the norms are
discrete RMS values on the coarse grid. One MPI rank. The 2/3
projection stays on. Taylor–Green is not this comparison.

```bash
./examples/ns2d_vorticity/ns2d_spatial_refine \
  --resolutions 64,128,256,512 --reference 512 \
  --times 0.4,0.8,1.2 \
  --control-n 256 --control-dt-factor 0.5 \
  --outdir /scratch/project_462001519/juaho/ns2d-224/ladder-512
```

$N=1024$ at this same $\Delta t$ matches $N=512$ through
$t=0.8$ (kinetic energy differs by $3\times 10^{-11}$) and is
non-finite at the $t=1.2$ sample. Its speed CFL at $t=0.8$ is
1.15, below the driver's abort threshold of 2, so the run records
the failure and does not use that grid as the reference. Rows:
`/scratch/project_462001519/juaho/ns2d-224/ladder/`. The completed
64–512 rows are under `ladder-512/`. Both `run.json` files record
parent `6316f822425a` and driver content `a474b887e3bb`.

Relative vorticity $L^2$ against $N=512$:

| Grid size $N$ | Error at $t=0.4$ | Error at $t=0.8$ | Error at $t=1.2$ | outer fraction, $t=1.2$ |
|------:|----------:|----------:|----------:|--------------------------:|
| 64 | $1.12\times 10^{-2}$ | $8.44\times 10^{-2}$ | $1.71\times 10^{-1}$ | 0.289 |
| 128 | $8.81\times 10^{-5}$ | $6.50\times 10^{-3}$ | $1.52\times 10^{-2}$ | 0.093 |
| 256 | $6.58\times 10^{-8}$ | $1.37\times 10^{-5}$ | $3.76\times 10^{-5}$ | 0.0091 |

The outer fraction is the share of vorticity RMS in modes with
$\max(|k_i|,|k_j|) > N/6$. The 2/3 mask already drops modes past
about $N/3$, so this is the outer half of the retained band.

At $t=1.2$, $N=64$ (about 2.1 cells per $1/\rho$) is still
under-resolved: relative vorticity error $0.17$, and 29% of the
vorticity RMS sits in that outer band. $N=128$ is at
$1.5\times 10^{-2}$. $N=256$ is at $3.76\times 10^{-5}$, with
velocity relative $L^2$ $6.1\times 10^{-7}$ and
$L^\infty(\omega)=1.6\times 10^{-3}$. Doubling $N$ from 128 to
256 cuts the $t=1.2$ vorticity $L^2$ by about 400. That drop is
spectral, not a low algebraic order.

Repeating $N=256$ at $\Delta t/2$ changes the vorticity by a
relative $L^2$ of $4.1\times 10^{-8}$ at $t=1.2$. That gap is
about a thousand times smaller than the $N=256$ versus $N=512$
difference, so the spatial table is not timestep-limited.

Kinetic energy at $t=1.2$ is already 0.425162 from $N=128$ upward.
Enstrophy agrees to five digits from $N=256$ (32.1923; $N=64$ is
32.45 and $N=128$ is 32.1946). Those scalars hide the
$1.5\times 10^{-2}$ vorticity error still present at $N=128$.

Spectral divergence on every completed row stays at most about
$10^{-12}$. Mean $\omega$ stays at its grid value, near
$10^{-7}$.

This establishes rapid common-band convergence of the shear once the
layer is past $N=64$, at a step small enough that time error does
not set the gap. It does not establish an error against a grid finer
than 512, a temporal order, or a finished $1024^2$ run at this
$\Delta t$.

### Fixed-grid temporal order (issue #225)

The spatial table changes $N$ at one $\Delta t$. The runs below
hold the grid fixed and change the step. Error is the same-grid
common-band difference (`restrict_hat_by_k` with scale 1). Order is
$p=\log_2\bigl(e(\Delta t)/e(\Delta t/2)\bigr)$ on absolute $L^2$
against the finest step, and only when both errors exceed
$10^{-11}$. The finest step is not an order denominator: its error
against itself is zero. A step that becomes non-finite is kept. Its
diagnostics are left at zero, so the status column is the record.
Taylor–Green is not a case. One MPI rank. The 2/3 projection stays on.
Parent `7aa7cc9a57a8`, driver content `804741a32904`. Rows are under
`/scratch/project_462001519/juaho/ns2d-225/` and are not committed.

Shear, $N=512$, unit square, $\nu=10^{-4}$, $\rho=30$,
$\varepsilon=0.05$, at $t=0.4$ (before roll-up), $0.8$, and
$1.2$. The steps are
$\Delta t=(0.4/2048)\times\{16,8,4,2,1\}$, so the reference is
$0.4/2048$.

```bash
./examples/ns2d_vorticity/ns2d_temporal_refine \
  --case shear \
  --outdir /scratch/project_462001519/juaho/ns2d-225/shear
```

Absolute vorticity $L^2$ against $\Delta t=0.4/2048$. The order
on a row is the comparison with the next finer step:

| Time step $\Delta t$ | Error at $t=0.4$ | Error at $t=0.8$ | Error at $t=1.2$ | Order $p(0.4)$ | Order $p(0.8)$ | Order $p(1.2)$ |
|---:|---:|---:|---:|---:|---:|---:|
| $0.4/128$ | non-finite |  |  |  |  |  |
| $0.4/256$ | $6.64\times 10^{-10}$ | $1.95\times 10^{-6}$ | non-finite | 4.005 | 4.004 |  |
| $0.4/512$ | $4.14\times 10^{-11}$ | $1.22\times 10^{-7}$ | $3.48\times 10^{-7}$ | roundoff | 4.087 | 4.088 |
| $0.4/1024$ | $2.54\times 10^{-12}$ | $7.17\times 10^{-9}$ | $2.05\times 10^{-8}$ |  |  |  |
| $0.4/2048$ | 0 | 0 | 0 |  |  |  |

A fourth-order gap measured against this reference, rather than
against an exact solution, has observed order 4.005 on the
$0.4/256$ pair and 4.087 on the $0.4/512$ pair. The measured
vorticity orders agree to about $0.001$. Velocity $L^2$ gives
4.005 at $t=0.8$ on the coarse pair and 4.088 on the fine pair.
At $t=0.4$ the velocity errors are already below the floor, so
that order is withheld. The blank cells on $0.4/1024$ are the reference pair,
not a missing run.

$\Delta t=0.4/128$ is non-finite at the $t=0.4$ sample, so no
CFL is recorded for it. $\Delta t=0.4/256$ is finite at $t=0.8$
with speed CFL 1.15 and non-finite at the $t=1.2$ sample. That is
the same CFL as the last finite sample of the $N=1024$ spatial run.
The abort threshold of 2 does not catch it. Every finer step
finishes. Speed CFL at $t=0.8$ is then 0.58, 0.29, and 0.14.
Spectral divergence on the finished rows stays at most about
$10^{-12}$. Mean $\omega$ stays at its grid value, near
$-1.4\times 10^{-7}$, and the drift from the initial mean is about
$10^{-14}$.

At the spatial-study step $\Delta t=0.4/512$ and $t=1.2$, the
vorticity error is $3.5\times 10^{-7}$ absolute and
$4.3\times 10^{-8}$ relative. Kinetic energy differs by
$4\times 10^{-12}$ and enstrophy by $4\times 10^{-9}$. The outer
half of the retained band holds $2.8\times 10^{-5}$ of the
vorticity RMS, so this $N=512$ window is not resolution-limited.

Two-mode, $N=64$ on $[0,2\pi]^2$, $\nu=0.05$, $t=0.2$. The
initial field is modes 1 and 2, and the quadratic products through
mode 4 sit inside the 2/3 mask, so this $N$ is set by that content
rather than by the shear study. Steps are
$\Delta t=(0.2/160)\times\{32,16,8,4,2,1\}$.

```bash
./examples/ns2d_vorticity/ns2d_temporal_refine \
  --case two-mode \
  --outdir /scratch/project_462001519/juaho/ns2d-225/two-mode
```

Every step finishes, with speed CFL at most 0.22. Absolute vorticity
$L^2$ is $1.63\times 10^{-10}$ at $\Delta t=0.04$ and
$1.02\times 10^{-11}$ at $\Delta t=0.02$. That is the one pair
above the floor, and it gives $p=3.994$. Finer steps are roundoff.
Velocity $L^2$ on that pair is below the floor on the fine member,
so its order is withheld. This case does not show fourth order over
a range. The existing Catch2 check still rejects a first-order step:
$L^\infty$ on the two-mode field at $N=32$ must drop by more than
8 when $\Delta t$ is halved, and that test does not read these
roundoff values.

The shear field errors support fourth order on the two completed
doublings that stay above the floor at $t=0.8$, and on the inner
doubling at $t=1.2$. The stable step for this shear through
$t=1.2$ is $\Delta t\le 0.4/512$. The two coarser steps do not
finish that window. The runs do not establish an order inside the
floor, or a cause for the non-finite rows.

### Closed-box flux (issue #226)

The recovered velocity is a curl in the stored Fourier basis,
$\hat u = i k_y^{\mathrm{odd}}\hat\psi$ and
$\hat v = -i k_x^{\mathrm{odd}}\hat\psi$. The 2/3 mask is a
multiplier, so it leaves that curl a curl. The modal divergence
$i k_x^{\mathrm{odd}}\hat u + i k_y^{\mathrm{odd}}\hat v$ is zero
in exact arithmetic. `face_flux` integrates the trigonometric
interpolant of those coefficients, with the unstored negative-$k_x$
conjugate filled in. On any closed axis-aligned box that integral is
zero in exact arithmetic. The numbers below are the size of that
summation. They stay in one band across $N$, $\Delta t$, and box
size, and that band is roundoff of the represented field.

The domain integral of the divergence is the zero mode times the
area. Every periodic Fourier series has that coefficient equal to
zero. The column is computed from the mode. A zero there is the
periodic sum, shared by a field with a large local divergence.

The `trap_net` column is a composite trapezoid on grid nodes, filled
only when all four corners lie on nodes. It samples the same nodal
values with a low-order rule. On a full period the opposite faces
use the same nodes, so that trapezoid is zero by the periodic
identification.

`relative` divides `net` by the sum of the absolute face integrals.
That ratio tracks the residual when the denominator is larger than
roundoff. On the Taylor–Green full-period box the denominator is
about $10^{-31}$ and the printed relative reaches $0.81$, while
`net` stays at $4\times 10^{-31}$.

One MPI rank. The 2/3 projection stays on. Parent `b80e241c721e`,
driver content `fed509516444`. Rows are under
`/scratch/project_462001519/juaho/ns2d-226/` and are not committed.
Shear uses the steps that finish in the temporal study: $N=128$,
$256$, and $512$ at $\Delta t=0.4/512$, and $N=256$ again at
$\Delta t=0.4/1024$. Unit square, $\nu=10^{-4}$, $\rho=30$,
$\varepsilon=0.05$, at $t=0.4$, $0.8$, and $1.2$. Boxes are
the full period, the half box $[0,L/2]^2$, the quarter
$[L/8,3L/8]^2$, one cell $[0,\Delta x]^2$, and the off-node
rectangle $[0.2L,0.7L]\times[0.15L,0.45L]$.

```bash
./examples/ns2d_vorticity/ns2d_flux_balance \
  --case shear \
  --outdir /scratch/project_462001519/juaho/ns2d-226/shear
```

All 60 shear rows have status `ok`. Absolute spectral `net` at
$t=1.2$ and $\Delta t=0.4/512$:

| box | Grid $N=128$ | Grid $N=256$ | Grid $N=512$ |
|---|---:|---:|---:|
| full | $9\times 10^{-33}$ | $8\times 10^{-32}$ | $2\times 10^{-31}$ |
| half | $1\times 10^{-16}$ | $6\times 10^{-17}$ | $5\times 10^{-16}$ |
| quarter | $8\times 10^{-16}$ | $2\times 10^{-15}$ | $7\times 10^{-16}$ |
| cell | $9\times 10^{-18}$ | $1\times 10^{-17}$ | $1\times 10^{-18}$ |
| offset | $4\times 10^{-16}$ | $1\times 10^{-15}$ | $2\times 10^{-15}$ |

The largest `|net|` on the ladder, including $t=0.4$, $t=0.8$,
and the half-step, is $4.5\times 10^{-15}$. The largest `|imag|`
is $7\times 10^{-17}$. Halving $\Delta t$ at $N=256$ leaves
both in that range. On the half, quarter, cell, and offset boxes the
sum of absolute face integrals is $10^{-2}$ to $10^{-1}$, and
`relative` is at most about $10^{-14}$. The full-period denominator
is itself about $10^{-16}$.

Grid $\|\nabla\cdot u\|_\infty$ at $t=1.2$ is
$1.9\times 10^{-13}$, $3.9\times 10^{-13}$, and
$1.0\times 10^{-12}$ at $N=128$, $256$, and $512$. It grows
with $N$ and stays the same size at half the step
($4.2\times 10^{-13}$ at $N=256$). The modal amplitude of the
odd-multiplier divergence is at most $2.2\times 10^{-14}$.
`domain_div` is 0 on every row. These are the FFT round-trip of a
zero spectrum.

The quarter box is a fixed fraction of the square. Its trapezoid at
$t=0.8$ is $7.4\times 10^{-5}$, $1.8\times 10^{-5}$, and
$4.6\times 10^{-6}$, a factor of about 4 each time $N$ doubles.
At $t=1.2$ it is $2.1\times 10^{-5}$, $5.0\times 10^{-6}$, and
$1.3\times 10^{-6}$. The one-cell trapezoid at $t=1.2$ is
$-3.2\times 10^{-6}$, $-2.4\times 10^{-7}$, and
$-1.8\times 10^{-8}$. The spectral `net` on those boxes stays at
$10^{-15}$ or smaller. The $N=256$ half-step reproduces the
quarter and cell trapezoids to the printed digits. The offset corners
miss the nodes, so `trap_net` is blank there; the spectral `|net|`
on that box is at most $1.8\times 10^{-15}$.

Mean $\omega$ stays at the grid value of the sampled shear,
$-5.7\times 10^{-7}$, $-2.9\times 10^{-7}$, and
$-1.4\times 10^{-7}$. The drift from the initial mean is at most
about $5\times 10^{-14}$. Kinetic energy at $t=1.2$ is
$0.425162$. Enstrophy is $32.1946$ at $N=128$ and $32.1923$
at $N=256$ and $N=512$, the scalars from the spatial study.
Halving $\Delta t$ at $N=256$ changes that kinetic energy by
$3\times 10^{-12}$. Speed CFL on the finished rows is at most
$0.58$.

Taylor–Green uses $N=32$ and $N=64$, $\nu=0.1$,
$\Delta t=0.05$, $t=0$ and $t=1$, on $[0,2\pi]^2$.

```bash
./examples/ns2d_vorticity/ns2d_flux_balance \
  --case taylor \
  --outdir /scratch/project_462001519/juaho/ns2d-226/taylor
```

Every box has `|net|` at most $2.2\times 10^{-16}$ and `|imag|` at
most $2.8\times 10^{-17}$. $\|\nabla\cdot u\|_\infty$ is
$3\times 10^{-15}$ to $1.0\times 10^{-14}$. `domain_div` is 0.
Mean drift is at most $1\times 10^{-17}$. Kinetic energy at
$t=1$ differs from $\tfrac14 e^{-0.4}$ by about $10^{-15}$.
The on-node trapezoids on this field are roundoff as well: a mode
with $|k_x|=|k_y|$ cancels under this trapezoid.

Two-mode uses $N=64$, $\nu=0.05$, $\Delta t=0.01$, $t=0$ and
$t=0.2$.

```bash
./examples/ns2d_vorticity/ns2d_flux_balance \
  --case two-mode \
  --outdir /scratch/project_462001519/juaho/ns2d-226/two-mode
```

Spectral `|net|` is at most $2.2\times 10^{-16}$,
$\|\nabla\cdot u\|_\infty$ is about $5\times 10^{-15}$, and the
mean drift at $t=0.2$ is $3\times 10^{-18}$. At $t=0.2$ the
quarter trapezoid is $4.4\times 10^{-11}$ and the one-cell
trapezoid is $-1.2\times 10^{-6}$. The initial field is modes
$(1,1)$ and $(2,2)$, which cancel under the trapezoid. The step
produces unequal wavenumbers, and the trapezoid moves while the
spectral flux stays at roundoff.

The Catch2 checks cover the analytic flux and the curl. A cosine
velocity matches the analytic partial-box flux to $10^{-12}$.
Taylor–Green spectral flux on several boxes is within $10^{-9}$.
Mode $(3,1)$ keeps that flux at roundoff on the index box
$\{2,\ldots,7\}\times\{1,\ldots,4\}$, and the trapezoid there
exceeds $10^{-4}$.

### Preserved ETD1 failure (do not treat as the #21 gate)

The first PR #22 runs used ETD1 (forward Euler on the Jacobian),
$\rho=30$ on $[0,2\pi]^2$ (layer $2\pi$ times too thin), and
post-product-only 2/3 masking. Those blew up: $N=128$, $\nu=10^{-4}$
NaN by $t=4$; $N=256$, $\nu=5\times10^{-4}$ NaN by $t=4.5$ at
CFL $0.15$. Raw files remain at
`/scratch/project_462001519/juaho/ns2d-21/`. That failure was a
numerical-method artefact, not evidence that vorticity–streamfunction
NS is a poor fit.

## Decision gate (issue #21)

After the three review blockers:

The corrected CPU solver is a **credible 2-D incompressible core**:
Taylor–Green is spectrally exact, IFRK4 has nonlinear order well above
one, 2/3 projection passes an aliasing test, and the standard
Minion–Brown shear is grid-converged at 256²/512² through roll-up
without blow-up.

Issue #23 (below) is the MHD coupling experiment. CHNS is still a
separate later choice.

Still unverified for NS: HIP; multi-rank; ParaView inspection of VTK;
vortex *merger* as a distinct later-time event beyond the $t=0.8$
roll-up / $t=1.6$ persistence already integrated.

## 2-D incompressible visco-resistive MHD (issue #23)

Not Strauss (1976) reduced MHD (no guide field, no parallel dynamics).
Literature for this experiment: Orszag & Tang, JFM 90 (1979);
Pouquet, JFM 88 (1978).

### Model and signs

$$
\mathbf u=(\partial_y\phi,-\partial_x\phi),\quad
\omega=-\nabla^2\phi,\quad
\mathbf B=(\partial_y a,-\partial_x a),\quad
j=-\nabla^2 a,
$$

$$
\partial_t\omega+\mathbf u\cdot\nabla\omega
=\mathbf B\cdot\nabla j+\nu\nabla^2\omega,
\qquad
\partial_t a+\mathbf u\cdot\nabla a=\eta\nabla^2 a.
$$

The Lorentz term is **$+B\cdot\nabla j$**. A test that flips the sign
fails on an Alfvénic two-mode state ($N_\omega$ jumps from
$<10^{-10}$ to $>1$).

Shared with NS: `SpectralPlane` Poisson, odd-$k$ derivatives, state
2/3 projection, IFRK4. Prognostic fields are $\omega$ (stack field)
and $a$ (second inbox `Field`). Density and $\mu_0$ are 1.

### Diagnostics (2-D means)

| Name | Normalization |
|------|----------------|
| `ke` | $\frac12\langle\Vert u\Vert ^2\rangle$ |
| `me` | $\frac12\langle\Vert B\Vert ^2\rangle$ |
| `energy` | `ke+me` |
| `cross_helicity` | $\langle u\cdot B\rangle$ |
| `a2` | $\frac12\langle(a-\langle a\rangle)^2\rangle$, with $\hat a(0)=0$ |
| `enstrophy` | $\frac12\langle\omega^2\rangle$ so $\langle\omega^2\rangle=2Z$ |
| `mean_sq_j` | $\langle j^2\rangle$ |
| `dissipation` $D$ | $\nu\langle\omega^2\rangle+\eta\langle j^2\rangle$ |
| `budget_residual` | $(E_n-E_{n-1})/\Delta t_{\mathrm{diag}}+\frac12(D_n+D_{n-1})$ |
| `cfl_nominal` | $\Delta t/\Delta x$ (what `--cfl` sets) |
| `cfl_ub` | $\Delta t\,\max_i(\vert u_i\vert ,\vert B_i\vert )/\Delta x$ (deprecated) |
| `cfl_elsasser` | $\Delta t\,\max_i\vert z^\pm_i\vert /\Delta x$ |
| `cfl_elsasser_mag` | $\Delta t\,\max\Vert z^\pm\Vert /\Delta x$ (Euclidean) |
| `cfl_elsasser_sum` | $\Delta t\,\max(\vert z_x^\pm\vert /\Delta x+\vert z_y^\pm\vert /\Delta y)$ |

`--cfl` remains a **nominal fixed-$\Delta t$ selector**:
$\Delta t=\mathrm{CFL}_{\mathrm{nom}}\Delta x$ (characteristic speed 1).
The fail-closed measured CFL is the 2-D Elsasser sum
`cfl_elsasser_sum`. Component-max and Euclidean values are logged.
The driver aborts if `cfl_elsasser_sum > 2`. No adaptive timestepping.
On a square grid a diagonal $z=(1,1)$ makes the sum bound twice the
component-max bound.

$A_2$ is gauge-safe: the DC mode of $a$ is zeroed after every
stage, and the reported invariant uses the variance of $a$.

### Commands

```bash
# force-free magnetic eigenmode (CLI verify)
./examples/ns2d_vorticity/mhd2d --verify --N 16 --steps 8 \
  --dt 0.05 --nu 0.1 --eta 0.1

# incompressible Orszag–Tang, Pm=1, peak-current window at nu=eta=0.005
./examples/ns2d_vorticity/mhd2d --case orszag_tang --N 256 \
  --steps 255 --cfl 0.4 --nu 0.005 --eta 0.005 --diag 4 --dump 4 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak

# doubly-periodic island coalescence (issue #113). Frozen seed;
# do not retune epsilon with eta. Stage-0 protocol:
# examples/ns2d_vorticity/COALESCENCE.md
./examples/ns2d_vorticity/mhd2d --case island_coalescence --N 256 \
  --cfl 0.4 --nu 0.01 --eta 0.01 --steps 4074 --diag 40 --dump 40 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic256_nu001_cfl04

./examples/ns2d_vorticity/mhd2d --case orszag_tang --N 512 \
  --steps 509 --cfl 0.4 --nu 0.005 --eta 0.005 --diag 8 --dump 8 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/ot512_nu0005_peak

# matched hydro control (same u, a=0, same nu)
./examples/ns2d_vorticity/mhd2d --case hydro_control --N 256 \
  --steps 255 --cfl 0.4 --nu 0.005 --eta 0.005 --diag 4 \
  --outdir /scratch/project_462001519/juaho/mhd2d-23/hydro256_nu0005_peak
```

`--diag` writes CSV/stdout without requiring a VTK dump. Field dumps
are Fortran-order `double` bricks (`a_%04d.bin`, `j_%04d.bin`,
`omega_%04d.bin`) plus VTK. Overlay $j$ with $a$ contours; do
**not** call that reconnection.

Common-band comparison and rendering (cray-python 3.11 + numpy;
matplotlib for figures):

```bash
python3 examples/ns2d_vorticity/scripts/compare_mhd_fields.py \
  --coarse-dir /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak \
  --fine-dir   /scratch/project_462001519/juaho/mhd2d-23/ot512_nu0005_peak \
  --coarse-n 256 --fine-n 512 --coarse-inc 228 --fine-inc 456

python3 examples/ns2d_vorticity/scripts/render_mhd_frames.py \
  --dir /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak \
  --n 256 --inc 228 \
  --out /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak/frame_0228.png

python3 examples/ns2d_vorticity/scripts/plot_mhd_diagnostics.py \
  --mhd /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak/diagnostics.csv \
  --hydro /scratch/project_462001519/juaho/mhd2d-23/hydro256_nu0005_peak/diagnostics.csv \
  --out /scratch/project_462001519/juaho/mhd2d-23/ot256_nu0005_peak/mhd_vs_hydro.png
```

### Verification (Catch2 `mhd2d-cpu`, 13 cases)

- $a=0$ MHD matches NS on two-mode IC ($L^\infty<10^{-11}$).
- Force-free $a=\sin x\sin y$, $u=0$: $a(t)=a(0)e^{-2\eta t}$
  to $10^{-11}$; velocity stays 0.
- Alfvénic $u=B$: $N_\omega\sim 0$ with Lorentz $+$; $N_\omega>1$
  if the sign is flipped.
- Elsasser: OT at $t=0$ has $\max_i|z^\pm_i|=2$, so
  `cfl_elsasser = 2 cfl_nominal`. A diagonal $z=(1,1)$ has
  component-max 1 and L1 sum 2; `cfl_elsasser_sum` is strictly
  larger. The driver fails closed on the sum.
- Gauge: $a+\mathrm{const}$ is projected to $\langle a\rangle=0$
  and $A_2=0.3125$ for the OT flux.
- Ideal OT, $N=32$, $T=0.2$, $\nu=\eta=0$. Relative drift of
  $E$, $H_c$ and $A_2$ (high-precision CSV):

  | Time step $\Delta t$ | Energy change $\delta E/E$ | Cross-helicity change $\delta H_c/H_c$ | Potential-square change $\delta A_2/A_2$ |
  |-------------:|---------------:|-------------------:|-------------------:|
  | 0.02 | $1.18\times10^{-9}$ | $2.44\times10^{-9}$ | $5.60\times10^{-10}$ |
  | 0.01 | $3.54\times10^{-11}$ | $8.19\times10^{-11}$ | $2.25\times10^{-11}$ |
  | 0.005 | $1.00\times10^{-12}$ | $2.91\times10^{-12}$ | $1.02\times10^{-12}$ |
  | 0.0025 | $2.15\times10^{-14}$ | $1.13\times10^{-13}$ | $4.90\times10^{-14}$ |

  Halving $\Delta t$ from 0.02 to 0.01 reduces the three drifts by
  $\approx 25$–$33$. That is clearly higher than first order
  (ratio 2). The window is too close to roundoff on the last halving
  to claim a formal RK4/fourth-order rate.
- Trapezoidal budget residual drops as the diagnostic interval is
  coarsened in reverse: every-step residual $<5\times10^{-3}$ and
  smaller than the 4-step and 8-step residuals.
- `restrict_hat_by_k` recovers a shared trigonometric polynomial
  from $N=32$ onto $N=16$.
- $\nabla\cdot u$ and $\nabla\cdot B$ at roundoff.

### Orszag–Tang ladder (incompressible, $[0,2\pi]^2$)

$\phi=\cos x+\cos y$, $a=\frac12\cos 2x+\cos y$.
CPU HeFFTe 2.3.0 FFTW/Milan, 1 rank. Raw:
`/scratch/project_462001519/juaho/mhd2d-23/`.
Nominal `--cfl 0.4` so $\Delta t=0.4\cdot 2\pi/N$. Measured Elsasser
CFL starts at 0.8.

**$P_m=1$, $\nu=\eta=0.02$, $t=2$:** 128² and 256² agree to six
digits on $E$, KE, ME, $\max\|j\|$. Finite, `div` $\sim10^{-14}$.

| Grid size $N$ | Energy $E(t=2)$ | KE | ME | Peak current $\max\Vert j\Vert $ | Peak vorticity $\max\Vert \omega\Vert $ |
|------:|----------:|---:|---:|-------------:|------------------:|
| 128 | 0.760592 | 0.219881 | 0.540711 | 15.752 | 4.322 |
| 256 | 0.760592 | 0.219881 | 0.540711 | 15.752 | 4.322 |

**Matched hydro control.** The OT velocity has vanishing Jacobian, so
hydro is an exact decaying eigenmode. At $\nu=0.005$, $N=256$,
$t=2.50$:

| Model | KE | ME | Total energy $E$ | Peak vorticity $\max\Vert \omega\Vert $ | Peak current $\max\Vert j\Vert $ |
|---|---:|---:|---:|---:|---:|
| hydro | 0.4876 | 0 | 0.4876 | 1.975 | 0 |
| MHD | 0.2299 | 0.6219 | 0.8518 | 13.58 | 31.62 |

Hydro KE decays $0.50\to 0.488$. MHD transfers kinetic to magnetic
energy and forms current sheets. Figure:
`ot256_nu0005_peak/mhd_vs_hydro.png`.

**$\nu=\eta=0.005$, peak current from the time series.** $t=1.5$
is still on the rise. On 256² with `--diag 4` the global
$\max\|j\|$ peaks at $t=2.238$ ($j=38.237$), with an earlier
shoulder at $t=1.924$ ($j=35.751$). 512² agrees in those
scalars to five digits. Measured Elsasser CFL at the peak is 1.10
(nominal 0.4). Extending 256² toward $t=3$ **fails closed** at
step 312 ($t=3.063$): `cfl_elsasser` jumps $1.28\to 8.71$ after
$\max\|j\|$ explodes. The defensible window is $t\le 2.5$.

**Field-level 256² vs 512²** (restrict 512 r2c hats onto the 256
integer-$k$ lattice, then relative $L^2$):

| Time $t$ | Potential error $\Vert a\Vert _{2,\mathrm{rel}}$ | Current error $\Vert j\Vert _{2,\mathrm{rel}}$ | Vorticity error $\Vert \omega\Vert _{2,\mathrm{rel}}$ | ME-spectrum | $j$-spectrum |
|------:|---------------------------:|---------------------------:|-------------------------------:|------------:|---------------:|
| 1.492 | $4.4\times10^{-8}$ | $9.6\times10^{-5}$ | $5.2\times10^{-5}$ | $2.3\times10^{-9}$ | $6.4\times10^{-8}$ |
| 1.924 | $5.6\times10^{-7}$ | $1.0\times10^{-3}$ | $4.9\times10^{-4}$ | $4.4\times10^{-9}$ | $1.3\times10^{-6}$ |
| 2.238 | $5.1\times10^{-7}$ | $8.7\times10^{-4}$ | $4.3\times10^{-4}$ | $6.5\times10^{-9}$ | $9.9\times10^{-7}$ |

At the $\max\|j\|$ time the operational FWHM of $|j|$ is
$0.123$ (5 cells) on 256² and $0.135$ (11 cells) on 512²,
about 10% apart in physical units. Two resolutions do not
support a formal thickness-order. 256² and 512² frames at
$t=2.24$ are visually the same: current sheets, deformed flux
contours, no claim of reconnection or plasmoids. Rendered:
`ot256_nu0005_peak/frame_0228.png`,
`ot512_nu0005_peak/frame_0456.png`.

**Exploratory $\nu=\eta=0.0025$.** The previous 512² `--cfl 0.4`
blow-up at $t=2.474$ is **timestep-limited**. Repeating 512²
at `--cfl 0.2` reaches $t=2.50$ with `cfl_elsasser_sum=0.93`,
$E=0.902$, $\max\|j\|=49.91$. Before the old failure, 512²
`--cfl 0.4` vs `0.2` agree to $\|j\|_{2,\mathrm{rel}}=2\times10^{-7}$.
256² vs 512² at the same smaller dt still differ by
$\|j\|_{2,\mathrm{rel}}=1.4\times10^{-2}$ at $t=1.885$
(FWHM $0.074$ vs $0.061$). So: late-time crash was $\Delta t$;
256² remains spatially under-resolved. 512²/`cfl 0.2` is a usable
exploratory run, not a second-resolution gate. 1024² was **not**
run in this #23 peak-current ladder. Issue #38 later admitted 1024²
as the spatial gate for $\eta=0.0025$ on $t\le 0.80$
([`SCALING.md`](SCALING.md)). That run is not a #23 peak-current
result. Do not go to still lower $\eta$ from this prototype.

### Topology and candidate reconnection observables (analysis only)

Offline scripts, not part of the time stepper:

```bash
python3 examples/ns2d_vorticity/scripts/find_mhd_critical_points.py --self-test
python3 examples/ns2d_vorticity/scripts/find_mhd_critical_points.py \
  --dir .../ot256_nu0005_peak --n 256 --inc 228
python3 examples/ns2d_vorticity/scripts/mhd_reconnection_observables.py \
  --dir .../ot256_nu0005_peak --n 256 --eta 0.005
```

$B=(\partial_y a,-\partial_x a)$, so magnetic nulls are critical
points of $a$. Saddles of $a$ are candidate X-points; extrema
are candidate O-points. The finder recovers the four exact X- and
O-points of $a=\sin x\sin y$.

On incompressible OT, $t=0$ has four X and four O as expected.
Through the peak-current window the flux extrema persist. Hessian
X/O classification of the high-$|j|$ structure is **not robust**
on a 5-cell sheet (256² reports 0 X at $t=2.24$; 512² reports
one X at a neighbouring extremum). Do not treat a $|j|$ peak as
an X-point.

For this sign convention Faraday is $E_z=-\partial_t a$. The
induction equation is $\partial_t a+u\cdot\nabla a=\eta\nabla^2 a
=-\eta j$. At a rest null that reduces to $E_z=\eta j$. On the
converged $\nu=\eta=0.005$ 256² dumps, $E_z$ at the
instantaneous $\max|j|$ site tracks $\eta j$ through the
current-sheet peak ($t=2.238$: $E_z\approx-0.191$,
$\eta j\approx-0.191$). That pair is the mathematically
meaningful local diagnostic. Competing quantities for a later
issue, not a rate yet:

* $E_z=-\partial_t a$ at $\max|j|$ or at a tracked X-point;
* $\eta j$ at the same site;
* $a_{\mathrm{O,max}}-a_{\mathrm{O,min}}$ (flux between extrema,
  independent of X classification);
* operational FWHM and aspect ratio of $|j|$.

Global Lundquist $S=L V_A/\eta$ with $L=2\pi$, $V_A\sim 1$
is $\sim 1.3\times10^3$ at $\eta=0.005$ and $\sim 2.5\times10^3$
at $\eta=0.0025$. Both sit below the usual $S_c\sim10^4$
plasmoid threshold. This prototype is a moderate-$S$ current-sheet
problem, not a plasmoid campaign.

### Literature for the next issue

Primary 2-D incompressible / resistive-MHD reconnection, not a
general MHD survey:

* Orszag & Tang, JFM 90 (1979) — incompressible OT vortex.
* Pouquet, JFM 88 (1978) — 2-D MHD invariants.
* Parker (1957), Sweet (1958) — Sweet–Parker rate
  $V_{\mathrm{rec}}/V_A\sim S^{-1/2}$.
* Biskamp, Phys. Fluids 29 (1986) — Petschek X-point collapses
  to an SP sheet in uniform-resistivity MHD.
* Loureiro, Schekochihin & Cowley, Phys. Plasmas 14 (2007) —
  plasmoid instability of high-$S$ SP sheets.
* Huang & Bhattacharjee, Phys. Plasmas 17 (2010) — high-$S$
  plasmoid-mediated scaling.
* García Morillo & Alexakis, JFM 1007 (2025) R3
  (arXiv:2406.08951) — OT; under-resolved sheets produce
  apparent plasmoids; well-resolved runs with $\delta/h\gtrsim 10$
  showed none up to $S\sim5\times10^5$.
* Vicentin, Kowal, de Gouveia Dal Pino & Lazarian,
  arXiv:2510.01060 — $\delta/h>10$ as a resolution gate;
  $V_{\mathrm{rec}}\sim S^{-1/2}$ then $\sim S^{-1/3}$; still
  resistivity-dependent.
* Baty, arXiv:2604.02065 — physical vs spurious plasmoids in
  OT, using current/enstrophy spectra.

Issue #26 measured moderate-$S$ resistive reconnection on this
prototype. Issue #38 asked how that rate and the local sheet scale
with $\eta$. Neither question is reopened here. Conference
figures: [`VISUALS.md`](VISUALS.md).

### Decision gate (issue #23)

OpenPFC now has a verified and spatially converged 2-D
incompressible MHD research prototype whose magnetic coupling
generates robust current-sheet dynamics beyond the matched
hydrodynamic control.

That statement is limited to $P_m=1$, $\nu=\eta=0.005$,
$t\le 2.5$, 256²/512² common-band fields. It does **not**
cover a spatially converged $\nu=\eta=0.0025$ ladder,
$t>2.5$, reconnection, or plasmoids.

Issue #26 (`RECONNECTION.md`) is the quantitative reconnection /
topology experiment on this prototype. It does not change the
solver. The merged #27 decision is **slow, converged persistent-X
reconnection for $t\le 0.80$**. Peak-current topology change is
not demonstrated.

Issue #38 (`SCALING.md`) is the follow-on resistive scaling and
local current-sheet geometry experiment. It reuses the #26/#27
topology machinery and does not change the solver. Conference
figures and animations for those accepted windows:
[`VISUALS.md`](VISUALS.md).

Issue #113 (`COALESCENCE.md`) is the controlled island-coalescence
geometry test of that OT interpretation. Stage 0 freezes the
perturbation and the flux-progress event statistic at
$\eta=0.01$. It does not change the solver.

HIP is still blocked by the same pointwise `SpectralETDOps` pipeline.
