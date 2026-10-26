<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Magnetic-island coalescence (issue #113)

Stage-0 qualification of a doubly-periodic coalescence control for
the accepted Orszag–Tang persistent-X result. Parent research work
order: .

This note freezes the benchmark, the perturbation, and — after the
$\eta=0.01$ run — the cross-$\eta$ event statistic. It does
**not** run $\eta=0.005$, fit a scaling exponent, add noise, hunt
plasmoids, or add Hall/kinetic physics.

The solver is unchanged. Topology, flux-budget, and local-sheet
definitions are the accepted #26/#38 machinery.

## Literature-matched field

Ng & Ragunathan (arXiv:1106.0521) use, on the unit square,

$$
A(x,y,0)=\bar A\sin(2\pi x)\sin(2\pi y),\qquad \bar A=0.4.
$$

On OpenPFC's $[0,2\pi]^2$ domain the coordinate-equivalent flux is

$$
a(x,y,0)=0.4\sin x\sin y.
$$

Then $j=-\nabla^2 a=2a$, so $\mathbf B\cdot\nabla j=0$. With
$u=0$ the magnetic field is an exact static state of the current
vorticity formulation; it decays only as the $k^2=2$ resistive
eigenmode $a(t)=a(0)e^{-2\eta t}$.

Critical points of $a$:

| kind | locations | $a$ |
|------|-----------|------:|
| X | $(0,0)$, $(\pi,0)$, $(0,\pi)$, $(\pi,\pi)$ | $0$ |
| $\mathrm{O}_{\max}$ | $(\pi/2,\pi/2)$, $(3\pi/2,3\pi/2)$ | $+0.4$ |
| $\mathrm{O}_{\min}$ | $(\pi/2,3\pi/2)$, $(3\pi/2,\pi/2)$ | $-0.4$ |

Like-signed islands sit on the diagonals. Magnetic separatrices of
the $a=0$ network are the coordinate axes, not the Morse
diagonals (already tested on $a=\sin x\sin y$).

Tracked **primary family**: the X nearest $(\pi,\pi)$ and the
enclosed $\mathrm{O}_{\max}$ nearest $(\pi/2,\pi/2)$. At $t=0$,
$F=a_O-a_X=0.4$. The negative pair that coalesces toward
$(0,0)$ is the symmetry image and is not a second experiment.

## Frozen perturbation

Ng & Ragunathan state that a small flow triggers the coalescence
instability but do not give its form in the short paper. Random
noise is forbidden in the accepted baseline.

The smallest deterministic, divergence-free, low-wavenumber
streamfunction that displaces like-signed islands toward one
another is the $k=1$ diagonal strain

$$
\phi=\varepsilon(\cos x-\cos y),\qquad
\varepsilon=0.01\ \text{(frozen)}.
$$

Then

$$
\mathbf u=(\varepsilon\sin y,\ \varepsilon\sin x),\qquad
\omega=\varepsilon(\cos x-\cos y).
$$

Direction check, $\varepsilon>0$:

- $\mathrm{O}_{\max}(\pi/2,\pi/2)$: $\mathbf u=\varepsilon(1,1)$
  toward $(\pi,\pi)$.
- $\mathrm{O}_{\max}(3\pi/2,3\pi/2)$: $\mathbf u=\varepsilon(-1,-1)$
  toward $(\pi,\pi)$.
- $\mathrm{O}_{\min}(\pi/2,3\pi/2)$: $\mathbf u=\varepsilon(-1,1)$
  toward $(0,0)$.
- $\mathrm{O}_{\min}(3\pi/2,\pi/2)$: $\mathbf u=\varepsilon(1,-1)$
  toward $(0,0)$.
- All four X-points are stagnation points.

This is not $\phi\propto a$ (which stagnates at the O-points) and
not a single Fourier mode (which only translates the lattice). The
shape and amplitude are frozen during Stage 0 and are **not**
retuned with $\eta$. Constants live in
`../flow/ns2d/mhd_cases.hpp` as `coalescence_abar` and
`coalescence_eps`.

Stage-0 parameters: $\nu=\eta=0.01$, $P_m=1$.

## Diagnostics (unchanged definitions)

Reuse `scripts/mhd_topology.py`, `island_flux_budget.py`,
`mhd_sheet_geometry.py`, and `sheet_scaling.py`:

- Fourier sub-grid X/O tracker;
- magnetic separatrix / enclosure;
- $F=a_O-a_X$;
- $E_{z,X}=-\mathrm{d}a_X/\mathrm{d}t$;
- $\eta j_X$ Ohm check;
- X/O flux budget;
- signed-current-ridge sheet frame;
- connected-component $\delta$ and $L$;
- spectral $B_{\mathrm{up}}$;
- $S_{\mathrm{local}}=L B_{\mathrm{up}}/\eta$;
- $R=|E_{z,X}|/B_{\mathrm{up}}^2$;
- Hessian-conditioning and Elsasser-CFL fail-closed gates.

Do not use $\max|j|$ as a reconnection proxy. Pass
`--family coalescence` so the tracker seeds $(\pi,\pi)$ rather
than the OT $a\approx-0.5$ island.

## Cross-$\eta$ event statistic (corrected and frozen before Stage 1)

The first Stage-0 draft used raw island-flux progress

$$
p_F=(F(0)-F(t))/|F(0)|,\qquad F=a_O-a_X.
$$

That is **not** a valid cross-$\eta$ event coordinate for this
benchmark. The unperturbed magnetic field is a resistive eigenmode, so
$a_O$ and therefore $F$ decay even when the X-point electric field
is zero. At $\eta=0.01$, pure eigenmode decay alone gives
$1-e^{-2\eta t}\approx0.165$ by the first \`sheet_ok\` time
$t=9.032$. The previously proposed $p_F\in[0.30,0.60]$ bracket
is therefore **withdrawn as a cross-resistivity comparator before any
$\eta=0.005$ run**.

The corrected progress coordinate is cumulative flux transferred
through the tracked X-point,

$$
\Psi_X(t)=\int_0^t E_z(X,t')\,dt'=a_X(0)-a_X(t),
$$

and

$$
p_X(t)=s_X\frac{\Psi_X(t)}{|F(0)|},
$$

where $s_X$ is the sign of $E_z(X)$ at the first finite-sheet
dump. A pure O-mode resistive decay with $a_X=0$ therefore has
$p_X=0$ even while $F$ changes substantially. A synthetic test
requires exactly this control.

The frozen Stage-1 comparator is

$$
p_X\in[0.05,0.10].
$$

These numbers were fixed after identifying the diffusion contamination
in the original Stage-0 statistic and **before** any $\eta=0.005$
run. They represent 5--10% of the initial island flux transferred
through the X-point, not 5--10% of total island-flux decay. The event
is additionally required to be entirely \`sheet_ok\`; if the upper
bound is not reached after a finite sheet forms, the run fails closed.

Rebound/sloshing is detected from a persistent sign reversal of
$E_z(X)$, not from $dF/dt$, because the latter also contains
O-point diffusion.

Later runs report on this frozen X-flux interval:

| quantity | definition |
|----------|------------|
| mean $R$ | time mean of $\vert E_{z,X}\vert /B_{\mathrm{up}}^2$ |
| mean $S_{\mathrm{local}}$ | time mean of $L B_{\mathrm{up}}/\eta$ |
| mean $\delta/L$ | time mean of connected-component aspect |
| mean $j_X$ | time mean of spectral $j$ at the tracked X |
| mean $L$ | time mean of connected-component length |
| mean $B_{\mathrm{up}}$ | time mean of spectral upstream field |
| flux pile-up | $B_{\mathrm{up}}^{\max}/B_{\mathrm{up}}(t_{\mathrm{first}})$ on the event window |
| sloshing sentinel | first persistent $E_z(X)$ reversal, or none |

Peak reconnection rate remains secondary. A two-point log--log slope
is out of scope for Stage 1.

### Frozen numbers after the correction

| symbol | value | status |
|--------|------:|--------|
| $\varepsilon$ | $0.01$ | frozen before any run |
| $\bar A$ | $0.4$ | literature match |
| $p_{X,\mathrm{lo}}$ | $0.05$ | frozen before $\eta=0.005$ |
| $p_{X,\mathrm{hi}}$ | $0.10$ | frozen before $\eta=0.005$ |
| raw $p_F\in[0.30,0.60]$ | withdrawn | diffusion-contaminated; history only |
| event clock window | n/a | not used as the comparator |

## Qualification ladder

Start at $128^2$ and $256^2$. Timestep refinement at $256^2$
(`--cfl 0.4` vs `0.2`). Increase $N$ only if a concrete
convergence failure requires it.

Demonstrate:

1. analytic $t=0$ topology;
2. intended islands approach and coalesce;
3. one unambiguous primary X/O family;
4. X/O topology survives refinement;
5. $E_{z,X}\approx\eta j_X$;
6. island-flux budget closes;
7. $\delta$, $L$, and $B_{\mathrm{up}}$ finite and
   `sheet_ok`;
8. the same event spatially and timestep converged;
9. representative $j+a$ contours + X/O frames preserved.

## Commands

CPU HeFFTe / LUMI-C. Large I/O under scratch.

```bash
# 128^2 scout, eta=nu=0.01, nominal CFL 0.4
mhd2d --case island_coalescence --N 128 --cfl 0.4 \
  --nu 0.01 --eta 0.01 --steps 2038 --diag 20 --dump 20 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic128_nu001_cfl04

# 256^2, CFL 0.4
mhd2d --case island_coalescence --N 256 --cfl 0.4 \
  --nu 0.01 --eta 0.01 --steps 4074 --diag 40 --dump 40 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic256_nu001_cfl04

# 256^2 timestep refinement, CFL 0.2
mhd2d --case island_coalescence --N 256 --cfl 0.2 \
  --nu 0.01 --eta 0.01 --steps 8148 --diag 80 --dump 80 \
  --outdir /scratch/project_462001519/juaho/mhd2d-113/ic256_nu001_cfl02
```

`--steps` above reaches $t\approx 40$ at each nominal CFL. Shorten
only after the scout shows that the first coalescence and any first
rebound both sit well below that time.

```bash
python3 examples/ns2d_vorticity/scripts/island_flux_budget.py \
  --dir .../ic256_nu001_cfl04 --n 256 --eta 0.01 --t-max 40 \
  --family coalescence --json-out .../flux.json

python3 examples/ns2d_vorticity/scripts/sheet_scaling.py \
  --dir .../ic256_nu001_cfl04 --n 256 --eta 0.01 --t-max 40 \
  --family coalescence --json-out .../sheet.json

python3 examples/ns2d_vorticity/scripts/coalescence_event.py \
  --json .../sheet.json --json-out .../event.json

python3 examples/ns2d_vorticity/scripts/plot_mhd_topology.py \
  --dir .../ic256_nu001_cfl04 --n 256 --inc INC \
  --out .../frame.png
```

Slurm: `slurm/coalescence_stage0.sbatch`,
`slurm/coalescence_analyze.sbatch`, and
`slurm/coalescence_field_gate.sbatch`.

## Stage-0 evidence

Parent SHA at run time: `e98d8a87`. Jobs used the dirty
`feat/113-island-coalescence` worktree that this PR records.
Binary:
`/flash/project_462001519/juaho/build/openpfc-113-cpu/examples/ns2d_vorticity/mhd2d`.
Raw dumps:
`/scratch/project_462001519/juaho/mhd2d-113/`.
CPU HeFFTe 2.3.0, LUMI-C `debug`, 8 OpenMP threads.

| job | run | $N$ | CFL | steps | dt |
|----:|-----|------:|----:|------:|---:|
| 22188502 | `ic128_nu001_cfl04` | 128 | 0.4 | 2038 | 0.019635 |
| 22188521 | `ic256_nu001_cfl04` | 256 | 0.4 | 4074 | 0.009817 |
| 22188522 | `ic256_nu001_cfl02` | 256 | 0.2 | 8148 | 0.004909 |
| 22188569 | analyze 256 CFL 0.4 | | | | |
| 22188570 | analyze 256 CFL 0.2 | | | | |

128² topology/sheet analysis ran on login with cray-python 3.11
after job 22188502; 256² analysis is jobs 22188569/22188570.

CTest on compute job 22188495: 6/6 MHD tests passed (`mhd2d-cpu`,
`mhd2d-cli-verify`, `mhd2d-cli-coalescence`). Topology script with
cray-python 3.11: `ALL TOPOLOGY TESTS PASSED`.

### Ladder

1. **t=0 topology.** Four X, two $\mathrm{O}_{\max}$, two
   $\mathrm{O}_{\min}$. Tracked family: X nearest $(\pi,\pi)$,
   enclosed $\mathrm{O}_{\max}$ nearest $(\pi/2,\pi/2)$,
   $F(0)=0.40000$. Analytic test in
   `scripts/test_mhd_topology.py`. Frame:
   `figures/coalescence/ic256_t0.png`.
2. **Islands approach and coalesce.** The $\mathrm{O}_{\max}$
   walks along the diagonal toward the stationary X at
   $(\pi,\pi)$: at $t=0$, $(x_O,y_O)\approx(1.57,1.57)$; at
   first `sheet_ok`, $\approx(1.66,1.66)$; by $t\approx 40$,
   $\approx(2.67,2.67)$. $F: 0.400\to 0.031$ ($p\approx 0.92$).
3. **One primary family.** The tracker holds the $(\pi,\pi)$ X
   and its enclosed $\mathrm{O}_{\max}$ through the run.
   Hessian $\mathrm{cond}(X)$ at $t=0$ is 1.
4. **Topology survives refinement.** At matched times, $F(t)$
   and the event means agree among 128² CFL 0.4, 256² CFL 0.4, and
   256² CFL 0.2 to relative $\sim 10^{-11}$. The X coordinate
   differs by half a cell (stationary saddle at $\pi$); that
   does not move the scalars. No 512² run is required.
5. **Ohm.** On 256² CFL 0.4,
   $\mathrm{rms}(E_{z,X}-\eta j_X)=1.86\times 10^{-6}$.
6. **Flux budget.** Identity
   $\mathrm{rms}(\mathrm{d}F/\mathrm{d}t-(E_{z,X}-E_{z,O}))
   =1.8\times 10^{-17}$. Resistive residual
   $\mathrm{rms}(\mathrm{d}F/\mathrm{d}t-\eta(j_X-j_O))
   =2.0\times 10^{-6}$.
7. **Sheet.** First \`sheet_ok\` at $t=9.032$,
   $\delta\approx 0.45$, $L\approx 1.00$,
   $B_{\mathrm{up}}\approx 0.23$. The old raw-$F$ progress there
   was $p_F\approx0.189$, but that number is retained only as a
   diffusion-contaminated diagnostic. Cross-$\eta$ alignment uses
   $p_X$, defined above.
8. **Scalar/timestep convergence.** The previously reported
   raw-$F$-window means (mean $R=0.1404$,
   $S_{\mathrm{local}}=17.15$, $\delta/L=0.4385$,
   $j_X=-0.4062$, $L=0.9773$, $B_{\mathrm{up}}=0.1767$)
   remain useful Stage-0 diagnostics, but are **withdrawn as the
   cross-resistivity statistic** because the window coordinate was
   diffusion-contaminated. Their 128²/256² and CFL 0.4/0.2 agreement
   remains a scalar/timestep check.
9. **Field-level spatial gate.** Scalar agreement alone is not used to
   admit the spatial step. \`compare_mhd_fields.py\` restricts 256²
   Fourier fields onto the 128² common band at two exactly matched
   times: $t=9.032078879$ (increments 460/920) and
   $t=15.707963268$ (increments 800/1600). The fail-closed
   \`coalescence_field_gate.py\` requires relative $L^2$ errors
   $<10^{-5}$ for $a$ and $<10^{-3}$ for $j$,
   $\omega$, magnetic-energy spectrum, current spectrum and
   enstrophy spectrum. **Stage 1 remains forbidden until this raw-data
   gate is executed and recorded.**
10. **Frames.** \`figures/coalescence/\`: $t=0$, first \`sheet_ok\`,
   mid-transfer and the $F(t)$ / O-path series. Late frames with
   collapsing $B_{\mathrm{up}}$ are not comparison evidence.

$\max|j|$ decays from 0.80 because that peak is O-point current,
not a reconnection proxy.

Re-run `coalescence_event.py` on the existing Stage-0 `sheet.json` files with the corrected $p_X$ definition and execute the common-band field gate before authorizing $\eta=0.005$.

Do not proceed to $\eta=0.005$ in this PR. Do not fit a scaling
exponent.

