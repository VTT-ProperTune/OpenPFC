<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# `higher_order_pfc` — two-mode PFC crystal-selection benchmark

A deliberately **very high-order** phase-field-crystal example, and (`#118`) a
reproducible **crystal-selection experiment**: does a second correlation-
function peak change which 2D lattice symmetry a periodic PFC quench selects,
verified with a real-space structural metric rather than reciprocal-space ring
power alone. The free-energy kernel runs through \((\nabla^2)^4\) (eighth
order in space); conserved dynamics adds one more Laplacian, so the evolution
operator runs through \((\nabla^2)^5\) — **tenth order**.

The high derivative order is not incidental, but it is not the physical story
either: in a spectral code a tenth-order operator is a five-term Horner loop,
while the equivalent real-space stencil would be a wide, awkward,
hard-to-scale object. **High differential order is not high implementation
complexity here.** The physical story is in the next section.

## Problem setup

| Item | Description |
|---|---|
| **Use case** | Crystal-symmetry selection in a periodic 2D solid: does a second correlation-function peak favour square order over the single-mode kernel's triangular default? |
| **Question** | How does adding a second correlation length scale (`q1`, `r1`) change which lattice symmetry a PFC quench selects, and can that be verified in real space, not just from reciprocal-space ring power? |
| **Domain** | 2D periodic, \(128\times128\) cells. Square box \(L_x=L_y=32\pi\) (\(dx=dy=2\pi/8\)) for the noise-seeded and square-seeded cases; a differently-shaped box \(L_x=32\pi,\ L_y=32\pi/\sqrt3\) (\(dx=2\pi/8,\ dy=\pi/(4\sqrt3)\)) for the triangular single-crystal seed. See "Why these box sizes" below. |
| **Boundary conditions** | Periodic on both active axes |
| **Initial condition** | Either `seeded_noise` (decomposition-independent hashed noise, polycrystalline nucleation-like) or `lattice_seed` (an explicit sum of 2–3 plane waves at the target symmetry's reciprocal vectors, a controlled single-crystal seed) |
| **Key parameters** | `eps` (quench depth), `q1`/`r1` (second-peak position/weight), `n_modes` (1 = single-mode, 2 = two-mode); mean density \(\bar\psi\) (see below) |
| **Observable** | Free-energy density after relaxation; reciprocal-space peak amplitudes at \(\|k\|=1\) and \(\|k\|=q_1\) (`pfc::apps::shell_average`); real-space bond-orientational order \(\psi_4\)/\(\psi_6\) (`order_parameter.hpp`) |
| **Model maturity** | numerical verification: **analytical** (kernel/symbol tests, exact-lattice order-parameter tests) · physical completeness: **reduced** (2D, one order parameter, no elastic/thermal coupling) · calibration: **representative** — see "Literature model" below |

## Literature model and mapping to OpenPFC's kernel

**Source.** Kuo-An Wu, Ari Adland, Alain Karma, "Phase-field-crystal model for
fcc ordering," *Phys. Rev. E* **81**, 061601 (2010),
[10.1103/PhysRevE.81.061601](https://doi.org/10.1103/PhysRevE.81.061601)
(arXiv:1001.1349). The paper's dimensionless free-energy density is

\[
  f = \frac\psi2\Bigl[-\epsilon+(\nabla^2+1)^2\bigl((\nabla^2+Q_1^2)^2+R_1\bigr)\Bigr]\psi
      + \frac{\psi^4}{4},
\]

built from **two** correlation-function peaks so that a second family of
reciprocal-lattice vectors, not just the first, can be independently tuned —
their target application is fcc (`Q1 = 2/sqrt(3)`, coupling the \(\langle
111\rangle\) and \(\langle200\rangle\) families), and their Appendix works the
same construction for the 2D square lattice (`Q1 = sqrt(2)`, coupling
\(\langle10\rangle\) and \(\langle11\rangle\)).

**Term-by-term map.** \(\epsilon\leftrightarrow\)`eps`, \(Q_1\leftrightarrow\)`q1`,
\(R_1\leftrightarrow\)`r1`. OpenPFC's
\(\Lambda(\nabla^2)=-\varepsilon+(1+\nabla^2)^2[r_1+(q_1^2+\nabla^2)^2]\)
(`correlation_kernel.hpp`) is the same expression with `r1` and the second
bracket's square written in the other order — addition commutes, so this is
notation only, not a different model. The `q1 = sqrt(2)` (2D square) and
`q1 = 2/sqrt(3)` (3D fcc) values already documented below are exactly the
paper's `Q1` choices, independently of this literature search — that
agreement is what gives the mapping confidence, not just the shared algebraic
form.

**Honesty about what was and was not verified from this machine.** The
functional form and the `Q1` values above were confirmed against the arXiv
preprint (`ar5iv` HTML rendering of 1001.1349), fetched in this session; the
published APS-typeset version was not accessible from this machine, and the
2D-square appendix's solid-phase amplitude free energy came back through that
fetch with rendering artefacts in its sub/superscripts that could not be
cleanly resolved. So: the **reciprocal-space kernel structure and the `Q1`
ratios** are treated as verified against the primary source; the paper's
**quantitative phase diagram and elastic-constant fits** (e.g. their Ni/Fe
parameter sets, or their reported `R1`/`eps` boundary between one-mode and
two-mode stability) were not independently reproduced here and are not
claimed. `eps`, `r1` and \(\bar\psi\) in this benchmark are chosen to
demonstrate the phase-selection *mechanism* the paper's kernel provides, not
fitted to its phase diagram — calibration is **representative**, not
**quantitative**, and every number below is a real measurement from this
repository's own code, not a reproduction of the paper's reported values.

## Free energy and the reciprocal-space kernel

One conserved density \(\psi\):

\[
  F[\psi] = \int \Bigl[\tfrac12\,\psi\,\Lambda(\nabla^2)\,\psi
                       - \tfrac{g}{3}\psi^3 + \tfrac14\psi^4\Bigr]\,d\mathbf r,
  \qquad
  \partial_t\psi = M\nabla^2\frac{\delta F}{\delta\psi}.
\]

The two-mode correlation kernel approximates the first **two** peaks of the
liquid-state direct correlation function:

\[
  \Lambda(\nabla^2) = -\varepsilon
    + (1+\nabla^2)^2\bigl[r_1 + (q_1^2+\nabla^2)^2\bigr].
\]

Writing \(u = k_{\mathrm{lap}} = -|k|^2\) (OpenPFC's Laplacian symbol) and
expanding with \(a=q_1^2\), \(A=r_1+a^2\), \(B=2a\), \(C=1\):

\[
  \Lambda(u) = (A-\varepsilon) + (B+2A)\,u + (C+2B+A)\,u^2
               + (2C+B)\,u^3 + C\,u^4 ,
\]

a **quartic in \(u\)**, i.e. eighth order in \(k\). The conserved evolution
symbol is one power higher:

\[
  L(u) = M\,u\,\Lambda(u)
  \quad\text{(quintic in } u\text{, tenth order in } k\text{)},
  \qquad
  M_{\mathrm{nl}}(u) = M\,u .
\]

Both live in [`correlation_kernel.hpp`](include/higher_order_pfc/correlation_kernel.hpp)
as a small generic `PolynomialInKLap<N>`, not as hand-written \(k^8\) / \(k^{10}\)
special cases.

**Mass conservation is exact, not approximate.** Multiplying by \(u\) shifts
every coefficient up one power, so `L.coeff[0]` is a literal `0.0` and
\(L(k{=}0)\) is bit-exactly zero. The mean density can never drift.

## Band structure

\(\Lambda\) has minima where each factor vanishes: at \(|k|=1\) with depth
\(-\varepsilon\), and at \(|k|=q_1\) with depth \(-\varepsilon+(1-q_1^2)^2 r_1\).
Growth needs \(\Lambda<0\), so \(r_1\) sets how much of the second band survives:

- `r1 = 0` — both minima exactly degenerate at \(-\varepsilon\).
- `r1 > 0` — the \(|k|=1\) peak is the deeper one, by exactly \((1-q_1^2)^2 r_1\).
- `n_modes = 1` — the classical fourth-order kernel \(-\varepsilon+(1+\nabla^2)^2\),
  one band only.

For 2D square ordering use \(q_1=\sqrt2\) (the \((10)\) and \((11)\) families);
for 3D FCC, \(q_1=2/\sqrt3\).

## What the extra terms actually buy you

Both presets run the same seed, box and quench depth for \(t=400\); the only
difference is the kernel. Fraction of spectral power in each band at the end:

| kernel | `r1` | power at \|k\|=1 | power at \|k\|=√2 |
|---|---|---|---|
| single-mode \(k^4\) | — | 94.3 % | 0.0 % |
| two-mode \(k^8\) | 0.00 | 3.8 % | 91.0 % |
| two-mode \(k^8\) | **0.02** | **70.9 %** | **21.0 %** |
| two-mode \(k^8\) | 0.05 | 78.4 % | 13.7 % |
| two-mode \(k^8\) | 0.10 | 86.4 % | 6.5 % |
| two-mode \(k^8\) | 0.20 | 92.4 % | 1.2 % |

Read across: the fourth-order kernel **cannot** put power at \(|k|=\sqrt2\) — it
has no minimum there. The eighth-order kernel opens that band, and \(r_1\) tunes
the balance continuously, converging back onto the single-mode result as it
grows. The \(|k|=\sqrt3\) band stays empty throughout, so the ordering is the
square \((10)+(11)\) family rather than triangular.

`r1 = 0.02` (the shipped `two_mode_square.json`) is the value that actually
populates *both* square families. At `r1 = 0` the degenerate bands compete and
the \(\sqrt2\) family wins outright, because in 2D its ring carries more modes —
mathematically tidy, but not a square lattice. From random noise on a periodic
box the result is **polycrystalline**: several square-ordered grains at
different orientations, not a single crystal.

This table is exactly the complaint `#118` raises: it is a ring-power
inference. Two peaks both carrying modes at \(\|k\|=\sqrt2\) says the pattern
has *that wavelength*, not that it is *square* — the same ring is also
populated by, say, a rotated square grain boundary or a modulated stripe
phase. The next two sections add what a ring cannot give: a real-space
symmetry check, and a reproducible experiment built on both together.

## Real-space order metric

`order_parameter.hpp` adds the check `#118` says is missing: a real-space
bond-orientational order parameter, not another reciprocal-space proxy.
Density peaks are treated as particles; for each one, the near neighbours'
bond angles \(\theta\) feed

\[
  \psi_n(j) = \frac1{N_b(j)}\sum_{k\in\text{neighbours}(j)} e^{in\theta_{jk}} .
\]

\(n=4\) and \(n=6\) are the discriminator: a perfect square lattice (4
neighbours, \(90^\circ\) apart) gives \(|\psi_4|=1,\ \psi_6=0\) **exactly**; a
perfect triangular lattice (6 neighbours, \(60^\circ\) apart) gives the
reverse, \(\psi_4=0,\ |\psi_6|=1\) **exactly** — algebra worked out in the
header comment and checked against analytically constructed lattices in
`tests/test_order_parameter.cpp`, independently of any peak detector. The
neighbour cutoff (`1.3`× the point set's own median nearest-neighbour
distance) sits strictly between both lattices' first shell and second shell,
so it needs no lattice-specific tuning.

Two averages are reported for each harmonic:

- **global** — magnitude of the complex average of \(\psi_n(j)\) over every
  peak. Misoriented grains point in different directions and partially
  cancel, so this number answers "is this one crystal, or many grains" — it
  is what tells a noise-seeded polycrystal from a `lattice_seed` single
  crystal at otherwise identical `(psi4_local, psi6_local)`.
- **local** — average of \(|\psi_n(j)|\), orientation-blind. Answers "is each
  neighbourhood ordered at all", independent of whether grains agree.

Peak detection (`detect_peaks`) is a periodic 8-neighbour local-maximum test
at grid resolution — no sub-pixel refinement, so this module reports
*symmetry* (a normalised ratio), not a calibrated lattice constant; the
reciprocal-space peak positions from `shell_average` already give that.

## Timestepping: why ETD

The \(k^{10}\) term makes this brutally stiff. On the shipped grid
(\(\Delta x = 2\pi/8\), so \(k_{\max}=\pi/\Delta x = 4\)):

\[
  |L(k_{\max})| \approx 7\times10^{5}
  \;\Longrightarrow\;
  \Delta t_{\text{explicit}} = 2/|L| \approx 3\times10^{-6}.
\]

The app runs at \(\Delta t = 0.05\), **four orders of magnitude larger**,
because ETD integrates the linear operator *exactly* — the stiffness lives
entirely in \(e^{L\Delta t}\), which is just a pointwise multiplier. Only the
cubic remainder is treated explicitly, and it is not stiff. A test asserts the
ratio exceeds 1000 and that the run stays bounded.

Dealiasing (Orszag 2/3) is on: \(\psi^3\) triples the highest wave number
present, and without it the aliased content folds straight back onto the
\(k\approx1\) band the kernel is trying to select.

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `eps` | 0.25 | quench depth; the \(\|k\|=1\) minimum sits at \(-\varepsilon\) |
| `q1` | \(\sqrt2\) | second correlation peak |
| `r1` | 0.0 | second-peak offset; 0 = degenerate minima |
| `M` | 1.0 | mobility |
| `g` | 0.0 | cubic coefficient; breaks \(\psi\to-\psi\) |
| `n_modes` | 2 | 1 = classical \(k^4\), 2 = \(k^8\) |

**The mean density is a control parameter, not a detail.** About a nonzero
\(\bar\psi\) the effective rate is \(\sigma(k)=M u\,[\Lambda(u)+n'(\bar\psi)]\)
with \(n'(\psi)=3\psi^2-2g\psi\). At \(\bar\psi=-0.05,\ g=0.5\) that leaves
\(\sigma(|k|{=}1)=0.19\); at \(\bar\psi=-0.15\) only \(0.033\), and ordering
takes roughly six times longer. Conserved dynamics never repairs a mean the
initial condition got wrong, so `seeded_noise` removes its mean by an integer
reduction over global cell indices — the same field on any number of ranks.

`lattice_seed` (`lattice_seed.hpp`) is the other initial condition this
benchmark adds: a controlled single-crystal seed, an explicit sum of 2–3
plane waves at the target lattice's reciprocal vectors (integer grid modes,
so every term is exactly periodic), rather than growing a pattern out of
noise. Two modes at \(90^\circ\) seed a square lattice; three at \(120^\circ\)
seed a triangular one.

## Crystal-selection benchmark (`#118`)

Five 2D runs, all \(128\times128\), `eps = 0.25`, `g = 0.5`, single rank (the
real-space order metric needs the whole grid; see `diagnostics.hpp`).
`diagnostics.csv` on every case reports mean density, free-energy density, the
reciprocal-space peak amplitudes at \(\|k\|=1\) and \(\|k\|=q_1\), and
\((\psi_4,\psi_6)\) every `saveat`.

| case | kernel | IC | box | \(\bar\psi\) | role |
|---|---|---|---|---|---|
| `single_mode_triangular.json` | `n_modes=1` | noise | square | \(-0.15\) | single-mode reference **and** triangular-favouring control (2D single-mode PFC has one band, at \(\|k\|=1\); Elder et al., *Phys. Rev. Lett.* **88**, 245701 (2002), is the classical result that this selects triangular order) |
| `two_mode_square.json` | `n_modes=2`, `q1=√2`, `r1=0.02` | noise | square | \(-0.15\) | two-mode square-favouring, matched box/IC/quench to the row above (issue's "same conditions" comparison) |
| `two_mode_degenerate.json` | `n_modes=2`, `q1=√2`, `r1=0` | noise | square | \(-0.15\) | known-result re-check: degenerate bands, \(\|k\|=\sqrt2\) wins outright — **not** a clean square lattice (see "What the extra terms actually buy you") |
| `single_mode_seed_triangular.json` | `n_modes=1` | `lattice_seed`, 3 modes at \(120^\circ\) | triangular | \(-0.15\) | clean single-crystal triangular check |
| `two_mode_seed_square.json` | `n_modes=2`, `q1=√2`, `r1=0.02` | `lattice_seed`, 2 modes at \(90^\circ\) | square | \(-0.15\) | clean single-crystal square check |

### Measured phase-selection table

Real numbers from the shipped cases, `higher_order_pfc` CPU build, single
rank, LUMI login node (see the PR body for the run log). \(f\) is free-energy
density; \((\psi_4,\psi_6)\) are `(global, local)`, see "Real-space order
metric" above; mean neighbours is the average bond count the order-parameter
cutoff found (4 for square, 6 for triangular, in an ideal lattice).

| case | \(f\): initial \(\to\) final | \(k_{\text{peak}}\) | \(S(\|k\|{=}1)\): initial \(\to\) final | \(S(\|k\|{=}q_1)\): initial \(\to\) final | \(\psi_4\) | \(\psi_6\) | mean nb. | real-space verdict |
|---|---|---|---|---|---|---|---|---|
| `single_mode_triangular` (noise, \(t{:}0\to400\)) | \(0.00970\to0.00585\) | \(1.031\) | \(0.15\to160132\) | \(0.14\to39.5\) | \((0.014,\ 0.094)\) | \((0.335,\ \mathbf{0.852})\) | \(5.87\) | **triangular**, polycrystalline (global ≪ local) |
| `two_mode_square` (noise, \(t{:}0\to400\)) | \(0.2051\to0.0389\) | \(1.031\) | \(0.15\to111849\) | \(0.14\to27574\) | \((0.341,\ \mathbf{0.806})\) | \((0.022,\ 0.247)\) | \(4.04\) | **square**, mostly one orientation (global ≈ 0.4× local, not ≪) |
| `two_mode_degenerate`, `r1=0` (noise, \(t{:}0\to400\)) | \(0.2048\to0.0398\) | \(1.406\approx\sqrt2\) | \(0.15\to3947\) | \(0.14\to156486\) | \((0.005,\ 0.122)\) | \((0.792,\ \mathbf{0.896})\) | \(5.70\) | ring at \(\sqrt2\) wins outright, but real space is **triangular**, confirming the README's "not a square lattice" claim directly rather than by ring-power inference |
| `two_mode_seed_square` (`lattice_seed`, \(t{:}0\to100\)) | \(0.0430\to0.0377\) | \(1.031\) (unchanged) | \(41087\to419780\) | \(\approx0\to66685\) | \((\mathbf{1.000},\ \mathbf{1.000})\) | \((\approx0,\ \approx0)\) | \(4.00\) | **square** preserved exactly; the \(\sqrt2\) band grows in spontaneously on top of the imposed \((10)\) family |
| `single_mode_seed_triangular` (`lattice_seed`, \(t{:}0\to100\)) | \(0.00891\to0.00511\) | \(1.028\) (unchanged) | \(17207\to322036\) | \(\approx0\to\approx0\) | \((\approx0,\ \approx0)\) | \((\mathbf{1.000},\ \mathbf{1.000})\) | \(6.00\) | **triangular** preserved exactly; no \(\sqrt2\) band exists for `n_modes=1` to grow |

Reading down: free-energy density decreases monotonically in every run (the
conserved gradient flow relaxing, as it must) — different kernels have
different absolute \(\Lambda(0)\), so \(f\) is not meaningfully compared
*across* rows, only *within* one. The columns that *are* directly comparable
across kernels are \(\psi_4\) and \(\psi_6\), and they are the ones that
settle the question: the two-mode `q1=√2, r1=0.02` kernel gives real-space
square order (\(\psi_4\) dominant, \(4.04\) mean neighbours), the single-mode
kernel gives real-space triangular order (\(\psi_6\) dominant, \(5.87\)
neighbours) from the *same* box, seed and quench, and the degenerate
`r1=0` kernel — despite putting nearly all reciprocal power on the
\(\sqrt2\) ring, which a ring-power-only report would likely call "square
harmonics" — is real-space **triangular**, exactly as `#118` warns a ring
alone cannot tell you. Both `lattice_seed` runs hold their imposed symmetry
to \(\psi\) at 3+ decimal places over 2000 ETD steps, the cleanest possible
statement that a kernel *stabilises* the phase it is given, independent of
whatever a noise-seeded run's grain structure happens to look like.

What this benchmark does **not** measure: an explicit grain/orientation
*count* for the noise-seeded runs. The global-vs-local \(\psi_n\) gap
(`single_mode_triangular`: \(0.335\) vs \(0.852\); `two_mode_square`:
\(0.341\) vs \(0.806\)) is used as a documented polycrystallinity proxy —
smaller gap means more mutually aligned grains — but no per-grain
segmentation/labelling algorithm was written; that is deferred (see the PR
body).

### Why these box sizes are commensurate

The **square** box (\(L_x=L_y=32\pi\), \(128\) cells, \(dx=2\pi/8\)) puts
\(\|k\|=1\) exactly on grid mode 16 (\(k=2\pi\cdot16/32\pi=1\)) and
\(\|k\|=\sqrt2\) exactly on the diagonal mode \((16,16)\)
(\(k=\sqrt{1^2+1^2}\)) — both correlation peaks land on exact grid
frequencies, so neither is an artifact of interpolation between bins, and
`lattice_seed`'s square modes are `[16,0,0]`/`[0,16,0]`.

The **triangular** box uses a different aspect ratio because a triangular
lattice's primitive cell is oblique: three \(\|k\|=1\) plane waves at
\(0^\circ,120^\circ,240^\circ\), summed, have real-space period \(4\pi\) along
\(x\) and \(4\pi/\sqrt3\) along \(y\) (expand
\(\cos x+2\cos(x/2)\cos(y\sqrt3/2)\) and read off each factor's period).
Replicating that unit cell \(8\times8\) and resolving it at \(128\) cells per
side gives \(L_x=32\pi\) (\(dx=2\pi/8\), same as the square box) and
\(L_y=32\pi/\sqrt3\) (\(dy=\pi/(4\sqrt3)\)) — an anisotropic grid spacing, not
a smaller domain; still \(128\times128\) cells. `lattice_seed`'s triangular
modes, `[16,0,0]`, `[-8,8,0]`, `[-8,-8,0]`, are the integer mode counts that
put all three wavevectors exactly at \(\|k\|=1\) on *this* grid (checked
directly: \(k_x=n_x/16\), \(k_y=n_y\sqrt3/16\) from the grid spacings above,
so e.g. \((-8,8)\to(-\tfrac12,\tfrac{\sqrt3}2)\), magnitude 1).

A square box forced onto the triangular seed would misalign the three plane
waves with the periodic boundary and inject a spurious strain/defect at the
seam — using each lattice's own commensurate box is what keeps the seeded
single-crystal runs a clean test of *whether the kernel preserves the
imposed symmetry*, not of *whether the box fights it*. Conversely, the two
noise-seeded runs sharing the *same* square box (`single_mode_triangular` and
`two_mode_square`) is deliberate: the issue's matched-conditions comparison
needs one box, not each kernel's favourite.

## Running

```bash
# two-mode (k^8) square ordering, 128^2
mpirun -n 4 ./apps/higher_order_pfc/higher_order_pfc \
    ../apps/higher_order_pfc/inputs_json/two_mode_square.json

# same case with the classical k^4 kernel, for comparison
mpirun -n 4 ./apps/higher_order_pfc/higher_order_pfc \
    ../apps/higher_order_pfc/inputs_json/single_mode_triangular.json
```

Create `results/higher_order_pfc/` first. GPU builds also provide
`higher_order_pfc_hip` when rocFFT HeFFTe is on.

```bash
# crystal-selection benchmark, single rank (order metric needs the whole grid)
for case in single_mode_triangular two_mode_square two_mode_degenerate \
            single_mode_seed_triangular two_mode_seed_square; do
  ./apps/higher_order_pfc/higher_order_pfc \
      ../apps/higher_order_pfc/inputs_json/${case}.json
done
# results/higher_order_pfc/<case>_diagnostics.csv: mean_psi, free_energy_density,
# k1, domain_length, k_peak, dominant_wavelength, S_at_1, S_at_q1,
# psi4_global, psi4_local, psi6_global, psi6_local, n_peaks, mean_neighbours
```

### LUMI-G

`tests/two_mode_hip_smoke.json` (64², `t=1`) on `standard-g`:

| run | job | `sum` (hex) | `sumsq` (hex) |
|---|---|---|---|
| CPU, 1 rank | — | `-0x1.333333333332fp+9` | `0x1.70a757850fcb7p+6` |
| HIP, 1 GCD | 21829597 | `-0x1.333333333332fp+9` | `0x1.70a757850fcb4p+6` |
| HIP, 2 GCD | 21829755 | `-0x1.3333333333338p+9` | `0x1.70a757850fcbcp+6` |

`sum` is the conserved quantity: bit-identical between CPU and one GCD, and
within 9 ULP on two GCDs where the MPI reduction order changes. `sumsq` agrees
to 3 and 5 ULP. Both jobs exited `0:0`.

## Tests

`ctest -R higher-order-pfc`. The suite checks the claims above rather than a
golden dump:

- \(\Lambda(u)\) matches the factored analytical form to \(10^{-12}\) over
  \(0\le|k|\le4\), and its five coefficients equal the derived expansion.
- \(L\) is quintic in \(u\) with `coeff[0] == 0.0` bit-exactly, and equals
  \(M u \Lambda(u)\) pointwise.
- Both minima degenerate at \(-\varepsilon\) when \(r_1=0\); the offset is
  exactly \((1-q_1^2)^2 r_1\) when it is not.
- The unstable band is exactly where the kernel is negative.
- The \(k^8\) kernel destabilises \(|k|=\sqrt2\); the \(k^4\) kernel does not.
- An ETD run reproduces \(e^{L(k)t}\) at \(|k|=1\) to \(10^{-6}\) and holds the
  mean to \(10^{-13}\).
- ETD stays bounded at \(\Delta t\) more than 1000× the explicit limit.
- A seeded run selects the preferred band and leaves out-of-band power three
  orders of magnitude below it.

`#118` additions, all still checked against exact or analytical references,
not golden dumps:

- `bond_orientational_order` on an analytically constructed ideal square
  lattice gives \((\psi_4,\psi_6)=(1,0)\) to \(10^{-9}\); an ideal triangular
  lattice gives \((0,1)\) to \(10^{-9}\) — the key test the issue asks for.
- The order parameter is orientation-blind (a rotated square lattice still
  gives \(\psi_4=1\)) but sensitive to *relative* orientation: two square
  grains at \(45^\circ\) to each other suppress the global average while the
  local one stays near 1 — the polycrystal-vs-single-crystal distinction the
  benchmark relies on.
- Peak detection plus bond order on a `lattice_seed`-initialised field
  recovers \(\psi_4\gtrsim0.8\) end-to-end, not just on hand-built points.
- `FreeEnergySampler` matches the elementary \(k=0\) formula exactly on a
  constant field, and its `mean_psi` tracks exact mass conservation through
  ETD steps to \(10^{-12}\).

## Code layout

| Piece | Location |
|---|---|
| Polynomial kernel helper | [`correlation_kernel.hpp`](include/higher_order_pfc/correlation_kernel.hpp) |
| Physics / schema / symbols | [`higher_order_pfc_physics.hpp`](include/higher_order_pfc/higher_order_pfc_physics.hpp) |
| Local nonlinearity (host + device) | [`higher_order_pfc_pointwise.hpp`](include/higher_order_pfc/higher_order_pfc_pointwise.hpp) |
| Session wiring + diagnostics hook | [`higher_order_pfc_session.hpp`](include/higher_order_pfc/higher_order_pfc_session.hpp) |
| Initial conditions | [`cosine_mode.hpp`](include/higher_order_pfc/cosine_mode.hpp), [`seeded_noise.hpp`](include/higher_order_pfc/seeded_noise.hpp), [`lattice_seed.hpp`](include/higher_order_pfc/lattice_seed.hpp) |
| Free energy + reciprocal-space sample | [`free_energy.hpp`](include/higher_order_pfc/free_energy.hpp) |
| Real-space bond-orientational order | [`order_parameter.hpp`](include/higher_order_pfc/order_parameter.hpp) |
| Diagnostics sample + CSV | [`diagnostics.hpp`](include/higher_order_pfc/diagnostics.hpp) |
