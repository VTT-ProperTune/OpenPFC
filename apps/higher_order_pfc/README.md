<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# `higher_order_pfc` — eighth-order PFC correlation kernel

A deliberately **very high-order** phase-field-crystal example. The free-energy
kernel runs through \((\nabla^2)^4\) (eighth order in space); conserved dynamics
adds one more Laplacian, so the evolution operator runs through
\((\nabla^2)^5\) — **tenth order**.

The point of the example is not the physics alone. It is that in a spectral code
a tenth-order operator is a five-term Horner loop, while the equivalent
real-space stencil would be a wide, awkward, hard-to-scale object. **High
differential order is not high implementation complexity here.**

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

## Code layout

| Piece | Location |
|---|---|
| Polynomial kernel helper | [`correlation_kernel.hpp`](include/higher_order_pfc/correlation_kernel.hpp) |
| Physics / schema / symbols | [`higher_order_pfc_physics.hpp`](include/higher_order_pfc/higher_order_pfc_physics.hpp) |
| Local nonlinearity (host + device) | [`higher_order_pfc_pointwise.hpp`](include/higher_order_pfc/higher_order_pfc_pointwise.hpp) |
| Session wiring | [`higher_order_pfc_session.hpp`](include/higher_order_pfc/higher_order_pfc_session.hpp) |
| Initial conditions | [`cosine_mode.hpp`](include/higher_order_pfc/cosine_mode.hpp), [`seeded_noise.hpp`](include/higher_order_pfc/seeded_noise.hpp) |
