<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Science and numerics — practical limits

This page collects **honest** expectations: OpenPFC is a serious HPC framework, but **no** numerical method is universal. For code layering, read [`architecture.md`](../concepts/architecture.md); for spectral vs FD direction, read [`when_not_to_use_openpfc.md`](../when_not_to_use_openpfc.md) and [`adr/0002-gradient-operators-fd-vs-spectral.md`](../adr/0002-gradient-operators-fd-vs-spectral.md).

## Spectral semi-implicit style (typical shipped models)

- **Linear** operators are often treated **implicitly** in Fourier space; **nonlinear** terms are evaluated in **real space** and coupled through FFTs.  
- **Timestep** `dt` must respect the **explicit** pieces of your splitting; blowing up often means `dt` is too large for the nonlinear or stabilized explicit stages—not “MPI is broken.”  
- **Resolution** changes physics: interface widths, defect cores, and nucleation depend on grid spacing; do not compare runs at different resolutions without a convergence mindset.
- Aliasing: a degree-p polynomial of a field restricted to integer modes
  `|m_i| <= K_i` produces frequencies up to `p*K_i`. To obtain the exact
  retained Galerkin product by sampling and transforming, require
  `(p+1)*K_i < N_i` on each axis. Thus quadratic products need `3*K_i < N_i`,
  while cubic products need `4*K_i < N_i`. The input state and every
  nonlinear stage must respect the intended retained band. Filtering only
  the product's high modes cannot remove aliases already folded into low
  modes. `pfc::fft::kspace::fill_two_thirds_mask` supplies a two-thirds
  Nyquist-band cutoff; it does not by itself fully dealias a cubic. A
  sufficiently padded product grid is another option. No general filter
  is enabled automatically for arbitrary models.
- **Incompressible velocity:** `pfc::field::leray_project` removes the
  longitudinal part of a periodic Fourier velocity,
  $P = I - kk^{\mathsf T}/|k|^2$. The wavevector is the odd multiplier, so
  the mean and the even-grid Nyquist mode stay unchanged. The quadratic term
  is $P(u\times\omega)$ after a state-level 2/3 truncation
  (`solvers/incompressible/rotational.hpp`). That mask is on for this term only.
  A 3/2 padded product is not used.
- **Nyquist (odd derivatives):** `SpectralGradient` zeros `i k` at the even-grid Nyquist mode. The real-to-complex Nyquist coefficient is real; a nonzero `i k_N` is not uniquely defined.
- **2-D vorticity NS (issue #21) and incompressible MHD (issue #23):**
  `examples/ns2d_vorticity` reuses the 2/3 mask and odd-$k$ rule, and
  **projects the state** onto the retained band before products. The
  nonlinear term is IFRK4. MHD adds a magnetic potential $a$ with
  Lorentz $+B\cdot\nabla j$; it is 2-D incompressible MHD, not Strauss
  reduced MHD. A current-sheet movie is not reconnection.

## Finite differences (kernel helpers and examples)

- FD stencils need **halo exchange**; **in-place** vs **separated** halos interact with FFT layouts ([`halo_exchange.md`](../concepts/halo_exchange.md)).  
- FD and spectral operators can coexist in **principle**; user-facing **unified switching** for gradients/Laplacians is **still maturing** (see ADR 0002).

## What pretty pictures do not prove

- Visualizations (VTK, PNG) are **diagnostics**. They do not replace mesh/time convergence, mass/energy checks where applicable, or comparison to reference solutions.

## See also

- [`science_tungsten_quicklook.md`](tungsten_quicklook.md) — what the tungsten model encodes  
- [`science_cahn_hilliard_vs_allen_cahn.md`](cahn_hilliard_vs_allen_cahn.md) — example entry points  
