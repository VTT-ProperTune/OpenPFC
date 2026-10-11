<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Decomposition and FFT

OpenPFC spectral workflows partition a `Domain` across MPI ranks and transform
local data through an implementation of the FFT contract. Read the
[spectral-stack concept](../concepts/spectral_stack.md) before using the exact
API below.

For uniform bounded axes, see [boundary transforms](boundary_transforms.md):
explicit DCT/DST grids and periodic x/y with bounded z.

## Chebyshev Helmholtz solves

The rank-local functions in `openpfc/kernel/fft/chebyshev.hpp` solve
`u'' - lambda*u = f` on the Lobatto grid of `[-1,1]`. These are not
uniform-grid DCT/DST Laplacian solves. Boundary slopes are derivatives
with respect to increasing z; an outward Robin slope has positive weight
at +1 and negative weight at -1.

For nonzero lambda, let J integrate a Chebyshev series with integration
constant zero. The solver enforces
`a[k] - lambda*(J*J*a)[k] = (J*J*f)[k]`, k=2,...,N, and two exact
coefficient boundary rows. Both extra modes of the first integral are
retained through the second integration before truncating the equations.
This integral formulation replaces the downward recurrence and its
cancellation between exponentially growing boundary-matching columns.
The finite truncation is an integrated coefficient formulation, not
bitwise the previous tau recurrence or pointwise collocation equations.

Row equilibration, partial pivoting and extended-precision elimination
are internal; the public values remain doubles. No small forcing modes
are discarded. Singular or numerically unresolved boundary systems and
nonfinite data are rejected. Negative lambda remains allowed, but
pure Dirichlet/Neumann resonances at `lambda=-(m*pi/2)²`, m>=1, are
rejected explicitly. General Robin poles are only guarded by the pivot
test, not a condition-number estimate. Near-resonance sensitivity and
insufficient spatial resolution are not cured by this formulation.
Numerical differentiation of a returned high-degree field
can still amplify double-precision sampling roundoff.

Zero lambda retains the existing Poisson integrations, Neumann flux
compatibility check and zero-integral gauge. Fourier-Chebyshev callers
reuse this solve with lambda plus the periodic squared wavenumbers.
The current direct solve uses O(N²) storage and O(N³) worst-case work
per bounded line; factorization/plan caching is not provided.

## `pfc::decomposition::Decomposition`

```{doxygenstruct} pfc::decomposition::Decomposition
:project: OpenPFC
:members:
:no-link:
```

## `pfc::fft::IHostFFT`

```{doxygenstruct} pfc::fft::IHostFFT
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

## `pfc::fft::CPUFFT`

```{doxygenclass} pfc::fft::CPUFFT
:project: OpenPFC
:members:
:protected-members:
:no-link:
```
