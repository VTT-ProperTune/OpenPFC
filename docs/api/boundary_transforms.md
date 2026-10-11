<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Uniform-grid boundary transforms

Include `openpfc/kernel/fft/real_transform.hpp` for a rank-local
`RealTransform`. This additive API preserves `IHostFFT`,
`IDeviceFFT` and [Chebyshev collocation](../../include/openpfc/kernel/fft/chebyshev.hpp).
It requires an FFTW-enabled build.
Link the installed `OpenPFC::openpfc` target as usual.

## Select the discrete problem explicitly

Both bounded ends have the same homogeneous condition. A boundary label
does not select the grid or transform type. `BoundedAxis` requires both:

| Forward type | Grid | Unknown coordinates on [o,o+L] | Boundary condition |
| --- | --- | --- | --- |
| DCT1 | Endpoints | o+j L/(n-1), j=0..n-1 | Neumann |
| DCT2 | CellCentered | o+(j+1/2)L/n | Neumann |
| DST1 | Interior | o+(j+1)L/(n+1) | Dirichlet, endpoints fixed zero |
| DST2 | CellCentered | o+(j+1/2)L/n | Dirichlet |

DCT1 needs n>=2; other types need n>=1. Inconsistent type/grid pairs
throw. These are **uniform** physical grids. A DCT in this API does not
differentiate on Chebyshev–Lobatto nodes.

The cosine basis is `cos(m pi (x-o)/L)`, m=0..n-1; the sine basis is
`sin(m pi (x-o)/L)`, m=1..n. Forward coefficients use the raw FFTW
convention, including its factors of two. Type I is its own inverse;
type II uses type III for the inverse. `backward` divides by 2(n-1)
for DCT1, 2(n+1) for DST1, and 2n for either type II. Scaling is fixed
and explicit; this interface does not use orthonormal coefficients.

An even extension implements cosine and an odd extension implements sine.
Endpoint DCT1 mirrors without duplicating either endpoint. DST1 inserts
the two zero endpoints before mirroring. Cell-centered transforms reflect
around the two cell faces; their logical DFT has a half-sample phase.
The physical samples and the extension represent the same discrete problem
only when spacing, multiplier and normalization match.

## Operators and nullspace

`laplacian` applies Delta; `poisson` solves **-Delta u=f**.
`LaplacianSymbol::Spectral` uses positive eigenvalue k² for -Delta.
`CenteredDifference` instead uses `4 sin²(k h/2)/h²`, the three-point
second difference on the mirrored grid. For endpoints this corresponds
to reflected ghost values, with trapezoidal weights for the Neumann
inner product. These symbols must not be interchanged.

Cosine has a constant zero mode. `poisson` rejects a zero coefficient
larger than `compatibility_tolerance * max_abs_coefficient` and sets the
solution coefficient to zero. The mean is arithmetic for DCT2 and
trapezoidal (half weight at endpoints) for DCT1. Sine has no nullspace.
Nonfinite forcing is rejected.

`RealTransform::evaluate(coefficients,x,order)` evaluates the interpolant,
its first derivative or its spectral second derivative at any physical
point, using the analytic sine/cosine basis. Cost is O(n) per point;
this is for arbitrary-point sampling and boundary traces. First derivatives
change parity: cosine differentiates to minus sine, sine to plus cosine.
At a cell-centered DST2 highest mode the first derivative samples vanish
although the spectral second derivative does not. The same issue arises
at endpoint DCT1 Nyquist. Applying a sampled first derivative twice is
therefore not an implementation of the spectral Laplacian.

Nonhomogeneous data can be handled by a caller-provided lifting g:
solve for v=u-g with forcing `f+Delta g`, then restore g. Neumann
compatibility includes the prescribed flux through that lifting.
Unlike conditions at the two ends, Robin conditions, variable-coefficient
diagonal solves, and type IV are unsupported by this API.

## Usage

```cpp
using namespace pfc::fft;
BoundedAxis z{RealTransformType::DST2, UniformGrid::CellCentered, 64, 2.0};
RealTransform line(z);
std::vector<double> f(z.n), u(z.n);
// Fill f at coordinate(z,j); homogeneous Dirichlet is imposed by the basis.
line.poisson(f, u);
```

Plans and scratch are reused; input buffers are preserved. `buffer_bytes`
counts owned arrays, excluding opaque plan allocations. Planning and
destruction obey FFTW's serial planner requirements. One instance must
not execute concurrently.
