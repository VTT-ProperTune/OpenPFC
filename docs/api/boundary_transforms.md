<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Uniform-grid boundary transforms

Include `openpfc/kernel/fft/real_transform.hpp` for a rank-local
`RealTransform`, or `openpfc/kernel/fft/mixed_transform.hpp` for a
distributed `MixedTransform`. These additive APIs preserve `IHostFFT`,
`IDeviceFFT` and [Chebyshev collocation](../../include/openpfc/kernel/fft/chebyshev.hpp).
They require FFTW; the mixed API also requires a heFFTe-enabled build.
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
trapezoidal (half weight at endpoints) for DCT1. For the mixed solve,
this applies only to the global x=y=z zero mode, and compatibility is
checked collectively. Sine has no nullspace. Nonfinite forcing is rejected.

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

## Usage and distributed layout

```cpp
using namespace pfc::fft;
BoundedAxis z{RealTransformType::DST2, UniformGrid::CellCentered, 64, 2.0};
RealTransform line(z);
std::vector<double> f(z.n), u(z.n);
// Fill f at coordinate(z,j); homogeneous Dirichlet is imposed by the basis.
line.poisson(f, u);

// After MPI_Init: periodic x/y, same bounded z, y slabs with full x/z.
MixedTransform mixed(64, 64, 3.0, 4.0, z, MPI_COMM_WORLD);
std::vector<double> rhs(mixed.size_inbox()), solution(rhs.size());
// Fill rhs using mixed.inbox(), x-fastest, then local y, then z.
mixed.poisson(rhs, solution);
```

All ranks must pass identical geometry and valid buffer extents.
The communicator is borrowed and must outlive the plan; destroy plans
before MPI_Finalize. At most ny ranks are supported, each with complete
z lines. The outbox reduces x to 0..nx/2 and retains all y frequencies
and z modes. Both buffers are x-fastest. The bounded transform is a
rank-local FFTW batch. The periodic planes are one heFFTe batched 2D
FFT, including heFFTe's existing communication and layout machinery.
No new transpose implementation is involved. This supports all four
rows of the table with periodic x/y, including unit periodic dimensions.

Plans and scratch are reused; input buffers are preserved. `buffer_bytes`
counts owned arrays, excluding opaque plan allocations. Planning and
destruction obey FFTW's serial planner requirements. One instance must
not execute concurrently. MPI execution is collective. Arbitrary
decompositions that split z are not implemented.
Native distributed homogeneous sine/cosine transforms remain accessible
through heFFTe's versioned `rtransform` API; its raw scaling depends on
the backend and must not be assumed to match this FFTW contract.

Regression tests cover mirrored DFT coefficients, round trips, analytic
derivatives and boundary traces, both Laplacian symbols, Poisson solves,
small odd/even grids, weighted Neumann compatibility and MPI mixed layouts.

## Batched device transforms

`openpfc/kernel/fft/batched_transform.hpp` adds
`BatchedRealTransform<MemorySpace>` and `BatchedMixedTransform<MemorySpace>`.
Include `openpfc/runtime/gpu/memory_space_gpu.hpp` for `pfc::CUDASpace` or
`pfc::HIPSpace`. The corresponding vendor spectral build and FFTW are
required. Only the cell-centered DCT2/DST2 rows are supported; type I,
type IV and mixed endpoint conditions are rejected, not inferred.

The line plan accepts explicit batch count, sample stride and batch
distance, all positive, in doubles. Lines must not overlap. Buffer extents
include padding: `(batches-1)*distance+(n-1)*stride+1`. Padding is untouched.
Forward exposes the same raw real coefficients as `RealTransform` and
inverse is normalized. Input and coefficient arrays are preserved;
an identical input/output span is allowed for rank-local line transforms.

The mixed plan exposes the same complex coefficients and y-slab layout
as `MixedTransform`. All arrays passed as spans point to device memory:

```cpp
BoundedAxis z{RealTransformType::DST2, UniformGrid::CellCentered, 64, 2};
BatchedMixedTransform<pfc::CUDASpace> mixed(64, 64, 3, 4, z);
// rhs and solution are device spans of mixed.size_inbox() doubles.
mixed.helmholtz(rhs, solution, 2.0); // (-Delta+2)u=f
// mixed.poisson(rhs, solution) and mixed.laplacian(rhs, solution) also work.
```

Plans/workspaces are reused without execution-time buffer allocation.
The legacy default stream is used; synchronize it before accessing results
from another stream or host. Construct, execute and destroy on the same
current device. One instance is not reentrant. The communicator is borrowed
and must remain live until destruction. One rank uses a vendor 3D FFT;
multiple ranks reuse heFFTe's distribution with full z output lines.
GPU-aware MPI follows the configured `OpenPFC_MPI_CUDA_AWARE` or
`OpenPFC_MPI_HIP_AWARE` setting. Without it, heFFTe stages communication
through host buffers; no extra field copy is made by the bounded transform.
At most ny ranks are supported. The checked vendor extent limit is
nx*ny*n <= INT_MAX. Input values must be finite. GPU Neumann Poisson
additionally detects nonfinite coefficients while checking compatibility.
The ny bound is necessary for the owned y slabs, not sufficient for every
small grid: heFFTe must also be able to assign nonempty intermediate FFT
boxes. Its planner can reject an undersized intermediate decomposition.

`helmholtz` accepts finite nonnegative reaction; zero reaction is Poisson.
Both Laplacian symbols remain available. All ranks must agree on reaction,
symbol and compatibility tolerance. Neumann Poisson checks the global
constant mode against the maximum raw mixed coefficient, transfers only
three diagnostic scalars from device to host, then reduces them over MPI.
Its solution has zero arithmetic mean. Positive reaction retains the
constant mode. No field is copied to the host during a solve.

Internally the line transform reorders even-indexed samples ascending and
odd-indexed samples descending, using an n-point real FFT. DST2 uses
alternating signs and reversed cosine modes. The mixed solve permutes
whole z planes and uses a half-volume 3D FFT. With complex periodic modes
A=F[k], B=F[n-k] and theta=pi*k/(2n), its cosine coefficients are
`C[k]=cos(theta)(A+B)-i sin(theta)(A-B)` and
`C[n-k]=sin(theta)(A+B)+i cos(theta)(A-B)`.
The two different diagonal symbols are applied together before recovering
the Fourier pair. Taking a real part here would lose the periodic phase.
Constant modes, even-size Nyquist modes and odd-size partners are handled
explicitly. Forward/backward still expose actual DCT/DST coefficients;
only complete operator applications fuse conversion and diagonalization.

Device regressions compare raw coefficients with direct trigonometric
sums and independent host transforms, including padded/interleaved batches,
odd/even periodic dimensions, nullspaces and both diagonal symbols.

### Distributed exchange policy

The original `BatchedMixedTransform` constructor retains heFFTe's
`alltoallv` exchange policy. An overload accepts the installed heFFTe
`reshape_algorithm` directly; include `<heffte_plan_logic.h>` when naming
its enumerators. Supported choices are `alltoallv`, `alltoall`, `p2p` and
`p2p_plined`, on both CUDA and HIP builds:

```cpp
pfc::fft::BatchedMixedTransform<pfc::HIPSpace> transform(
    nx, ny, period_x, period_y, z,
    heffte::reshape_algorithm::p2p_plined, communicator);
```

All ranks must supply the same valid policy; invalid or mismatched values
are rejected collectively before planning. This does not change y-slab /
full-z ownership, normalization, the borrowed communicator or the existing
build-time GPU-aware MPI policy. One-rank execution still uses the vendor
FFT directly. No automatic policy selection or arbitrary decomposition is
provided. Choose an exchange policy using verified end-to-end measurements
on the target MPI installation, rather than assuming one is universally
fastest.

### Retained spectral ownership

Physical fields always use the existing y-slab/full-z decomposition. The
default spectral ownership remains y slabs. An explicit overload allows
full-z **spectral x slabs**, without changing the boundary grid or operator:

```cpp
pfc::fft::BatchedMixedTransform<pfc::HIPSpace> transform(
    nx, ny, period_x, period_y, z,
    heffte::reshape_algorithm::p2p_plined,
    pfc::fft::SpectralSlab::X, communicator);
```

`inbox()` still describes physical y slabs; `outbox()` now describes the
selected spectral ownership. Obtain buffer extents from the plan rather
than assuming they equal another rank's extents. Both forward and backward
use canonical x-fast exposed coefficients, even though distributed complete
operators retain heFFTe's y-fast/z-slow internal order. Conversion is fused
into the existing coefficient kernels; physical fields are not staged on
the host. Complete bounded-z lines remain local in both layouts.

All ranks must agree on the valid choice. X additionally requires
`ranks <= nx/2 + 1`; physical ownership still requires `ranks <= ny`.
These are necessary ownership bounds, not a promise that every tiny
intermediate library plan is valid. One rank keeps the original vendor
FFT and storage order. `SpectralSlab::Y` preserves the original layout and
all existing constructor defaults. Normalization, signs, zero modes,
compatibility checks, borrowed-communicator and device-lifetime contracts
are unchanged. No arbitrary decomposition or automatic selection is added.
