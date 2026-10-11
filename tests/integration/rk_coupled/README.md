<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Coupled production Runge–Kutta regression

This test invokes the production `create(fields, eval, model, dt, tableau)`
factory with model-owned gradient and increment aggregates. It uses midpoint
RK2 and classical RK4, rather than an independently orchestrated stage loop.
The homogeneous coefficients and coupled mixed-derivative equations match
[the teaching model](../../../examples/coupled_fields/model.hpp). Optional
first-derivative, nonlinear and explicit-time coefficients are active in the
independent stage control; they are explicitly zero for the harmonic Fourier
control.

`oracle.hpp` uses its own periodic dense-array FD6 coefficients and its own
RK2/RK4 coefficients. Every actual stage input, stage time, owned output and
periodic face/edge/corner halo is checked. Before preparation the stage halos
are poisoned. The Field factory must expose the current stage simultaneously
in both fields, refresh the evaluator, scatter at the owned origin and restore
accepted storage. These checks use the existing factory signature and fail
numerically with the original stale-input factory.

The host test also verifies FD6 spatial refinement on the same bare analytic
mode used by the spectral and native stage controls, and
nonlinear, explicitly time-dependent temporal refinement for the exact
uniform solution `u=exp(t), v=exp(-t)`. The latter has active nonlinear terms
and imposed time-dependent sources; it is a numerical correctness control.
With HeFFTe enabled the same model and production factory check spectral
stages against an independent Fourier-mode recurrence and nonlinear temporal
order. No physical-model or stability claim follows from these cases.

A checked successful C++ `new` observer covers a warmed production step.
Host FD and spectral cases require zero requests across the complete step,
including Field stage copies, Full halo callbacks and preparation. Native
HIP also observes the complete step: observation stays active inside every
RHS and runtime call. It requires zero requests in host RK orchestration and
unclassified RHS code, while separately reporting uploads, Full halos,
audits, dispatch, downloads, background threads and observer instrumentation.
A controlled strict-total-zero run exposed HIP/HSA runtime requests: sampled
main-thread stacks entered `libamdhip64`, and sampled background-thread
stacks entered `libhsa-runtime64`. Runtime commands are therefore not claimed
allocation-free. Every observed native step prints total and per-phase count
deltas, plus each active phase's first process trace when fully published.
Background-thread requests remain reported; they are not suppressed or
reclassified as stepper requests.

Ordinary and aligned requests, including alignment below the POSIX pointer
minimum, independently verify the provider. Known scoped RHS and subsequent
ordinary requests also verify phase attribution and restoration. Field,
stage, oracle, halo and FFT scratch is constructed before observation. Native
mirrors, runtime and exchange warm before the observed step. Native trace
capture uses fixed storage and atomic publication; nested diagnostic requests
remain counted in a separate instrumentation bucket. The observer excludes
`malloc`, MPI/runtime pools and accelerator allocations. These checks verify
C++ allocation attribution, not performance or a globally allocation-free
accelerator runtime.

The native HIP test invokes the production `MultiExplicitRKStepper` with a
preallocated RHS realization: it uploads actual stage buffers to padded
fields, performs native Full halo exchange, uses the same pure model through
the existing fixed-arity scatter adapter, and downloads derivatives for the
production stepper. All stages and outputs compare with the independent
CPU stencil recurrence. It also checks the same nonlinear, explicitly timed
uniform manufactured solution at two timestep sizes for RK2 and RK4.
Transfers and full-array audits are intentional test costs; this is not a
resident-device integrator or a performance test. Actual hardware is required; there is no no-device skip or host fallback.

The resident CUDA/HIP tests (`resident.cu` / `resident.hip`) call
`create(std::tie(u, v), eval, model, dt, tableau)` on
`Field<double, CUDASpace>` / `HIPSpace`. Stage construction, Full halo
refresh, RHS, and accumulation stay on the device. Per-stage downloads in
the test are audits of `eval.prepare()` against `oracle.hpp` (stage
inputs, face/edge/corner ghosts, accepted state), not stepper transfers.

The one-rank entry point always runs. With MPI suites enabled the host,
native HIP, and resident CUDA/HIP tests register 1/2/4/8-rank cases,
subject to the configured maximum world size. Stage cases use a periodic 24³ global grid with x slabs, including owned
x extent three at eight ranks, and audit every actual stage halo. The spatial
refinement additionally uses 32³ and 64³ grids. Native
rank/device metadata is printed to stderr. Build and run through
`scripts/build.sh`; no standalone manual build command is required.
