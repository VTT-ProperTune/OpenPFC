<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Coupled pointwise fields

This teaching example uses the existing model/evaluator/scatter contract for

```
u_t = v + a*u_xy
v_t = b*(u_xx + u_yy + u_zz) + c*v_xz + d*u_yz
```

Start with the [Quarto lesson](index.qmd) and its configurable `model.cpp`.
It writes complete snapshot fields and rates; `input.json` selects FD6 and
`spectral.json` selects Fourier gradients. Keep `../equations.hpp` alongside
the chapter when using `openpfc run --source-root ..`.

`Model::rhs(stage_time, Local{UGrads u, VGrads v})` returns model-owned
`Increments{du,dv}` with the host tuple protocol. `UGrads` requests only
`xx,yy,zz,xy,yz`; `VGrads` requests only `value,xz`. Derivatives include
physical grid spacing; the model does not scale them a second time. The HIP
adapter converts those same increments to the existing fixed-arity device
scatter pack. It contains no second copy of the equations. Host output pointers
are offset to the owned origin because padded FD evaluator indices are relative
to that origin; device scatter accepts the padded allocation base.

The default coefficients `.3,.04,-.2,2/15` give the exact periodic solution
`u=cos(t)*cos(x+2y+3z)`, `v=(-sin(t)+.6*cos(t))*cos(x+2y+3z)`. Analytic gradients
and time derivatives are independent of the finite-difference evaluator.
Every mixed term contributes. A separate explicit-source adapter exercises
stage time and physical position forwarding through the same canonical RHS
call, without changing the homogeneous equations or their exact solution. This is a numerical API control, not a physical
model or a performance comparison.

Build and check through `scripts/build.sh`. The preserved `test_coupled_fields_example`
verification entry point is also built and invoked by CTest when other examples are off.
It requires one MPI rank. FD6 uses padded fields and explicitly declares
`HaloConnectivity::Full`; the real Full halo exchange refreshes poisoned
faces, edges and corners before each stage. A preallocated midpoint RK
composition binds the evaluator to the actual stage fields and compares
four evolving steps against an independent discrete Fourier-mode recurrence.
It counts exactly one evaluation of each field at every owned point and
checks that tuple scatter leaves output ghosts untouched. An executable-local
observer checks successful C++ allocation requests in the warmed stage loop;
it does not observe malloc, MPI/runtime memory pools or hardware accesses.

This example does not use the existing `MultiExplicitRKStepper` factory:
its evaluator can remain bound to the original field rather than the passed
stage buffer, and its current stage-vector construction allocates. Those
library limitations are tracked in [#355](https://github.com/ahojukka5/OpenPFC/issues/355)
and are not repaired or hidden by this demonstrator. The
model ABI and canonical `for_each_interior` driver remain unchanged.

With HeFFTe enabled, the same minimal aggregates, model and tuple scatter use
spectral gradients on the same resolved periodic mode. FD and spectral
residuals are compared to their analytic oracle, allowing FD truncation error.
The native HIP check uses actual device evaluation and scatter, preallocated
buffers, three changed inputs, and compares all output cells with the host
FD6 result. Host-filled Full halos are uploaded in this correctness fixture;
it does not claim to test distributed device exchange or a device RK stepper.
The independent Full-halo tests cover device communication.
