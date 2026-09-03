<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Time Integration Architecture Contract

This document defines the formal contract between time integrators (steppers) and
the OpenPFC simulation driver. It specifies the responsibilities of stepper
implementations, state access patterns, boundary preparation ordering, and output
scheduling guarantees. All stepper implementations must adhere to this contract to
ensure correct behavior across CPU and accelerator backends.

## 1. Contract scope and responsibilities

### Stepper role

A time stepper advances the simulation state through one or more computational
stages. It coordinates physics evaluation with boundary preparation and respects
output scheduling. It is the orchestration layer between the temporal evolution
logic and the spatial discretization implemented by the gradient evaluator or the
spectral system.

### Driver hooks

The formal contract boundary is the hook sequence of the free-function loop
[`pfc::sim::run`](../../include/openpfc/kernel/simulation/simulation_driver.hpp)
(also reachable through the thin `pfc::sim::SimulationDriver` bundle):

```cpp
pfc::sim::run(time, step, on_start, apply, on_save);
```

- `on_start(Time&)` — called once, when `pfc::time::increment(time) == 0`,
  before the first step (initial conditions, first boundary application).
- `apply(Time&)` — called after `pfc::time::next(time)` and before `step`
  (boundary conditions at the new time).
- `step(double t)` — the physics advance; receives the accepted time after
  `next()`.
- `on_save(const Time&)` — called whenever `pfc::time::do_save(time)` is true:
  after `on_start` on the first iteration and after every `step`.

There is no `Simulator` class and no virtual `Model::step`; sessions compose the
hooks from a `SimulationState`, a stepper (or a spectral ETD system), condition
lists, writers, and a `CheckpointService`.

## 2. Driver hook semantics

### Full ordering contract

`pfc::sim::run` implements the following ordering, per iteration:

1. **Initial path** (when `pfc::time::increment(time) == 0`):
   - `on_start(time)` — apply initial conditions and boundary conditions
   - `on_save(time)` if `pfc::time::do_save(time)`
2. **Every iteration**:
   - `pfc::time::next(time)` — advance increment and current time
   - `apply(time)` — boundary conditions at the new time
   - `step(pfc::time::current(time))` — physics advance
   - `on_save(time)` if `pfc::time::do_save(time)`

The loop terminates when `pfc::time::done(time)` is true.

### State access patterns

Steppers operate on fields owned by `pfc::SimulationState`
(`include/openpfc/kernel/simulation/simulation_state.hpp`):

- `state.get_field<double>(name)` — host real field (`pfc::data::Field`)
- `state.get_field<std::complex<double>>(name)` — host complex field
- `state.get_field<T, MemorySpace>(name)` — device-resident field

The raw stepper leaves in
[`steppers/`](../../include/openpfc/kernel/simulation/steppers/) take the field's
contiguous storage (`field.vec()` on host) or `HostFieldState` wrappers. All field
access must respect the halo exchange requirements in
[Section 4](#4-boundary-and-halo-preparation).

The `Time` object provides temporal state through free functions in
`pfc::time`:

- `pfc::time::current(time)` — current physical time
- `pfc::time::increment(time)` — current step counter (0-based)
- `pfc::time::do_save(time)` — whether output should be written at this time
- `pfc::time::next(time)` — advance to the next time step
- `pfc::time::done(time)` — whether `t1` has been reached

## 3. Physics evaluation patterns

### Single-stage steppers

Single-stage steppers (e.g. explicit Euler) perform exactly one advance per
`step` hook:

```cpp
pfc::sim::run(time, [&](double t) {
  stack.exchange_halos();
  (void)stepper.step(t, stack.u().vec());
});
```

[`pfc::sim::steppers::EulerStepper`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)
is the reference single-stage implementation with the signature:

```cpp
double step(double t, std::vector<double>& u);
```

### Multi-stage steppers

**Critical contract requirement:** the `apply` / `on_save` hooks bracket the
**full timestep**, not each substage. All intermediate substage computations occur
inside the `step` hook with no additional hook invocations.

For example, an explicit RK4 stepper performs all four slope evaluations inside one
`step` call:

```cpp
auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
auto rk4 = pfc::sim::steppers::create(stack.u(), grad, model, dt, tableau);

pfc::sim::run(time, [&](double t) {
  stack.exchange_halos();
  (void)rk4.step(t, stack.u().vec());   // k1..k4 inside; no hook calls
});
```

This design ensures that boundary conditions and output scheduling are applied
only at the physical time level, not at intermediate substage times that may not
correspond to meaningful physical states.

### Attempt / commit protocol

All stepper leaves implement the attempt/commit protocol
(`StepAttemptResult` in
[`steppers/step_attempt.hpp`](../../include/openpfc/kernel/simulation/steppers/step_attempt.hpp),
see [ADR 0003](../adr/0003-time-integrator-interface.md)): `attempt(t, u)`
computes an isolated candidate without mutating `u`; `commit_step_attempt`
publishes it. In-place `step()` is attempt followed by commit. Adaptive control
(`AdaptiveTimeController`) rejects an attempt by simply not committing it and
shrinking `dt` through `Time`'s attempt transaction.

### Physics evaluation API

Point-wise physics is a callable `rhs(double t, const Grads&)` consumed by
`pfc::sim::for_each_interior` through a gradient evaluator. Stiff spectral physics
is described by k-space symbols (`linear_symbol(k)`, optional
`nonlinear_symbol(k)` / `filter_mf(k)` / `correlation_kernel(k)`) plus a
device-capable pointwise functor (`pointwise()` returning an `OPENPFC_HD`
`nonlinearity(const SpectralCell&)`), consumed by the one framework-owned
`pfc::sim::SpectralETDSystem<Physics, MemorySpace>` on `SimulationState`. The
system itself follows the attempt/commit shape (`attempt(t)` forms the
candidate in system-owned scratch, `commit()` publishes it, `reject()` is free,
`set_dt` rebuilds the coefficients). Neither physics kind owns the time loop.
See [Extending OpenPFC](../extending_openpfc/README.md#add-a-spectral-etd-physics).

## 4. Boundary and halo preparation

### Halo exchange architecture

The halo exchange architecture is specified in
[`docs/concepts/halo_exchange.md`](halo_exchange.md). Steppers must respect the
halo exchange requirements when using real-space stencils on distributed domains.

### Finite difference steppers

Finite difference steppers **must** exchange halos before physics evaluation when
using real-space stencils. `FDCPUStack::exchange_halos()` (or
`comm::HaloExchange::exchange()` on a padded `Field`) synchronizes ghost cells
between neighboring ranks so that stencils see valid neighbor data:

```cpp
pfc::sim::run(time, [&](double t) {
  stack.exchange_halos();                 // ghost cells valid
  (void)stepper.step(t, stack.u().vec()); // stencils read them
});
```

### Spectral steppers

Spectral steppers skip halo exchange because FFT-based operations act on the global
k-space representation; neighbor communication is handled by the FFT
decomposition.

### Boundary condition application

Boundary conditions are applied in two places:

1. **In the driver hooks** — `on_start` (after ICs) and `apply` (after
   `time.next()`), typically by `FieldModifier` objects or a
   `StagePreparationService`.
2. **Post-physics** — required only for multi-stage schemes where intermediate
   states violate boundary constraints; the stepper applies them inside `step`.

**Single-stage steppers** generally do not require post-physics BC reapplication.

**Multi-stage steppers** may require post-physics BC application depending on the
scheme (predictor-corrector, split-step). Runge-Kutta methods typically do not.

## 5. Output scheduling guarantees

### Save point detection

Output scheduling is controlled by `Time` through `pfc::time::do_save(time)`,
which is true when the current time is a scheduled save point on the `saveat`
grid (floating-point modulo with a `1e-6` tolerance). A `saveat` of `0.0`
disables automatic saving. `Time` is the **only** save-point scheduler.

### Result writing mechanism

Sessions write results in the `on_save` hook by dispatching the registered
`ResultsWriter`s with the current field state and incrementing their result
counter (used for incrementing filenames such as `field_0000.bin`).

### Custom stepper responsibilities

Steppers must not write results themselves; they rely on the driver's `on_save`
hook. Steppers that need custom output timing check `pfc::time::do_save(time)` in
their own hook implementation.

## 6. Restart and checkpoint semantics

### Checkpoint writing

Scheduled field dumps go through `ResultsWriter`s as **headerless** MPI-IO bricks
(periodic output and post-processing); the raw layout is specified in
[`docs/reference/binary_field_io_spec.md`](../reference/binary_field_io_spec.md).

For a **durable accepted-state restart bundle**, sessions own a
`pfc::sim::CheckpointService`
(`include/openpfc/kernel/simulation/checkpoint_service.hpp`), configured by the
JSON keys `checkpoint.every` / `checkpoint.directory` / `restart_from`. It
publishes versioned `metadata.json` plus accepted field bricks through a staging
directory and an atomic rename, so incomplete writes are never loadable, and
restores fields, `Time`, the result counter, and the integrator identity on
`restart_from`. See
[`docs/development/checkpoint_publish.md`](../development/checkpoint_publish.md).

### Resume expectations

Restart requires:

1. **Consistent field state** — all fields contain values at the checkpoint time
2. **Matching `Time` configuration** — the same `t1`, `dt`, and `saveat`; `t0`
   and the increment are restored from the bundle
3. **Identity match** — grid, decomposition, and integrator method must equal the
   bundle's metadata (mismatch is a hard error)

## 7. CPU and accelerator numerical contract

### Numerical equivalence requirement

Both CPU and GPU paths must produce identical results within floating-point
tolerance (the parity suites pin `1e-10` for tungsten, allen_cahn, and wave2d).

### Architecture layering

The kernel/runtime separation described in
[`docs/concepts/architecture.md`](architecture.md) enforces this contract:

- **Kernel layer** — backend-agnostic numerical algorithms and data structures
- **Runtime layer** — backend-specific implementations (`runtime/gpu`, single
  source for CUDA and HIP)

### GPU implementation obligations

GPU implementations of steppers must:

1. **Obey the same stage ordering** as CPU implementations
2. **Use equivalent numerical algorithms** — same order of operations, same
   stencil coefficients
3. **Handle boundary conditions identically** — same BC application points
4. **Respect the same halo exchange contract** — same ghost cell synchronization

Deviations are acceptable only when documented as platform-specific limitations.

## 8. Example integration patterns

### Single-stage Euler on the FD CPU stack

```cpp
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/time.hpp>

void run_euler(pfc::sim::stacks::FDCPUStack &stack, heat3d::HeatModel &model,
               pfc::Time &time, double dt) {
  auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), /*order=*/2);
  auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, dt);
  pfc::sim::run(time, [&](double t) {
    stack.exchange_halos();
    (void)stepper.step(t, stack.u().vec());
  });
}
```

This is the pattern used by the heat3d tests
([`apps/heat3d/tests/test_heat3d.cpp`](../../apps/heat3d/tests/test_heat3d.cpp)).

### Multi-stage stepper with conditions and writers

```cpp
pfc::sim::run(
    time,
    [&](double t) {            // step: all RK stages inside
      stack.exchange_halos();
      (void)rk4.step(t, stack.u().vec());
    },
    [&](pfc::Time &) {         // on_start: ICs + BCs
      for (auto &ic : initial_conditions) apply_modifier(*ic, stack.u(), 0.0);
      for (auto &bc : boundary_conditions) apply_modifier(*bc, stack.u(), 0.0);
    },
    [&](pfc::Time &tm) {       // apply: BCs at the new time
      for (auto &bc : boundary_conditions)
        apply_modifier(*bc, stack.u(), pfc::time::current(tm));
    },
    [&](const pfc::Time &) {   // on_save
      writer.write(counter++, stack.u());
    });
```

### Spectral ETD system as the stepper

Production spectral apps (tungsten, aluminumNew) do not use a raw stepper leaf;
the framework-owned spectral ETD system on `SimulationState` is the `step` hook:

```cpp
pfc::sim::run(time, [&](double t) { system.step(t); }, on_start, apply, on_save);
```

---

**See also:** [`docs/concepts/architecture.md`](architecture.md) for overall
system architecture, [`docs/concepts/halo_exchange.md`](halo_exchange.md) for halo
exchange patterns, [`docs/development/checkpoint_publish.md`](../development/checkpoint_publish.md)
for atomic accepted-state publication, and
[`docs/reference/binary_field_io_spec.md`](../reference/binary_field_io_spec.md)
for headerless scheduled field dumps.
