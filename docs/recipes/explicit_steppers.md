<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Recipe: Migrate from Model::step(double t) to explicit steppers

**Goal:** Transition legacy spectral apps using `Model::step(double t)` to the new pluggable integrator architecture with `pfc::sim::steppers::create()`, separating pointwise physics from explicit Euler and Runge–Kutta integration. IMEX methods additionally need an appropriate implicit-operator contract.

## Prerequisites

- OpenPFC built with stepper support (default in recent builds)
- Familiarity with existing Model-based spectral apps
- Understanding of gradient evaluators (see [`../extending_openpfc/per_point_grads.md`](../extending_openpfc/per_point_grads.md))
- ADR-0003 contracts: [`../adr/0003-time-integrator-interface.md`](../adr/0003-time-integrator-interface.md)

## Legacy pattern vs new pattern

**Legacy (model owns time integration):**
```cpp
class LegacyModel : public Model {
  void step(double t) override {
    // Model owns FFTs, field updates, and time-stepping logic
    auto& u = get_real_field();
    auto& fft = get_fft(*this);
    // ... FFTs, spectral multiplies, inverse FFTs ...
  }
};

// Usage
model.step(t);  // Model internally advances its own state
```

**New (integrator owns time integration):**
```cpp
struct PhysicsModel {
  double rhs(double t, const MyGrads& g) const noexcept {
    // Pure function: reads gradients, returns increment
    return /* point-wise physics */;
  }
};

// Usage
auto grad = pfc::field::create<MyGrads>(u, eval_args...);
auto stepper = pfc::sim::steppers::create(u, grad, model, dt);
t = stepper.step(t, u.vec());  // Integrator owns the accumulation
```

## Step-by-step migration

### 1. Extract physics into rhs(double t, const G& g)

Move the point-wise physics from `Model::step()` into a pure function:

```cpp
// Before: physics mixed with FFTs and time-stepping
void step(double t) override {
  auto& u = get_real_field();
  // ... FFTs ...
  for (each cell) {
    du[i] = /* physics using spectral derivatives */;
  }
  u += dt * du;  // Time-stepping logic owned by model
}

// After: physics isolated
struct HeatGrads {
  double xx{}, yy{}, zz{};
};

struct HeatModel {
  double kD;
  [[nodiscard]] double rhs(double t, const HeatGrads& g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);  // Pure function
  }
};
```

### 2. Choose backend and construct gradient evaluator

**Finite difference (needs halo exchange):**
```cpp
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>

pfc::sim::stacks::FDCPUStack stack(
  pfc::domain::create(pfc::GridSize{{N, N, N}}, pfc::PhysicalOrigin{{0.0, 0.0, 0.0}}, pfc::GridSpacing{{dx, dx, dx}}),
  fd_order, rank, nproc);

auto& u = stack.u();
auto grad = pfc::field::create<HeatGrads>(u, fd_order);
```

**Spectral (no halo exchange, FFTs in evaluator):**
```cpp
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

pfc::sim::stacks::SpectralCPUStack stack(
  pfc::domain::create(pfc::GridSize{{N, N, N}}, pfc::PhysicalOrigin{{0.0, 0.0, 0.0}}, pfc::GridSpacing{{dx, dx, dx}}),
  rank, nproc);

auto& u = stack.u();
auto grad = pfc::field::create<HeatGrads>(u, stack.fft());
```

### 3. Build stepper with factory

```cpp
#include <openpfc/kernel/simulation/steppers/euler.hpp>

HeatModel model{D};
auto stepper = pfc::sim::steppers::create(u, grad, model, dt);
```

### 4. Replace time loop

**FD backend (halo exchange required):**
```cpp
for (int step = 0; step < n_steps; ++step) {
  stack.exchange_halos();  // Must call before stepper.step()
  t = stepper.step(t, u.vec());
}
```

**Spectral backend (no halo exchange):**
```cpp
for (int step = 0; step < n_steps; ++step) {
  t = stepper.step(t, u.vec());  // FFTs happen inside evaluator.prepare()
}
```

## Prepare resident stages at their actual time

The resident CUDA/HIP coupled field factories publish a stage and call
`eval.prepare(t + c_i*dt)` before its point RHS. A composite device evaluator
accepts either a zero-argument hook or a hook taking `double stage_time`:

```cpp
eval.set_prepare([&](double stage_time) {
  assemble_auxiliaries(stage_time);  // application-owned physics
  halos.exchange();
});
```

The clock comes from the same Butcher tableau as the RHS clock. This matters
when a time-dependent auxiliary rate enters another flux or divergence.
Applications do not count stages themselves. Installing either hook replaces
the previous hook; `eval.set_prepare({})` clears preparation. Legacy
`eval.set_prepare([&] { halos.exchange(); })` works with both `prepare()` and
`prepare(t)`. With a timed hook, calling `prepare()` throws `std::logic_error`
because there is no stage time to supply.

The hook runs on the host and may launch device assembly kernels. It must
finish communication and reach any required collective agreement before
throwing. Stepper rollback restores the bound evolving fields; callers own
auxiliary restoration or rebuilding before retry. The existing
[resident coupled test](../../tests/integration/rk_coupled/resident.hpp)
checks stage clocks, prepared forcing, rejection/retry and legacy callbacks.

## Check a coupled final state before acceptance

A constraint on a nonlinear derived variable can fail after the final RK
combination even when every stage was valid. The host multi-field factory
and resident CUDA/HIP two-/three-/four-field factories expose
`step_checked(t, check, fields...)` for this case. The ordinary `step`
operation retains its existing behaviour.

The host callback `check(t + dt)` runs once with all final fields published.
It returns `void` and throws to reject the candidate. The complete padded
accepted state is then restored in the same field allocations, and the
same stepper can retry. Existing stage/backup buffers supply rollback;
no additional device allocation or full-state host upload is introduced.
A host callback can inspect a device field through its read-only
`with_host_read` mirror. That explicit diagnostic download is separate
from device-resident evolution.

The check must not modify or resize the bound evolving fields. It may
compute model-owned derived quantities, but rollback of those auxiliaries,
external clocks and ledgers belongs to the caller. In an MPI calculation,
the caller must agree any rank-local failure before throwing; the fixed
field factory does not infer a communicator. A rejected step returns no
new time. A failed stage also restores accepted fields and never calls the
final check.

## Complete working examples

See [`../../examples/heat/finite-differences/model.cpp`](../../examples/heat/finite-differences/model.cpp) for the FD backend and [`../../examples/heat/fourier/model.cpp`](../../examples/heat/fourier/model.cpp) for the spectral backend. Both solve the same heat equation with the same pointwise Laplacian combination and different gradient evaluators. These teaching examples display the Euler accumulation explicitly rather than calling the stepper factory.

## Stability considerations

For the periodic heat equation $u_t=D\nabla^2u$, forward Euler requires
$\Delta t\,|\lambda_{\min}|\le2$, where $\lambda_{\min}$ is the most negative
eigenvalue of the spatial operator. The integrator's stability region is the
same across backends; the spatial eigenvalues are different.

On a 3-D uniform grid of spacing $h$, the second-order central FD Laplacian
has $|\lambda_{\min}|\le12D/h^2$, giving $\Delta t\le h^2/(6D)$. Fourier
collocation on an even grid retains modes up to $\pi/h$ in each direction,
giving $|\lambda_{\min}|=3D\pi^2/h^2$ and the tighter
$\Delta t\le2h^2/(3\pi^2D)$. These are the separate bounds checked by the
linked runnable heat examples. Use a safety factor below the limit. Higher
FD orders and other PDEs require their own operator bound.

Fourier differentiation is exact for the retained trigonometric polynomial;
a finite grid still truncates unresolved modes, and nonlinear products can
alias. It does not remove the explicit time-step restriction. Implicit methods
or an ETD treatment of a diagonal linear operator change that restriction;
they do not guarantee accuracy or unconditional stability of a nonlinear split.

## Benefits of pluggable integrators

- **Swap algorithms:** Pass an RK2 or RK4 `ButcherTableau` to the `ExplicitRKStepper` factory without changing the pointwise RHS
- **Backend-agnostic physics:** Same `HeatModel::rhs()` works with FD or spectral gradients
- **Clear separation:** Integrator owns time-stepping, evaluator owns spatial derivatives, model owns point-wise physics
- **Testability:** Pure `rhs(t, g)` functions are easy to unit test

## Next steps

- ADR-0003 contracts: [`../adr/0003-time-integrator-interface.md`](../adr/0003-time-integrator-interface.md)
- Gradient evaluator reference: [`../extending_openpfc/per_point_grads.md`](../extending_openpfc/per_point_grads.md)
- FD vs spectral trade-offs: [`../adr/0002-gradient-operators-fd-vs-spectral.md`](../adr/0002-gradient-operators-fd-vs-spectral.md)
- Halo exchange policies: [`../concepts/halo_exchange.md`](../concepts/halo_exchange.md)
