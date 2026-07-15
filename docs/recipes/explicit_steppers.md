<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Recipe: Migrate from Model::step(double t) to explicit steppers

**Goal:** Transition legacy spectral apps using `Model::step(double t)` to the new pluggable integrator architecture with `pfc::sim::steppers::create()`, enabling replaceable integrators (Euler, RK2, RK4, IMEX) without rewriting physics.

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

pfc::sim::stacks::FdCpuStack stack(
  pfc::GridSize{{N, N, N}},
  pfc::PhysicalOrigin{{0.0, 0.0, 0.0}},
  pfc::GridSpacing{{dx, dx, dx}},
  fd_order, rank, nproc);

auto& u = stack.u();
auto grad = pfc::field::create<HeatGrads>(u, fd_order);
```

**Spectral (no halo exchange, FFTs in evaluator):**
```cpp
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

pfc::sim::stacks::SpectralCpuStack stack(
  pfc::GridSize{{N, N, N}},
  pfc::PhysicalOrigin{{0.0, 0.0, 0.0}},
  pfc::GridSpacing{{dx, dx, dx}},
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

## Complete working examples

See [`../../examples/19_explicit_stepper_fd.cpp`](../../examples/19_explicit_stepper_fd.cpp) for the FD backend and [`../../examples/20_explicit_stepper_spectral.cpp`](../../examples/20_explicit_stepper_spectral.cpp) for the spectral backend. Both solve the same heat equation with identical physics (`HeatModel::rhs`) but different gradient evaluators.

## Stability considerations

Explicit Euler has the same stability limit regardless of backend:

```cpp
// Heat equation: dt <= dx^2 / (6*D) in 3D
const double dt = 0.15 * dx * dx / (6.0 * D);  // Conservative safety factor
```

Spectral methods are exact in space but still time-step limited for explicit integration. For unconditional stability, consider implicit methods or the legacy spectral `Model::step()` path which uses implicit integration.

## Benefits of pluggable integrators

- **Swap algorithms:** Change `EulerStepper` to `RK2Stepper` or `RK4Stepper` without touching physics
- **Backend-agnostic physics:** Same `HeatModel::rhs()` works with FD or spectral gradients
- **Clear separation:** Integrator owns time-stepping, evaluator owns spatial derivatives, model owns point-wise physics
- **Testability:** Pure `rhs(t, g)` functions are easy to unit test

## Next steps

- ADR-0003 contracts: [`../adr/0003-time-integrator-interface.md`](../adr/0003-time-integrator-interface.md)
- Gradient evaluator reference: [`../extending_openpfc/per_point_grads.md`](../extending_openpfc/per_point_grads.md)
- FD vs spectral trade-offs: [`../adr/0002-gradient-operators-fd-vs-spectral.md`](../adr/0002-gradient-operators-fd-vs-spectral.md)
- Halo exchange policies: [`../concepts/halo_exchange.md`](../concepts/halo_exchange.md)
