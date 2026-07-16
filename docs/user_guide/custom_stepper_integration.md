<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Custom stepper integration

This guide shows how to integrate custom time steppers with OpenPFC's `Simulator::step_with_physics()` method. The migration path from the legacy `Model::step(double t)` pattern to explicit stepper composition enables replaceable time integration methods (Euler, RK2, RK4, IMEX) without rewriting model physics code.

## Prerequisites

Before working with custom steppers, you should:

- Complete the [quickstart guide](../quickstart.md) to understand basic OpenPFC setup
- Read [ADR 0003: Time integrator interface contracts](../adr/0003-time-integrator-interface.md) for the formal contracts between integrators, models, and spatial discretizations
- Understand the [applications overview](applications.md) to see how spectral apps use the legacy `Model::step()` pattern
- Review the [installation guide](../../INSTALL.md) for building OpenPFC with the required dependencies

## Physics model with `rhs()` method

The foundation of custom stepper integration is a physics model that provides a `rhs(double t, const Grads&)` method. This method computes the right-hand side (time derivative) for each grid point based on the current time and spatial gradients.

### Minimal heat equation model

Following the pattern from [`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp), here's a minimal model for the 3D heat equation \(\partial_t u = D \nabla^2 u\):

```cpp
#include <cmath>

namespace heat3d {

// Diffusion coefficient (shared across all heat3d binaries)
inline constexpr double kD = 1.0;

/**
 * @brief Per-point gradient aggregate for the heat equation.
 *
 * Only the unmixed second derivatives are needed, so we declare exactly those.
 * The OpenPFC gradient evaluators will fill only these members.
 */
struct HeatGrads {
  double xx{};  // ∂²u/∂x²
  double yy{};  // ∂²u/∂y²
  double zz{};  // ∂²u/∂z²
};

/**
 * @brief Heat equation physics model.
 *
 * The model is const-correct: the rhs() method does not modify any state.
 * This enables pluggable integrators because the model has no ownership
 * of time-stepping logic.
 */
struct HeatModel {
  /**
   * @brief Right-hand side: ∂t u = D (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
   *
   * @param t Current time (not used for constant-coefficient diffusion)
   * @param g Spatial gradients at the current point
   * @return Time derivative du/dt at the current point
   */
  [[nodiscard]] double rhs(double /*t*/, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};

} // namespace heat3d
```

Key points about this model:

- **Header-only**: The model file contains only standard library includes (`<cmath>`). No OpenPFC headers are needed, making the model trivial to unit-test in isolation.
- **Const-correct**: The `rhs()` method is `const` and `noexcept`, enabling aggressive compiler optimizations.
- **Minimal aggregate**: `HeatGrads` declares only the derivatives actually needed. The kernel uses compile-time detection to fill only these members.
- **No state mutation**: The model doesn't modify any fields during RHS evaluation, enabling pluggable integrators.

## Gradient evaluator construction

The gradient evaluator bridges the spatial discretization (FD or spectral) with the physics model. OpenPFC provides two main evaluator types: finite difference and spectral.

### Finite difference gradient evaluator

For finite difference methods, use `pfc::gradient::FDGradient<G>`:

```cpp
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

// Assume we have a PaddedBrick field containing the solution
pfc::field::PaddedBrick<double> u = /* ... initialized field ... */;

// Create FD gradient evaluator (default: second-order central differences)
pfc::gradient::FDGradient<heat3d::HeatGrads> grad(u);

// Or specify an explicit even order (2, 4, 6, ..., 20 for second derivatives)
pfc::gradient::FDGradient<heat3d::HeatGrads> grad(u, 4);  // fourth-order
```

The FD gradient evaluator:

- Reads geometry (grid size, spacing, halo width) directly from the `PaddedBrick`
- Supports even orders 2–14 for first derivatives (`x, y, z`)
- Supports even orders 2–20 for second derivatives (`xx, yy, zz`)
- Performs stencil computation on-the-fly during evaluation (no pre-computation)
- Requires halo exchange before each step (application responsibility)

**Note**: FD gradient evaluators cannot compute mixed second derivatives (`xy, xz, yz`) because they would require corner-filled halos. Attempting to use these members triggers a compile-time error.

### Spectral gradient evaluator

For spectral methods, use `pfc::field::SpectralGradient<G>`:

```cpp
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>

// Assume we have FFT plan and field
pfc::fft::IFFT &fft = /* ... FFT plan ... */;
std::vector<double> u = /* ... field data ... */;
std::array<int, 3> global_size = {64, 64, 64};
std::array<double, 3> spacing = {0.1, 0.1, 0.1};

// Get local bounds from FFT
auto inbox = fft.get_inbox_bounds();
auto outbox = fft.get_outbox_bounds();

// Create spectral gradient evaluator
pfc::field::SpectralGradient<heat3d::HeatGrads> grad(
    fft, u, global_size, spacing, inbox, outbox);
```

The spectral gradient evaluator:

- Computes derivatives via FFT: forward FFT of input field, then spectral multiplication, then inverse FFT per derivative
- Supports all derivative types including mixed second derivatives (`xy, xz, yz`)
- Requires one forward FFT and one inverse FFT per requested derivative member in each `prepare()` call
- Performs all FFT work internally during `prepare()` (no halo exchange needed)
- Trades explicit time-stepping CFL limits for unconditional stability of the spectral spatial operator

**Trade-offs**: Spectral evaluators support arbitrary point-wise RHS (enabling custom physics) but use explicit time integration (CFL-limited). For pure constant-coefficient diffusion, the implicit-Fourier path (2 FFTs/step, unconditionally stable) may be preferable.

## Stepper creation with factory pattern

OpenPFC provides a factory function `pfc::sim::steppers::create()` that binds a model, gradient evaluator, and time step into a stepper object:

```cpp
#include <openpfc/kernel/simulation/steppers/euler.hpp>

// Assume we have:
// - grad: gradient evaluator (FD or spectral)
// - model: physics model with rhs() method
// - u: field vector (std::vector<double> or LocalField)
// - dt: time step size

double dt = 0.01;

// Create stepper using factory (derives local_size from u.size())
auto stepper = pfc::sim::steppers::create(grad, model, dt, u.size());

// Alternative: pass LocalField to derive size automatically
pfc::field::LocalField<double> u_local = /* ... */;
auto stepper = pfc::sim::steppers::create(u_local, grad, model, dt);
```

The factory function:

- Constructs an `EulerStepper` by default, or an `ExplicitRKStepper` (RK2/RK4) when a `ButcherTableau` is passed as an extra argument -- see [Higher-order steppers](#higher-order-steppers-rk2-and-rk4) below
- Captures the gradient evaluator and model by reference (they must outlive the stepper)
- Creates an internal RHS lambda that calls `pfc::sim::for_each_interior(model, eval, du, t)`
- Allocates an internal scratch buffer `du` sized to match the field
- Returns a stepper object with a `step(double t, std::vector<double>& u)` method

**Type inference**: The factory deduces all template arguments automatically. You don't need to specify the gradient type, model type, or RHS signature explicitly.

## Simulator integration with `step_with_physics()`

The `Simulator::step_with_physics()` method provides the integration point for custom steppers. It handles the prologue/epilogue contract (initial conditions, boundary conditions, result writing) while delegating the physics update to your stepper:

```cpp
#include <openpfc/kernel/simulation/simulator.hpp>

// Assume we have:
// - model: physics model
// - world: World object
// - fft: FFT plan
// - u: field vector
// - grad: gradient evaluator
// - stepper: created from pfc::sim::steppers::create()
// - dt: time step size

// Create time and simulator
double t0 = 0.0;
double t1 = 1.0;
double saveat = 0.1;
pfc::Time time({t0, t1, dt}, saveat);
pfc::Simulator sim(model, time);

// Add initial conditions, results writers, etc.
sim.add_initial_conditions(/* ... */);
sim.add_results_writer("u", /* ... */);

// Initialize the simulator (applies initial conditions)
pfc::initialize(sim);

// Time-stepping loop with custom stepper
double t = t0;
while (!sim.done()) {
  sim.step_with_physics([&]() {
    t = stepper.step(t, u);  // Advance one step with the custom stepper
  });
}
```

The `step_with_physics()` method:

1. Calls `begin_integrator_step()`: applies initial/boundary conditions, advances time, applies boundary conditions at new time
2. Executes your physics lambda (typically `stepper.step(t, u)`)
3. Calls `end_integrator_step()`: writes results if at a save point

This ordering contract is documented in [ADR 0003](../adr/0003-time-integrator-interface.md). Your stepper must respect that boundary conditions have already been applied when your lambda runs, and it must not modify model fields outside the `u` buffer it receives.

## Legacy comparison: `Model::step()` vs custom stepper

### Legacy pattern (spectral apps)

The legacy pattern used by spectral apps like `examples/05_simulator.cpp`:

```cpp
// Legacy: model owns time integration
class Diffusion : public Model {
  void step(double t) override {
    // Direct FFT-based time step (implicit Fourier)
    fft.forward(psi, psi_F);
    for (int k = 0; k < psi_F.size(); k++) {
      psi_F[k] = opL[k] * psi_F[k];  // Apply implicit operator
    }
    fft.backward(psi_F, psi);
  }
};

// Usage
Simulator sim(model, time);
while (!sim.done()) {
  sim.step();  // Calls model.step(t) internally
}
```

**Characteristics**:
- Model owns the time-stepping logic
- Hard to swap integration methods (requires modifying model code)
- Efficient for specific problems (e.g., implicit Fourier for diffusion)
- Limited to patterns implemented in the model

### New pattern (custom stepper composition)

The new pattern with explicit stepper composition:

```cpp
// New: model is pure physics, stepper owns time integration
struct HeatModel {
  [[nodiscard]] double rhs(double t, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};

// Usage
auto grad = pfc::gradient::FDGradient<HeatGrads>(u);
auto stepper = pfc::sim::steppers::create(grad, model, dt, u.size());

Simulator sim(model, time);
while (!sim.done()) {
  sim.step_with_physics([&]() {
    t = stepper.step(t, u);
  });
}
```

**Characteristics**:
- Model is pure physics (no time-stepping logic)
- Easy to swap integration methods (change stepper factory call)
- Supports arbitrary point-wise RHS (enables custom physics)
- Enables future RK2, RK4, IMEX methods without model changes

### Migration steps

To migrate from legacy to new pattern:

1. **Extract point-wise physics**: Move the core physics computation from `Model::step()` into a `rhs(double t, const Grads&)` method
2. **Choose spatial discretization**: Decide between FD (`pfc::gradient::FDGradient`) or spectral (`pfc::field::SpectralGradient`)
3. **Build gradient evaluator**: Create the evaluator with appropriate parameters
4. **Create stepper**: Use `pfc::sim::steppers::create()` to bind model, evaluator, and time step
5. **Replace `sim.step()`**: Use `sim.step_with_physics()` with a lambda calling `stepper.step()`

## Contract: `begin_integrator_step()` / `end_integrator_step()` ordering

The `Simulator::step_with_physics()` method implements a strict ordering contract documented in [ADR 0003](../adr/0003-time-integrator-interface.md):

### Call sequence

```cpp
void step_with_physics(PhysicsFn&& physics_fn) {
  begin_integrator_step();        // Prologue
  std::forward<PhysicsFn>(physics_fn)();  // Your stepper.step() call
  end_integrator_step();          // Epilogue
}
```

### `begin_integrator_step()` prologue

1. **First call only** (`increment == 0`):
   - Apply initial conditions
   - Apply boundary conditions
   - Write results if `Time::do_save()` is true
2. **Every call**:
   - Call `Time::next()` (increment advances, current time updates)
   - Apply boundary conditions at the new time

### Your physics lambda

- Execute `stepper.step(t, u)` or any custom physics update
- Boundary conditions have already been applied
- Time has already advanced to the new value
- Must not modify model fields outside the `u` buffer

### `end_integrator_step()` epilogue

- Write results if `Time::do_save()` is true at the new time

### Key constraints

- **Initial conditions**: Run only on the first iteration when `increment == 0`
- **Boundary conditions**: Applied after `Time::next()` but before your physics lambda
- **Result writing**: Happens after IC application (first call) and/or after your physics lambda
- **Time advancement**: `Time::next()` runs on every call, including the first

Your stepper must respect that boundary conditions are already valid when `stepper.step(t, u)` is called, and it must not rely on any specific timing of halo exchanges beyond what your gradient evaluator's `prepare()` method provides.

## Higher-order steppers: RK2 and RK4

The explicit stepper composition pattern generalizes to higher-order Runge-Kutta methods. OpenPFC implements RK2 and RK4 steppers in [`include/openpfc/kernel/simulation/steppers/explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp), built on top of the validated coefficient tables in [`include/openpfc/kernel/simulation/steppers/butcher_tableau.hpp`](../../include/openpfc/kernel/simulation/steppers/butcher_tableau.hpp).

### Butcher tableau infrastructure

The `ButcherTableau<T>` class template represents validated explicit Runge-Kutta method coefficients:

```cpp
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>

// Example: classic fourth-order Runge-Kutta coefficients
constexpr std::array<std::array<double, 4>, 4> rk4_a = {
  {{0.0, 0.0, 0.0, 0.0}},
  {{0.5, 0.0, 0.0, 0.0}},
  {{0.0, 0.5, 0.0, 0.0}},
  {{0.0, 0.0, 1.0, 0.0}}
};
constexpr std::array<double, 4> rk4_b = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
constexpr std::array<double, 4> rk4_c = {0.0, 0.5, 0.5, 1.0};

ButcherTableau<double> tableau(rk4_a, rk4_b, rk4_c);
```

The infrastructure includes:

- **Validation**: Ensures explicit lower-triangular structure, row-sum consistency, finite coefficients
- **Type safety**: Template parameter `T` must be a real floating-point type (float or double)
- **Immutable coefficients**: Tableaus are validated at construction and then read-only

Ready-made tableaus are available as factory functions in `butcher_tableau.hpp`: `make_rk2_midpoint<double>()`, `make_rk2_heun<double>()`, and `make_rk4_classical<double>()`.

### RK2/RK4 steppers

`ExplicitRKStepper` (single-field) and `MultiExplicitRKStepper` (multi-field) in `explicit_rk.hpp` consume a `ButcherTableau<double>` to implement any explicit RK method. They follow the same pattern as `EulerStepper`: own `dt`, pre-allocate scratch buffers, and take a user-supplied RHS. The `pfc::sim::steppers::create` factory overload that takes a tableau builds one directly from a gradient evaluator and model, mirroring the `EulerStepper` factory used earlier in this guide:

```cpp
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>

auto tableau = pfc::sim::steppers::make_rk4_classical<double>();
auto rk4_stepper = pfc::sim::steppers::create(grad, model, dt, u.size(), tableau);

sim.step_with_physics([&]() {
  t = rk4_stepper.step(t, u);  // Fourth-order accurate
});
```

The key advantages:

- **Same model interface**: Your `rhs(double t, const Grads&)` method works unchanged
- **Same gradient evaluator**: FD and spectral evaluators work with any stepper
- **Same Simulator integration**: `step_with_physics()` pattern remains identical

### Implementation status

- ✅ **Euler stepper**: Fully implemented in [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)
- ✅ **RK2 stepper**: Fully implemented in [`include/openpfc/kernel/simulation/steppers/explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp) (`make_rk2_midpoint`/`make_rk2_heun` tableaus)
- ✅ **RK4 stepper**: Fully implemented in [`include/openpfc/kernel/simulation/steppers/explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp) (`make_rk4_classical` tableau)
- ❌ **Adaptive RK stepper**: Exists on branch `ahojukka5/work-0070-add-adaptive-runge-kutta-stepper-with`, not yet merged to master
- ⏳ **IMEX methods**: Design phase, no implementation yet

Check the [refactoring roadmap](../development/refactoring_roadmap.md) for progress on higher-order stepper implementations.

## Complete working example

Putting it all together, here's a complete example showing FD gradient evaluation with explicit Euler stepping:

```cpp
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/du_field.hpp>

// Physics model (from earlier example)
namespace heat3d {
  struct HeatGrads { double xx{}, yy{}, zz{}; };
  inline constexpr double kD = 1.0;
  struct HeatModel {
    [[nodiscard]] double rhs(double, const HeatGrads& g) const noexcept {
      return kD * (g.xx + g.yy + g.zz);
    }
  };
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Setup world and decomposition
  auto world = pfc::world::create(
      pfc::GridSize({64, 64, 64}),
      pfc::PhysicalOrigin({-3.14, -3.14, -3.14}),
      pfc::GridSpacing({0.098, 0.098, 0.098}));

  auto decomp = pfc::decomposition::create(world, MPI_COMM_WORLD);
  auto fft = pfc::fft::create(decomp);

  // Create field and gradient evaluator
  pfc::field::PaddedBrick<double> u(fft.get_inbox_bounds(), 2);
  pfc::gradient::FDGradient<heat3d::HeatGrads> grad(u, 4);  // 4th order

  // Create model and stepper
  heat3d::HeatModel model;
  double dt = 0.001;
  auto stepper = pfc::sim::steppers::create(grad, model, dt, u.size());

  // Setup simulator
  pfc::Time time({0.0, 1.0, dt}, 0.1);
  pfc::Simulator sim(model, time);
  pfc::initialize(sim);

  // Time-stepping loop
  double t = 0.0;
  while (!sim.done()) {
    sim.step_with_physics([&]() {
      t = stepper.step(t, u);
    });
  }

  MPI_Finalize();
  return 0;
}
```

This example demonstrates the complete migration path: from physics model through gradient evaluator to stepper creation and Simulator integration.

## Additional resources

- **ADR 0003**: Formal contracts between integrators, models, and spatial discretizations
- **Heat3D application**: Production example using both FD and spectral paths ([`apps/heat3d/README.md`](../../apps/heat3d/README.md))
- **Wave2D application**: Multi-field model with tuple protocol ([`apps/wave2d/README.md`](../../apps/wave2d/README.md))
- **Gradient concepts**: Per-member detection and backend capabilities ([`include/openpfc/kernel/field/grad_concepts.hpp`](../../include/openpfc/kernel/field/grad_concepts.hpp))
- **DuField**: Stack-friendly residual field with prepare hooks ([`include/openpfc/kernel/simulation/du_field.hpp`](../../include/openpfc/kernel/simulation/du_field.hpp))
- **Refactoring roadmap**: Track progress on higher-order stepper implementations ([`docs/development/refactoring_roadmap.md`](../development/refactoring_roadmap.md))
