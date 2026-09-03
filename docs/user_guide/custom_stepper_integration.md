<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Custom stepper integration

This guide shows how to integrate custom time steppers with OpenPFC's
explicit stepper composition (heat3d / wave2d / examples 19–21). Production
spectral apps (tungsten, aluminumNew) use the framework spectral ETD system on
`SimulationState` and drive it with the same `pfc::sim::run` loop; the stepper
leaves described here are the explicit FD / point-wise spectral path.

## Prerequisites

Before working with custom steppers, you should:

- Complete the [quickstart guide](../quickstart.md) to understand basic OpenPFC setup
- Read [ADR 0003: Time integrator interface contracts](../adr/0003-time-integrator-interface.md) for the formal contracts between integrators, models, and spatial discretizations
- Understand the [applications overview](applications.md) to see how the shipped apps compose physics, stacks, and steppers
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

For finite difference methods, use `pfc::field::FDGradient<G>` over a
halo-padded `pfc::data::Field<double>`. The `FDCPUStack` bundle owns such a
field together with its halo exchanger:

```cpp
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>

const int order = 4;  // even orders 2, 4, ..., 20 for second derivatives
pfc::sim::stacks::FDCPUStack stack(domain, order, rank, nproc, MPI_COMM_WORLD);

// Create the FD gradient evaluator over the stack's padded Field
auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), order);
```

The FD gradient evaluator:

- Reads geometry (grid size, spacing, halo width) directly from the `Field`
- Supports even orders 2–14 for first derivatives (`x, y, z`)
- Supports even orders 2–20 for second derivatives (`xx, yy, zz`)
- Performs stencil computation on-the-fly during evaluation (no pre-computation)
- Requires halo exchange before each step (`stack.exchange_halos()`)

**Note**: FD gradient evaluators need a corner-filled halo for mixed second
derivatives (`xy, xz, yz`); use `comm::HaloExchange` in `Full` connectivity mode
for those.

### Spectral gradient evaluator

For spectral methods, use `pfc::field::SpectralGradient<G>`:

```cpp
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>

// Assume we have FFT plan and field
pfc::fft::IHostFFT &fft = /* ... FFT plan ... */;
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

**Trade-offs**: Spectral evaluators support arbitrary point-wise RHS (enabling custom physics) but use explicit time integration (CFL-limited). For stiff problems, the spectral ETD system (2 FFTs/step, unconditionally stable linear part) is preferable.

## Stepper creation with factory pattern

OpenPFC provides a factory function `pfc::sim::steppers::create()` that binds a model, gradient evaluator, and time step into a stepper object:

```cpp
#include <openpfc/kernel/simulation/steppers/euler.hpp>

// Assume we have:
// - grad: gradient evaluator (FD or spectral)
// - model: physics model with rhs() method
// - u: the state (std::vector<double> or pfc::data::Field<double>)
// - dt: time step size

double dt = 0.01;

// Create stepper using factory (derives local_size from u.size())
auto stepper = pfc::sim::steppers::create(grad, model, dt, u.size());

// Alternative: pass the Field to derive size automatically
auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, dt);
```

The factory function:

- Constructs an `EulerStepper` by default, or an `ExplicitRKStepper` (RK2/RK4) when a `ButcherTableau` is passed as an extra argument -- see [Higher-order steppers](#higher-order-steppers-rk2-and-rk4) below
- Captures the gradient evaluator and model by reference (they must outlive the stepper)
- Creates an internal RHS lambda that calls `pfc::sim::for_each_interior(model, eval, du, t)`
- Allocates an internal scratch buffer `du` sized to match the field
- Returns a stepper object with a `step(double t, std::vector<double>& u)` method and the attempt/commit protocol (`attempt` / `commit_step_attempt`)

**Type inference**: The factory deduces all template arguments automatically. You don't need to specify the gradient type, model type, or RHS signature explicitly.

## Identifier-driven composition (`compose_scalar` / `compose_multi`)

Typed `steppers::create(Eval&, Model&, dt)` binds physics and a gradient
evaluator when the method type is already known at the call site. For
config- or string-driven selection, use the composition boundary in
[`method_composition.hpp`](../../include/openpfc/kernel/simulation/steppers/method_composition.hpp):

```cpp
#include <openpfc/kernel/simulation/steppers/method_composition.hpp>

pfc::sim::steppers::IntegratorComposeConfig cfg{.dt = 0.01,
                                                .requires_adaptive = false};
auto composition =
    pfc::sim::steppers::compose_scalar("euler", cfg, u.size(), rhs);
// composition.stepper, .workspace_ownership, .method_state
```

- `compose_scalar` / `compose_multi` validate `IntegratorComposeConfig`
  (positive `dt`, adaptive capability via `validate_method`) and return an
  `IntegratorComposition` whose stepper is an `ExplicitRKStepper` /
  `MultiExplicitRKStepper` built with `make_tableau(method)`, plus declared
  `WorkspaceOwnership` and optional `MethodStateCapability` (empty for these
  stateless fixed-step methods).
- Builtin RK ids: `euler`, `rk2_midpoint`, `rk2_heun`, `rk4_classical`.
  JSON `"etd1"` / `"imex_euler"` are registered identity tokens;
  `compose_scalar("etd1")` fail-closes. Construct with `compose_etd1` /
  `compose_imex_euler` (ETD needs coefficient spans; IMEX needs a
  `SolveFunction` and `LinearOperatorDesc`).
- Unknown identifiers, invalid config, and capability mismatches throw
  `ComposeError` before any step is taken.
- Extend the table with `register_method_composer` so new methods do not
  require driver-side method switches.

JSON FD CPU sessions (`json_fd_session.hpp`) compose the RK method from
`Time::method()` and step `FDCPUStack::u()` with a Laplacian heat RHS.

See also [Time integration architecture](../development/time_integration_architecture.md).

## Driving the stepper with `pfc::sim::run`

The free-function loop
[`pfc::sim::run`](../../include/openpfc/kernel/simulation/simulation_driver.hpp)
is the integration point for custom steppers. It owns the prologue/epilogue
contract (initial conditions, boundary conditions, result writing) through four
hooks and delegates the physics update to your stepper:

```cpp
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/time.hpp>

// Assume we have:
// - stack:   pfc::sim::stacks::FDCPUStack (owns the padded Field + halo exchanger)
// - model:   physics model
// - stepper: created from pfc::sim::steppers::create()

pfc::Time time({t0, t1, dt}, saveat);

pfc::sim::run(
    time,
    [&](double t) {                       // step
      stack.exchange_halos();
      (void)stepper.step(t, stack.u().vec());
    },
    [&](pfc::Time &) {                    // on_start (increment == 0)
      stack.u().apply(model.initial_condition);
    },
    [&](pfc::Time &) { /* boundary conditions at the new time */ },
    [&](const pfc::Time &tm) {            // on_save
      writer.write(pfc::time::increment(tm), stack.u());
    });
```

Per iteration, `pfc::sim::run`:

1. On the first iteration only, calls `on_start(time)` and then `on_save(time)` if `pfc::time::do_save(time)`
2. Calls `pfc::time::next(time)` and then `apply(time)` (boundary conditions at the new time)
3. Calls your `step(t)` with the accepted time
4. Calls `on_save(time)` if `pfc::time::do_save(time)`

This ordering contract is documented in [ADR 0003](../adr/0003-time-integrator-interface.md) and [time_integration_contract.md](../concepts/time_integration_contract.md). Your stepper may assume boundary conditions are already applied when `step` runs, and it must not touch fields other than the state it receives.

## Comparison: model-owned stepping (0.1) vs stepper composition (0.2)

### 0.1 pattern (removed)

In 0.1 the virtual `Model::step(t)` owned the time integration and a `Simulator`
called it; swapping the integration method meant editing the model. That
`Model` / `Simulator` pair was deleted in 0.2 (see
[`MIGRATION_0.1_to_0.2.md`](../MIGRATION_0.1_to_0.2.md)).

### 0.2 pattern (custom stepper composition)

```cpp
// Model is pure physics, stepper owns time integration
struct HeatModel {
  [[nodiscard]] double rhs(double t, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};

// Usage
auto grad = pfc::field::create<HeatGrads>(stack.u(), order);
auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, dt);

pfc::sim::run(time, [&](double t) {
  stack.exchange_halos();
  (void)stepper.step(t, stack.u().vec());
});
```

**Characteristics**:
- Model is pure physics (no time-stepping logic)
- Easy to swap integration methods (change stepper factory call)
- Supports arbitrary point-wise RHS (enables custom physics)
- RK2, RK4, IMEX, and adaptive methods work without model changes

### Migration steps

To port model-owned stepping to the 0.2 pattern:

1. **Extract point-wise physics**: Move the core physics computation into a `rhs(double t, const Grads&)` method
2. **Choose spatial discretization**: Decide between FD (`pfc::field::FDGradient`) or spectral (`pfc::field::SpectralGradient`)
3. **Build gradient evaluator**: Create the evaluator with appropriate parameters
4. **Create stepper**: Use `pfc::sim::steppers::create()` to bind model, evaluator, and time step
5. **Drive with `pfc::sim::run`**: call `stepper.step()` inside the `step` hook; ICs, BCs, and writers go in the other hooks

## Contract: hook ordering in `pfc::sim::run`

`pfc::sim::run` implements a strict ordering contract documented in [ADR 0003](../adr/0003-time-integrator-interface.md):

### Call sequence

```cpp
while (!pfc::time::done(time)) {
  if (pfc::time::increment(time) == 0) {
    on_start(time);
    if (pfc::time::do_save(time)) on_save(time);
  }
  pfc::time::next(time);
  apply(time);
  step(pfc::time::current(time));
  if (pfc::time::do_save(time)) on_save(time);
}
```

### Your `step` hook

- Execute `stepper.step(t, u)` or any custom physics update
- Boundary conditions have already been applied by `apply`
- Time has already advanced to the new value
- Must not modify fields other than the state it steps

### Key constraints

- **Initial conditions**: Run only on the first iteration when `increment == 0`
- **Boundary conditions**: Applied after `Time::next()` but before your `step` hook
- **Result writing**: Happens after IC application (first iteration) and/or after your `step` hook
- **Time advancement**: `Time::next()` runs on every iteration, including the first

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
auto rk4_stepper = pfc::sim::steppers::create(stack.u(), grad, model, dt, tableau);

pfc::sim::run(time, [&](double t) {
  stack.exchange_halos();
  (void)rk4_stepper.step(t, stack.u().vec());  // Fourth-order accurate
});
```

The key advantages:

- **Same model interface**: Your `rhs(double t, const Grads&)` method works unchanged
- **Same gradient evaluator**: FD and spectral evaluators work with any stepper
- **Same driver integration**: the `pfc::sim::run` hook pattern remains identical

### Implementation status

- ✅ **Euler stepper**: [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)
- ✅ **RK2 stepper**: [`include/openpfc/kernel/simulation/steppers/explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp) (`make_rk2_midpoint`/`make_rk2_heun` tableaus)
- ✅ **RK4 stepper**: [`include/openpfc/kernel/simulation/steppers/explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp) (`make_rk4_classical` tableau)
- ✅ **ETD1 stepper (CPU)**: Isolated-candidate exponential update in [`include/openpfc/kernel/simulation/steppers/etd1.hpp`](../../include/openpfc/kernel/simulation/steppers/etd1.hpp) (`ETD1Stepper` / `MultiETD1Stepper`, real or complex state); consumes diagonal `exp`/`phi1` spans from [`spectral_exp_coefficients.hpp`](../../include/openpfc/kernel/integrator/spectral_exp_coefficients.hpp). Production spectral apps use the framework spectral ETD system on `SimulationState` (host and device).
- ✅ **Embedded RK + adaptive controller**: `EmbeddedRKStepper` (`make_embedded_rk45` / `make_embedded_rk23`) with `AdaptiveTimeController` closing error estimate → `Time` attempt transactions (example 21)
- ✅ **IMEX stage-composition seam**: [`imex_stage_composition.hpp`](../../include/openpfc/kernel/simulation/steppers/imex_stage_composition.hpp) (`ImexEulerComposer`); sequences explicit eval then `SolveFunction` into an isolated candidate with `apply_candidate` commit
- ⏳ **IMEX methods**: First-order IMEX Euler is landed on CPU in [`include/openpfc/kernel/simulation/steppers/imex_euler.hpp`](../../include/openpfc/kernel/simulation/steppers/imex_euler.hpp) (`ImexEulerStepper` / `MultiImexEulerStepper` attempt/commit isolation with injected `SolveFunction` + `LinearOperatorDesc`). Higher-order IMEX-RK and production spectral diagonal solvers remain separate work (e.g. board #161).

Check the [refactoring roadmap](../development/refactoring_roadmap.md) for progress on higher-order stepper implementations.

## Complete working example

Putting it all together, here's a complete example showing FD gradient evaluation with explicit Euler stepping:

```cpp
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/time.hpp>

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
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // Global geometry
  auto domain = pfc::domain::create(
      pfc::GridSize({64, 64, 64}),
      pfc::PhysicalOrigin({-3.14, -3.14, -3.14}),
      pfc::GridSpacing({0.098, 0.098, 0.098}));

  // Padded Field + halo exchanger + decomposition in one bundle
  const int order = 4;  // 4th-order stencils, halo width 2
  pfc::sim::stacks::FDCPUStack stack(domain, order, rank, nproc, MPI_COMM_WORLD);

  // Gradient evaluator, model, and stepper
  auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), order);
  heat3d::HeatModel model;
  const double dt = 0.001;
  auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, dt);

  // Initial condition on the owned cells
  stack.u().apply([](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z));
  });

  // Time-stepping loop
  pfc::Time time({0.0, 1.0, dt}, /*saveat=*/0.1);
  pfc::sim::run(time, [&](double t) {
    stack.exchange_halos();
    (void)stepper.step(t, stack.u().vec());
  });

  MPI_Finalize();
  return 0;
}
```

This example demonstrates the complete path: from physics model through gradient evaluator to stepper creation and the `pfc::sim::run` driver.

## Additional resources

- **ADR 0003**: Formal contracts between integrators, models, and spatial discretizations
- **Heat3D application**: Production example using both FD and spectral paths ([`apps/heat3d/README.md`](../../apps/heat3d/README.md))
- **Wave2D application**: Multi-field model with tuple protocol ([`apps/wave2d/README.md`](../../apps/wave2d/README.md))
- **Gradient concepts**: Per-member detection and backend capabilities ([`include/openpfc/kernel/field/grad_concepts.hpp`](../../include/openpfc/kernel/field/grad_concepts.hpp))
- **DuField**: Stack-friendly residual field with prepare hooks ([`include/openpfc/kernel/simulation/du_field.hpp`](../../include/openpfc/kernel/simulation/du_field.hpp))
- **Refactoring roadmap**: Track progress on higher-order stepper implementations ([`docs/development/refactoring_roadmap.md`](../development/refactoring_roadmap.md))
