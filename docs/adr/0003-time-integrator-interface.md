<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0003: Time integrator interface contracts

## Status

Proposed

## Context

OpenPFC has multiple integration paths: `Model::step(double t)` for legacy spectral apps, `EulerStepper`/`MultiEulerStepper` for new explicit methods, and `DuField` for ergonomic single-field loops. However, there is no unified design documentation specifying contracts between integrators, models, and spatial discretizations. As new time integration methods (RK2, RK4, IMEX) are added, without explicit contracts each stepper could evolve different assumptions about state ownership, halo timing, or multi-field composition. This ADR establishes explicit boundaries to enable replaceable integrators without rewriting model physics, advancing the long-term goal of "replaceable integrators without rewriting model physics" referenced in [`docs/development/refactoring_roadmap.md`](../development/refactoring_roadmap.md).

## Decision

The following six contract areas define the interface boundaries between integrators, models, and spatial discretizations in OpenPFC:

### 1. Integrator surface API contracts

The `Simulator::step()` method isolates integrator stages via prologue/epilogue hooks defined in [`include/openpfc/kernel/simulation/simulator_integrator.hpp`](../../include/openpfc/kernel/simulation/simulator_integrator.hpp):

```cpp
inline void Simulator::begin_integrator_step() {
  simulator_integrator::begin_integrator_step(*this);
}

inline void Simulator::end_integrator_step() {
  simulator_integrator::end_integrator_step(*this);
}
```

The ordering contract implemented in `simulator_integrator::begin_integrator_step()`:
1. On first call (`increment == 0`): apply initial conditions, apply boundary conditions, optionally write results if `do_save()`
2. Call `pfc::time::next()` (increment advances; current time becomes `t0 + increment * dt`, clamped to `t1`)
3. Apply boundary conditions at the new time
4. Call `Model::step()` or custom physics body with the new current time
5. Call `end_integrator_step()` to write results if at a save point

This contract ensures that initial conditions run only on the first step, boundary conditions apply before every physics update, and result writing happens at the appropriate times. Integrators using `begin_integrator_step()` and `end_integrator_step()` must respect this ordering.

### 2. State access patterns

Two distinct RHS evaluation patterns are supported:

**Legacy field-only RHS pattern (spectral apps):**
```cpp
// include/openpfc/kernel/simulation/model.hpp
class Model {
  virtual void step(double t);  // directly operates on get_real_field()
  // Model owns the field and updates it in place via FFTs
};
```

**Gradient-access pattern (new explicit methods):**
```cpp
// include/openpfc/kernel/simulation/for_each_interior.hpp
template <class Model, class Eval, class DuOut>
inline void for_each_interior(const Model &model, Eval &eval, DuOut du, double t) {
  // Calls model.rhs(t, eval(i,j,k)) and scatters to du
  // Model is const; only du is written
}

// Example from apps/wave2d/include/wave2d/wave_model.hpp
struct WaveModel {
  [[nodiscard]] WaveIncrements rhs(double /*t*/, double v_val,
                                   const WaveLaplacian &lap) const noexcept {
    const double lap_u = inv_dx2 * lap.lxx + inv_dy2 * lap.lyy;
    return WaveIncrements{v_val, kC * kC * lap_u};
  }
};
```

The gradient-access pattern treats the physics model as a pure function `rhs(t, g) → du` that reads spatial gradients but does not modify any state directly. This enables pluggable integrators because the model has no ownership of the time-stepping logic.

### 3. Workspace ownership

Stage state lives in integrator allocations, not in Model fields. Model fields are read-only during integrator stages.

**EulerStepper owns scratch buffer:**
```cpp
// include/openpfc/kernel/simulation/steppers/euler.hpp
template <class Rhs> class EulerStepper {
  template <class U> double step(double t, std::vector<double> &u) {
    m_rhs(t, u, m_du);  // fills m_du with increments
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] += m_dt * m_du[static_cast<std::size_t>(li)];
    }
    return t + m_dt;
  }
private:
  double m_dt{0.0};
  std::vector<double> m_du;  // owned by integrator
  Rhs m_rhs;
};
```

**DuField owns internal buffer:**
```cpp
// include/openpfc/kernel/simulation/du_field.hpp
template <class G, class Eval> class DuField {
  template <class PrepareFn>
  DuField(std::size_t local_size, Eval eval, PrepareFn &&prepare_parent)
      : m_data(local_size, 0.0), m_eval(std::move(eval)),
        m_prepare(std::forward<PrepareFn>(prepare_parent)) {}

  template <class RhsFn> void apply(RhsFn &&rhs_fn, double t = 0.0) {
    m_prepare();  // FD halo exchange or spectral no-op
    LambdaModel<std::decay_t<RhsFn>> model{std::forward<RhsFn>(rhs_fn)};
    pfc::sim::for_each_interior(model, m_eval, m_data.data(), t);
  }

private:
  std::vector<double> m_data;  // owned by DuField
  Eval m_eval;
  std::function<void()> m_prepare;
};
```

**Multi-field MultiEulerStepper owns per-field buffers:**
```cpp
// include/openpfc/kernel/simulation/steppers/euler.hpp
template <class Rhs, std::size_t N> class MultiEulerStepper {
private:
  double m_dt{0.0};
  std::array<std::vector<double>, N> m_du;  // one buffer per field
  Rhs m_rhs;
};
```

The integrator owns the `du` buffer and is responsible for accumulating `u += dt * du`. The model's fields are read-only during RHS evaluation; any updates happen via the stepper's accumulation step.

### 4. Multi-field composition boundaries

The tuple protocol from [`include/openpfc/kernel/field/tuple_protocol.hpp`](../../include/openpfc/kernel/field/tuple_protocol.hpp) enables multi-field models to bundle increments into a single record:

```cpp
// include/openpfc/kernel/field/tuple_protocol.hpp
namespace pfc::field::detail {

namespace sfinae {
template <class T, class = void>
struct has_as_tuple : std::false_type {};

template <class T>
struct has_as_tuple<T, std::void_t<decltype(std::declval<T &>().as_tuple())>>
    : std::true_type {};

template <class T> struct is_std_tuple : std::false_type {};
template <class... Ts> struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template <class T>
struct is_tuple : is_std_tuple<std::remove_cv_t<std::remove_reference_t<T>>> {};
} // namespace sfinae

// Boolean predicates preserve former concept call syntax (`has_as_tuple<T>`).
template <class T>
inline constexpr bool has_as_tuple = sfinae::has_as_tuple<T>::value;
template <class T>
inline constexpr bool is_tuple = sfinae::is_tuple<T>::value;

/**
 * @brief Normalize `t` into a tuple-like view for fan-out.
 *
 * Returns `t.as_tuple()` if `T` opts in, `t` itself if it is already a
 * `std::tuple`, otherwise `std::forward_as_tuple(t)` (one-element view).
 * Host-oriented — device multi-field scatter uses `DevicePtrPackN` instead.
 */
template <class T> constexpr decltype(auto) to_tuple(T &t) {
  if constexpr (has_as_tuple<T>) {
    return t.as_tuple();  // user-defined opt-in
  } else if constexpr (is_tuple<T>) {
    return (t);  // std::tuple accepted as-is
  } else {
    return std::forward_as_tuple(t);  // scalar handled as 1-tuple
  }
}

} // namespace pfc::field::detail
```

**Example from wave2d WaveModel:**
```cpp
// apps/wave2d/include/wave2d/wave_model.hpp
/** Increments \f$(du, dv)\f$ for `MultiEulerStepper` tuple protocol. */
struct WaveIncrements {
  double du = 0.0;
  double dv = 0.0;
  auto as_tuple() { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};
```

**MultiEulerStepper factory wiring:**
```cpp
// include/openpfc/kernel/simulation/steppers/euler.hpp
template <class... Ts, class Eval, class Model>
[[nodiscard]] auto create(std::tuple<pfc::field::LocalField<Ts> &...> fields,
                          Eval &eval, const Model &model, double dt) {
  constexpr std::size_t N = sizeof...(Ts);
  // ... build sizes array ...

  auto rhs = [&eval, &model](double t, auto & /*u_tuple*/, auto &du_tuple) {
    auto du_ptrs = std::apply(
        [](auto &...vs) { return std::make_tuple(vs.data()...); }, du_tuple);
    pfc::sim::for_each_interior(model, eval, du_ptrs, t);
  };
  return MultiEulerStepper<decltype(rhs), N>(dt, sizes, std::move(rhs));
}
```

The tuple protocol allows `for_each_interior` to scatter multi-field increments into the correct per-field `du` buffers without requiring the integrator to know the field names or order.

### 5. Halo exchange timing expectations

Halo exchange timing differs between finite-difference and spectral backends:

**Finite-difference: pre-RHS via prepare_parent hook**
```cpp
// include/openpfc/kernel/simulation/du_field.hpp
template <class RhsFn> void DuField::apply(RhsFn &&rhs_fn, double t = 0.0) {
  m_prepare();  // FD: MPI halo exchange via PaddedHaloExchanger or SparseHaloExchanger
  LambdaModel<std::decay_t<RhsFn>> model{std::forward<RhsFn>(rhs_fn)};
  pfc::sim::for_each_interior(model, m_eval, m_data.data(), t);
  // m_eval.prepare() is a no-op for FD evaluators
}
```

For FD, the application-supplied `prepare_parent` callable performs the exchange before gradients are computed. This is typically a `pfc::communication::PaddedHaloExchanger<T>` for classical padded-brick layouts or a `pfc::SparseHaloExchanger<T>` for FFT-safe separated layouts. See [`docs/concepts/halo_exchange.md`](../concepts/halo_exchange.md) for halo policies and exchange patterns.

**Spectral: internal via Eval::prepare()**
```cpp
// include/openpfc/kernel/field/spectral_gradient.hpp
template <class G> class SpectralGradient {
  void prepare() {
    m_fft->forward(*m_u_in, m_u_F);  // FFT happens inside eval
    if constexpr (has_x<G>) invert_complex_op(m_op_x, m_dx);
    if constexpr (has_y<G>) invert_complex_op(m_op_y, m_dy);
    if constexpr (has_z<G>) invert_complex_op(m_op_z, m_dz);
    if constexpr (has_xx<G>) invert_real_op(m_op_xx, m_dxx);
    // ... spectral multiplies and inverse FFTs per requested member
  }
};
```

Spectral evaluators run forward FFTs and spectral multiplies inside `eval.prepare()`, which is called by `for_each_interior` at the start of each RHS evaluation. No external halo exchange is needed for pure spectral methods. The `prepare_parent` hook is typically a no-op for spectral.

**Timing contract:**
- FD: `m_prepare()` (halo exchange) → `eval.prepare()` (no-op) → `for_each_interior` loop
- Spectral: `m_prepare()` (no-op) → `eval.prepare()` (FFT + spectral multiplies) → `for_each_interior` loop

### 6. Migration path from Model::step(double t) to explicit integrator composition

**Legacy pattern (spectral apps):**
```cpp
// The model owns time integration
model.step(t);  // updates internal fields directly via FFTs
```

**New pattern (explicit integrator composition):**
```cpp
// include/openpfc/kernel/simulation/steppers/euler.hpp
template <class Eval, class Model>
[[nodiscard]] auto create(Eval &eval, const Model &model, double dt,
                          std::size_t local_size) {
  auto rhs = [&eval, &model](double t, const std::vector<double> & /*u*/,
                             std::vector<double> &du) {
    pfc::sim::for_each_interior(model, eval, du.data(), t);
  };
  return EulerStepper<decltype(rhs)>(dt, local_size, std::move(rhs));
}

// Usage
auto grad = pfc::field::create<MyGrads>(u, fft);
auto stepper = pfc::sim::steppers::create(grad, model, dt, u.size());
t = stepper.step(t, u);  // integrator owns the step logic
```

**Migration steps:**
1. Extract point-wise physics into a `rhs(double t, const G& g)` callable
2. Build a gradient evaluator (`pfc::field::FdGradient<G>` or `pfc::field::SpectralGradient<G>`)
3. Use `pfc::sim::steppers::create` factory to bind model + evaluator + time step
4. Call `stepper.step(t, u)` instead of `model.step(t)`

**Simulator integration:**
```cpp
// Using stepper instead of Model::step(simulator, model)
auto stepper = pfc::sim::steppers::create(eval, model, dt, u.size());
sim.step_with_physics([&]() {
  t = stepper.step(t, u);
});
```

The model no longer owns time integration, enabling pluggable steppers (RK2, RK4, IMEX) without modifying model physics code.

## Consequences

- New integrator methods (RK2, RK4, IMEX) can be added without modifying model physics code by following the documented contracts
- Model authors can choose between legacy `Model::step(double t)` (spectral apps) and explicit integrator composition (new explicit methods) based on their needs
- Halo policy and gradient evaluator choices are now documented, enabling backend-agnostic numeric expectations
- Integration path divergence is minimized through explicit interface contracts
- Future integrators must respect the `begin_integrator_step()` / `end_integrator_step()` ordering and must not modify Model fields during RHS evaluation
- Spectrum of abstraction: `DuField` (compact single-field) → `EulerStepper` (stepper-owns logic) → `Model::step` (legacy model-owns logic) provides multiple entry points with clear contracts

## See also

- [`include/openpfc/kernel/simulation/simulator.hpp`](../../include/openpfc/kernel/simulation/simulator.hpp) — `Simulator::step()` orchestration
- [`include/openpfc/kernel/simulation/simulator_integrator.hpp`](../../include/openpfc/kernel/simulation/simulator_integrator.hpp) — `begin_integrator_step()` / `end_integrator_step()` implementation
- [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp) — `EulerStepper` and `MultiEulerStepper` implementation
- [`include/openpfc/kernel/simulation/for_each_interior.hpp`](../../include/openpfc/kernel/simulation/for_each_interior.hpp) — Canonical point-wise driver loop
- [`include/openpfc/kernel/field/tuple_protocol.hpp`](../../include/openpfc/kernel/field/tuple_protocol.hpp) — Multi-field bundling convention
- [`include/openpfc/kernel/simulation/du_field.hpp`](../../include/openpfc/kernel/simulation/du_field.hpp) — Stack-friendly residual field with `prepare_parent` hooks
- [`include/openpfc/kernel/field/spectral_gradient.hpp`](../../include/openpfc/kernel/field/spectral_gradient.hpp) — Spectral evaluator with internal `prepare()` for FFTs
- [`include/openpfc/kernel/simulation/model.hpp`](../../include/openpfc/kernel/simulation/model.hpp) — Legacy `Model::step(double t)` interface
- [`apps/wave2d/include/wave2d/wave_model.hpp`](../../apps/wave2d/include/wave2d/wave_model.hpp) — Example multi-field model with tuple protocol
- [`docs/concepts/halo_exchange.md`](../concepts/halo_exchange.md) — Halo exchange policies, timing, and separated layout recommendation
- [`docs/science/numerics_limits.md`](../science/numerics_limits.md) — Backend-specific stability constraints
- [`docs/adr/0002-gradient-operators-fd-vs-spectral.md`](0002-gradient-operators-fd-vs-spectral.md) — Spatial operator directionality (FD vs spectral)
- [`docs/development/refactoring_roadmap.md`](../development/refactoring_roadmap.md) — Phase integration and refactor tracking
