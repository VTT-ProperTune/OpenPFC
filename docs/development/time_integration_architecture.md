<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Time integration architecture

This is the architecture reference for OpenPFC's time-integration design: how the pieces — `Simulator`, steppers, gradient evaluators, and physics models — fit together, who owns what memory, and which contracts you must respect when writing a new model or a new stepper. It complements [Custom stepper integration](../user_guide/custom_stepper_integration.md), which is a task-oriented "how do I migrate my model" guide; this document is the "why is it shaped this way" reference for model authors who need the full picture before making a design decision (e.g. choosing FD vs. spectral, single-field vs. multi-field, or Euler vs. RK4).

## Prerequisites

- Read [ADR 0003: Time integrator interface contracts](../adr/0003-time-integrator-interface.md) first — this document expands on its six contract areas with the currently-implemented stepper code (RK2/RK4 have since landed; the ADR text describing them as "not yet implemented" is superseded by [`explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp)).
- Read [Per-point grads aggregates](../extending_openpfc/per_point_grads.md) for the grads-aggregate contract that every model's `rhs()` depends on.
- Read [Custom stepper integration](../user_guide/custom_stepper_integration.md) for a step-by-step migration walkthrough.

## The four layers

OpenPFC's explicit time-integration stack is built from four independent layers, each with a narrow, swappable interface:

```
Model::rhs(t, g) -> increment        (physics, pure function of gradients)
        ↑
Eval::operator()(i,j,k) -> g         (spatial discretization: FD or spectral)
        ↑
for_each_interior(model, eval, du, t) (the one driver loop that connects them)
        ↑
Stepper::step(t, u)                   (time integration: Euler, RK2, RK4, ...)
        ↑
Simulator::step_with_physics(fn)      (prologue/epilogue: IC, BC, results I/O)
```

Each arrow is a real dependency (the layer above calls into the layer below), and each layer only knows about the layer immediately below it through a duck-typed interface — not a virtual base class. This is why swapping Euler for RK4, or FD for spectral, does not require touching model code: the model only ever sees `rhs(t, g)`.

### 1. Physics model: `rhs(double t, const G&) -> Inc`

The model is the only file most users write. It is a plain struct (no OpenPFC includes required) exposing:

```cpp
struct HeatModel {
  [[nodiscard]] double rhs(double /*t*/, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};
```

from [`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp). `G` is a model-owned aggregate (here `HeatGrads`) naming exactly the derivatives the physics needs, drawn from the fixed catalog `{value, x, y, z, xx, yy, zz, xy, xz, yz}` — see [Per-point grads aggregates](../extending_openpfc/per_point_grads.md) for the full contract. `rhs()` is `const noexcept`: the model has no time-stepping state and cannot mutate anything during evaluation.

`Inc`, the return type, is either:

- a `double` — a single-field model (`HeatModel` above), or
- a tuple-protocol bundle — a multi-field model (see [Multi-field composition](#multi-field-composition) below).

### 2. Gradient evaluator: `Eval::operator()(i,j,k) -> G`

The evaluator turns raw field storage into the `G` aggregate at a point. OpenPFC ships two evaluator families, both templated on the model's `G`:

- `pfc::gradient::FDGradient<G>` (finite difference) — [`include/openpfc/kernel/field/fd_gradient.hpp`](../../include/openpfc/kernel/field/fd_gradient.hpp). `pfc::field::FdGradient<G>` is a deprecated alias kept for source compatibility.
- `pfc::field::SpectralGradient<G>` (spectral / FFT) — [`include/openpfc/kernel/field/spectral_gradient.hpp`](../../include/openpfc/kernel/field/spectral_gradient.hpp).

Both are constructed via the `pfc::field::create<G>(...)` factory family, which mirrors `world::create` / `decomposition::create` / `fft::create`:

```cpp
// From a LocalField (unpadded storage + separate face-halo buffers)
auto grad = pfc::field::create<heat3d::HeatGrads>(local_field, /*order=*/2);

// From a PaddedBrick (owned core + in-place halo ring)
auto grad = pfc::field::create<heat3d::HeatGrads>(padded_brick, /*order=*/4);
```

Every evaluator exposes the same surface `for_each_interior` needs: `imin()/imax()`, `jmin()/jmax()`, `kmin()/kmax()`, `idx(ix,iy,iz) -> std::size_t`, `operator()(ix,iy,iz) -> G`, and `prepare()`. The **only** semantic difference between FD and spectral is what `prepare()` does and where halo exchange happens — see [Halo timing contract](#halo-timing-contract) below. `FDGradient<G>` rejects mixed second derivatives (`xy, xz, yz`) at compile time because they require corner-filled halos that the FD backend does not populate; use spectral for those.

### 3. Driver loop: `pfc::sim::for_each_interior`

[`include/openpfc/kernel/simulation/for_each_interior.hpp`](../../include/openpfc/kernel/simulation/for_each_interior.hpp) is the one loop every stepper's RHS lambda calls:

```cpp
template <class Model, class Eval, class DuOut>
inline void for_each_interior(const Model &model, Eval &eval, DuOut du, double t) {
  eval.prepare();
  for (int iz = eval.kmin(); iz < eval.kmax(); ++iz) {
    for (int iy = eval.jmin(); iy < eval.jmax(); ++iy) {
      for (int ix = eval.imin(); ix < eval.imax(); ++ix) {
        const auto g = eval(ix, iy, iz);
        const auto inc = model.rhs(t, g);
        detail::scatter(du, eval.idx(ix, iy, iz), inc);
      }
    }
  }
}
```

(elided of the `#pragma omp parallel for collapse(2)` around the `iz`/`iy` loops). `model` is `const`, taken by reference; `du` is written, never read. `detail::scatter` is what makes single-field and multi-field models transparent to this loop: if `du` is a raw `double*`, it writes one scalar; if it is a tuple of `double*`, it fans the tuple-protocol increments out element-by-element (see [Multi-field composition](#multi-field-composition)). `Model` and `Eval` are duck-typed template parameters — nothing here names `HeatModel`, `FDGradient`, or any concrete type.

### 4. Steppers: `Stepper::step(double t, std::vector<double>& u) -> double`

Steppers own **only** integration bookkeeping — the timestep `dt`, per-stage scratch buffers, and (for the factory-built steppers) the RHS lambda that closes over the model and evaluator by reference. They never own the field `u`; the caller passes `u` in and out by reference every call. This is contract area 3 ("workspace ownership") from ADR 0003: stage state lives in integrator allocations, not in Model fields.

[`include/openpfc/kernel/simulation/steppers/`](../../include/openpfc/kernel/simulation/steppers/) is the whole namespace `pfc::sim::steppers` lives in. It currently holds nine headers:

| Header | What it provides |
|---|---|
| `euler.hpp` | `EulerStepper`, `MultiEulerStepper`, and their `create(...)` factories |
| `explicit_rk.hpp` | `ExplicitRKStepper`, `MultiExplicitRKStepper`, and their `create(...)` factories |
| `butcher_tableau.hpp` | `ButcherTableau<T>` and the `make_rk2_midpoint`/`make_rk2_heun`/`make_rk4_classical`/`make_embedded_rk23` factories |
| `rk2_heun.hpp` | `RK2HeunStepper<Rhs>` — standalone RK2 Heun, no `create(...)` factory |
| `rk3_heun.hpp` | `RK3HeunStepper<Rhs>` — standalone RK3 Heun, no `create(...)` factory |
| `stage_workspace.hpp` | `StageWorkspace<T>` — a reusable per-stage scratch-buffer helper |
| `stepper_validation.hpp` | `validate_rhs_signature`, `validate_spatial_compatibility`, `validate_field_count`, `StepperValidationError` |
| `integrator_method.hpp` | `RKIntegratorMethod` enum, `make_tableau(...)`, JSON (de)serialization for config-driven method selection |
| `stepper_concept.hpp` | `SingleFieldStepper`/`MultiFieldStepper` C++20 concepts |

The four stepper classes documented in depth immediately below (`EulerStepper`, `MultiEulerStepper`, `ExplicitRKStepper`/`MultiExplicitRKStepper`, `ButcherTableau`) are the mainstream, factory-composable path. The remaining five headers are covered in their own subsections right after `ButcherTableau<T>`, below.

#### `EulerStepper<Rhs>` — forward Euler

From [`euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp):

```cpp
template <class Rhs> class EulerStepper {
public:
  EulerStepper(double dt, std::size_t local_size, Rhs rhs);
  double step(double t, std::vector<double> &u);   // u += dt * rhs(t, u)
  double dt() const noexcept;

  // Checkpoint/rollback protocol (for adaptive step-size control):
  void save_state(const std::vector<double>& u);
  void restore_state(std::vector<double>& u);
  [[nodiscard]] bool can_rollback() const noexcept;   // always true

private:
  double m_dt{0.0};
  std::vector<double> m_du;             // scratch, sized `local_size`
  std::vector<double> m_u_checkpoint;   // scratch, sized `local_size`
  Rhs m_rhs;
};
```

`step()` calls `m_rhs(t, u, m_du)` to fill `m_du`, then performs `u[i] += dt * m_du[i]` for every element in `u`. It does **not** perform halo exchange or any other backend pre-processing — that is the application's responsibility for FD (spectral has no halo to exchange; see below).

#### `MultiEulerStepper<Rhs, N>` — forward Euler over N fields

Also in `euler.hpp`. Same algorithm, but owns one `du` buffer *per field*:

```cpp
template <class Rhs, std::size_t N> class MultiEulerStepper {
public:
  using RhsType = Rhs;
  static constexpr std::size_t field_count = N;
  MultiEulerStepper(double dt, std::array<std::size_t, N> local_sizes, Rhs rhs);
  template <class... U> double step(double t, std::vector<U> &...u_buffers);
  // ... save_state / restore_state / can_rollback, same protocol as EulerStepper

private:
  std::array<std::vector<double>, N> m_du;             // one scratch buffer per field
  std::array<std::vector<double>, N> m_u_checkpoint;    // one checkpoint buffer per field
};
```

`Rhs` for the multi-field variant is invocable as `rhs(t, u_pack, du_pack)` where both packs are `std::tuple<std::vector<double>&, ...>` — see [Multi-field composition](#multi-field-composition).

#### `ExplicitRKStepper<Rhs>` / `MultiExplicitRKStepper<Rhs, N>` — explicit Runge-Kutta

From [`explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp). Same ownership model as the Euler steppers, but with one scratch buffer *per RK stage*:

```cpp
template <class Rhs> class ExplicitRKStepper {
public:
  ExplicitRKStepper(double dt, std::size_t local_size,
                    ButcherTableau<double> tableau, Rhs rhs);
  double step(double t, std::vector<double>& u);
  double dt() const noexcept;

private:
  double m_dt{0.0};
  std::vector<double> m_du;                     // scratch for the current stage
  std::vector<std::vector<double>> m_k;         // one buffer per stage, pre-allocated in the ctor
  ButcherTableau<double> m_tableau;
  Rhs m_rhs;
};
```

`step()` computes, for each stage `i` in `[0, stage_count())`: `u_temp = u + dt * sum_j(a_ij * k_j)`, then `k_i = rhs(t + c_i*dt, u_temp)`; after all stages, `u += dt * sum_i(b_i * k_i)`. All stage buffers (`m_k`) are allocated once in the constructor, so `step()` performs zero heap allocations for the fixed-size scratch (it does build one `u_temp` per stage on the stack of `step()` today — see [Known allocation cost](#known-allocation-cost)). `MultiExplicitRKStepper<Rhs, N>` is the analogous multi-field variant, with `m_k` indexed `[field][stage]`.

#### `ButcherTableau<T>` — validated RK coefficients

From [`butcher_tableau.hpp`](../../include/openpfc/kernel/simulation/steppers/butcher_tableau.hpp). An immutable, validated coefficient table:

```cpp
template <typename T> class ButcherTableau {
public:
  ButcherTableau(unsigned int s, std::vector<T> a_ij, std::vector<T> b_i,
                 std::vector<T> c_i, std::vector<T> b_hat_i = {},
                 std::string_view name = "", unsigned int order = 0,
                 unsigned int embedded_order = 0);   // throws TableauValidationError

  unsigned int stage_count() const noexcept;
  T a(unsigned int i, unsigned int j) const;
  T b(unsigned int i) const;
  T c(unsigned int i) const;
  T b_hat(unsigned int i) const;               // throws if no embedded weights
  bool has_embedded() const noexcept;
};
```

Construction validates explicit lower-triangular structure (`a_ij == 0` for `i <= j`), row-sum consistency (`sum_j a_ij == c_i`), and that every coefficient is finite; failures throw `TableauValidationError`. Four ready-made tableaus are provided as free functions in the same header:

| Factory | Method | Order |
|---|---|---|
| `make_rk2_midpoint<T>()` | RK2 midpoint | 2 |
| `make_rk2_heun<T>()` | RK2 Heun | 2 |
| `make_rk4_classical<T>()` | Classical RK4 | 4 |
| `make_embedded_rk23<T>()` | Bogacki-Shampine 3(2), embedded | 3 (embedded 2) |

`ButcherTableau<T>` requires `T` to be a real floating-point type (enforced with `static_assert`); complex-valued state uses a real-valued tableau applied element-wise.

**Status vs. ADR 0003**: the ADR's original text describes RK2/RK4 as infrastructure-only ("implementation pending"). That is no longer accurate — `ExplicitRKStepper` and `MultiExplicitRKStepper` are fully implemented and tested (see [`tests/unit/kernel/simulation/test_steppers.cpp`](../../tests/unit/kernel/simulation/test_steppers.cpp), including convergence-order tests that verify RK2 shows ~4x error reduction and RK4 shows ~16x error reduction when `dt` halves). IMEX methods remain design-phase only.

#### `RK2HeunStepper<Rhs>` / `RK3HeunStepper<Rhs>` — standalone point-wise Heun steppers

From [`rk2_heun.hpp`](../../include/openpfc/kernel/simulation/steppers/rk2_heun.hpp) and [`rk3_heun.hpp`](../../include/openpfc/kernel/simulation/steppers/rk3_heun.hpp). Both take the *same* low-level `Rhs` signature as `EulerStepper` — `rhs(double t, const std::vector<double>& u, std::vector<double>& du)` — and pre-allocate their own scratch buffers in the constructor (three buffers each: `RK2HeunStepper` keeps `m_du`/`m_predictor`/`m_rhs_predictor`; `RK3HeunStepper` keeps `m_k1`/`m_k2`/`m_u_temp`, reusing `m_k2` in place to hold `k3` once the old `k2` value is no longer needed).

- `RK2HeunStepper::step()`: predictor `u_p = u + dt * rhs(t, u)`, corrector `u += dt/2 * (rhs(t, u) + rhs(t + dt, u_p))` — reusing the predictor's `rhs(t, u)` evaluation so the corrector costs only one extra RHS call.
- `RK3HeunStepper::step()`: Heun's third-order method, `k1 = rhs(t, u)`, `k2 = rhs(t + dt/3, u + dt/3*k1)`, `k3 = rhs(t + 2dt/3, u + 2dt/3*k2)` (deliberately no `k1` term at this stage), then `u += dt/4*k1 + 3dt/4*k3` (`k2` does not appear in the final combination).

**Relationship to the generic `ExplicitRKStepper` path**: `make_rk2_heun<T>()`'s tableau (`a=[[0,0],[1,0]]`, `b=[0.5,0.5]`, `c=[0,1]`) is mathematically the same Heun's method `RK2HeunStepper` implements by hand — so RK2 Heun is available two ways: (1) `ExplicitRKStepper` + `make_rk2_heun()`, wired through the `create(...)` factories with `validate_rhs_signature`/`validate_spatial_compatibility` run automatically, or (2) `RK2HeunStepper`, a hand-specialized implementation with no `create(...)` overload and no validation — a model author using it must write their own RHS lambda (typically closing over `pfc::sim::for_each_interior(model, eval, du.data(), t)`, the same pattern the `create(...)` factories generate) and pass it straight to the constructor, as [`apps/heat3d/tests/test_heat3d.cpp`](../../apps/heat3d/tests/test_heat3d.cpp)'s `RK2HeunStepper<decltype(rhs)> stepper(dt, local_size, rhs)` and [`tests/integration/scenarios/time_integration/test_rk3_convergence.cpp`](../../tests/integration/scenarios/time_integration/test_rk3_convergence.cpp) both do. `RK3HeunStepper` has no `ButcherTableau`-driven equivalent in-tree today — `butcher_tableau.hpp` only ships RK2 midpoint, RK2 Heun, classical RK4, and the embedded Bogacki-Shampine 3(2) — so it is currently the only plain (non-embedded) third-order option.

#### `StageWorkspace<T>` — reusable multi-stage scratch buffer

From [`stage_workspace.hpp`](../../include/openpfc/kernel/simulation/steppers/stage_workspace.hpp). A small, standalone helper that owns `num_stages` buffers of `local_size` elements each (`std::vector<std::vector<T>>`), value-initialized to zero, with bounds-checked `stage(i)` access (throws `std::out_of_range`) and a `reset()` that zero-fills every buffer. Move-only (copy is deleted) to keep stage-buffer ownership unambiguous. It exists as a reusable building block for stepper authors writing new multi-stage integrators — `ExplicitRKStepper` does **not** use it today (it manages its own `std::vector<std::vector<double>> m_k` directly); treat `StageWorkspace<T>` as available infrastructure for a future or custom stepper rather than something the shipped RK stepper already depends on.

#### `stepper_validation.hpp` — the validation checks used by the Euler `create(...)` factories

From [`stepper_validation.hpp`](../../include/openpfc/kernel/simulation/steppers/stepper_validation.hpp), included by `euler.hpp`. This is where `validate_rhs_signature<Model, Eval>()` and `validate_spatial_compatibility<Eval>()` (referenced above in the `create(...)` factories section) are actually defined, alongside a third check, `validate_field_count<Stepper, Model, Eval>()`, which `static_assert`s that a multi-field stepper's `field_count` matches the arity of `Model::rhs`'s tuple-protocol return value. All validation failures — compile-time `static_assert`s for the first and third, a thrown `StepperValidationError` for the second — carry one of three `StepperValidationError::ErrorType` values (`SignatureMismatch`, `IncompatibleBackend`, `FieldCountMismatch`) for programmatic handling.

#### `integrator_method.hpp` — config-driven RK method selection

From [`integrator_method.hpp`](../../include/openpfc/kernel/simulation/steppers/integrator_method.hpp). Defines `pfc::sim::steppers::RKIntegratorMethod`, an enum with five values (`Euler`, `RK2_Midpoint`, `RK2_Heun`, `RK4_Classical`, `BogackiShampine32`) plus `to_string(...)`, `is_embedded(...)`, `validate_method(method, requires_adaptive)`, a `make_tableau(method) -> ButcherTableau<double>` factory, and a `pfc::ui::from_json<RKIntegratorMethod>` specialization for JSON-configured simulations (e.g. `{"method": "rk4_classical"}`). This is the mechanism an application would use to let users pick an RK method by name in a config file rather than hardcoding a tableau in source. Note two things this enum does **not** cover: it has no `RK3_Heun` value (`RK3HeunStepper` above is not reachable through this config path today), and it is deliberately a separate type from the unrelated `IntegratorMethod` enum in `time.hpp` (that one tracks `Time`'s own internal state machine, not RK method choice) — the header comment calls this out explicitly to prevent confusing the two.

#### `stepper_concept.hpp` — compile-time stepper interface concepts

From [`stepper_concept.hpp`](../../include/openpfc/kernel/simulation/steppers/stepper_concept.hpp). Two C++20 concepts formalize the duck-typed interface every stepper class above already implements, for use in `static_assert`s or constrained templates rather than as a new runtime type:

- `SingleFieldStepper<T>` — satisfied by any `T` with `step(double, std::vector<double>&) -> double` and `dt() -> double`. `EulerStepper`, `RK2HeunStepper`, and `ExplicitRKStepper` are all verified against it in [`tests/unit/kernel/simulation/steppers/test_stepper_concept.cpp`](../../tests/unit/kernel/simulation/steppers/test_stepper_concept.cpp).
- `MultiFieldStepper<T>` — the two-field `step(double, std::vector<double>&, std::vector<double>&) -> double` / `dt()` shape, plus a static `field_count` member that must be an integral `>= 1`.

### `pfc::sim::steppers::create(...)` factories

Most applications never construct a stepper class directly; they call the `create` free-function overloads, which bind an evaluator + model into the canonical `for_each_interior` RHS and return a fully-constructed stepper. All overloads capture `eval` and `model` **by reference** — they must outlive the returned stepper.

From `euler.hpp` (three overloads, single-field then two multi-field-size-deriving variants):

```cpp
namespace pfc::sim::steppers {

// Explicit local_size.
template <class Eval, class Model>
[[nodiscard]] auto create(Eval &eval, const Model &model, double dt,
                          std::size_t local_size);

// Derives local_size from a LocalField.
template <class T, class Eval, class Model>
[[nodiscard]] auto create(const pfc::field::LocalField<T> &u, Eval &eval,
                          const Model &model, double dt);

// Multi-field: tuple of LocalField refs + composite evaluator + model
// whose rhs() returns a tuple-protocol bundle.
template <class... Ts, class Eval, class Model>
[[nodiscard]] auto create(std::tuple<pfc::field::LocalField<Ts> &...> fields,
                          Eval &eval, const Model &model, double dt);

} // namespace pfc::sim::steppers
```

The single-field overloads run two validation checks before returning, both defined in [`stepper_validation.hpp`](../../include/openpfc/kernel/simulation/steppers/stepper_validation.hpp) (see the `stepper_validation.hpp` subsection above):

- `validate_rhs_signature<Model, Eval>()` — a compile-time `static_assert` that `model.rhs(double, const G&)` is callable with the `G` the evaluator returns, and that its return type is `double` or tuple-protocol compatible.
- `validate_spatial_compatibility<Eval>()` — a runtime check (throws `StepperValidationError`) that `Eval` is a recognized backend: `FDGradient<G>`, `SpectralGradient<G>`, or `CompositeGradient<...>`. A hand-rolled mock evaluator that does not match one of these three template instantiations will compile but throw at construction time — build steppers against real evaluators from `pfc::field::create<G>(...)` / `pfc::field::create_composite<...>(...)`, not ad hoc structs, if you go through this factory.

`explicit_rk.hpp` provides the matching three overloads with an extra `const ButcherTableau<double>& tableau` parameter, and does **not** run `validate_spatial_compatibility` (only the Euler factory does today).

## Canonical single-field example: heat3d + FD + Euler

This is the real pattern used by [`apps/heat3d/tests/test_heat3d.cpp`](../../apps/heat3d/tests/test_heat3d.cpp), built on `pfc::sim::stacks::FdCpuStack` (a bundle of `World + Decomposition + LocalField + face-halo buffers + SparseHaloExchanger`, see [`include/openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp`](../../include/openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp)):

```cpp
#include <heat3d/heat_model.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

const int N = 16;
const int order = 2;

pfc::sim::stacks::FdCpuStack stack(
    pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
    pfc::GridSpacing({1.0, 1.0, 1.0}), order, /*rank=*/0, /*nproc=*/1,
    MPI_COMM_WORLD);

heat3d::HeatModel model;
stack.u().apply(model.initial_condition);

auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), order);
auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, /*dt=*/1.0e-3);

for (int step = 0; step < 5; ++step) {
  stack.exchange_halos();                                    // FD: halo exchange before the step
  (void)stepper.step(static_cast<double>(step) * 1.0e-3, stack.u().vec());
}
```

Note the ordering: `stack.exchange_halos()` runs **before** `stepper.step(...)`, because `FDGradient::prepare()` (called inside `for_each_interior`, inside `stepper.step()`) is a no-op — the halo ring must already be populated when the stencil reads it. Swapping `EulerStepper` for `ExplicitRKStepper` here only changes the `create(...)` call (adding a `ButcherTableau<double>` argument) and requires calling `stack.exchange_halos()` once per **stage**, not once per step, if the model reads neighbor data at intermediate stage states — see [Halo timing contract](#halo-timing-contract).

## Multi-field composition

Multi-field models (e.g. the wave equation, `du/dt = v`, `dv/dt = c² ∇²u`) bundle several fields' gradients into one composite aggregate and several fields' increments into one tuple-protocol return value.

### The tuple protocol

[`include/openpfc/kernel/field/tuple_protocol.hpp`](../../include/openpfc/kernel/field/tuple_protocol.hpp) defines the opt-in convention `for_each_interior`'s `detail::scatter` uses to fan increments out to per-field buffers:

```cpp
namespace pfc::field::detail {
template <class T> concept has_as_tuple = requires(T &t) { t.as_tuple(); };
template <class T> concept is_tuple = /* std::tuple specialization */;

// Normalizes t: t.as_tuple() if opted in, t itself if std::tuple,
// otherwise std::forward_as_tuple(t) (scalar -> one-element view).
template <class T> constexpr decltype(auto) to_tuple(T &t);
}
```

A model's increments struct opts in with `as_tuple()`:

```cpp
// apps/wave2d/include/wave2d/wave_model.hpp
struct WaveIncrements {
  double du = 0.0;
  double dv = 0.0;
  auto as_tuple() { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};

struct WaveModel {
  double inv_dx2 = 1.0, inv_dy2 = 1.0;
  [[nodiscard]] WaveIncrements rhs(double /*t*/, double v_val,
                                   const WaveLaplacian &lap) const noexcept {
    const double lap_u = inv_dx2 * lap.lxx + inv_dy2 * lap.lyy;
    return WaveIncrements{v_val, kC * kC * lap_u};
  }
};
```

### Composite evaluator

`pfc::field::create_composite<Composite>(evals...)` from [`include/openpfc/kernel/field/composite_gradient.hpp`](../../include/openpfc/kernel/field/composite_gradient.hpp) bundles per-field evaluators (each a normal `FDGradient<G>`) into one `CompositeGradient<Composite, PerField...>` whose `operator()` returns a `Composite` aggregate — one member per field:

```cpp
struct UGrads { double xx{}; double yy{}; };  // u's Laplacian components
struct VGrads { double value{}; };             // v's current value
struct WaveLocal { UGrads u; VGrads v; };      // composite the model.rhs() reads

auto grad_u = pfc::field::create<UGrads>(u_field, fd_order);
auto grad_v = pfc::field::create<VGrads>(v_field, fd_order);
auto composite = pfc::field::create_composite<WaveLocal>(grad_u, grad_v);
```

`WaveLocal` does not itself need `as_tuple()` — the composite evaluator's return value is only ever read by `model.rhs(t, g)`, which accesses `g.u.xx` / `g.v.value` directly as struct members. Only the **increments** returned by `rhs()` need the tuple protocol, because `for_each_interior` is the one that scatters them into per-field `du` buffers.

Putting it together (real, tested pattern from [`tests/unit/kernel/simulation/test_wave_model_multifield_integration.cpp`](../../tests/unit/kernel/simulation/test_wave_model_multifield_integration.cpp)):

```cpp
// Adapter: WaveModel::rhs takes (v_val, WaveLaplacian) directly, so bridge
// the composite WaveLocal aggregate to that call shape.
struct WaveModelAdapter {
  wave2d::WaveModel model;
  [[nodiscard]] wave2d::WaveIncrements rhs(double t, const WaveLocal &g) const noexcept {
    wave2d::WaveLaplacian lap{.lxx = g.u.xx, .lyy = g.u.yy};
    return model.rhs(t, g.v.value, lap);
  }
};

WaveModelAdapter adapter{.model = wave2d::WaveModel{
    .inv_dx2 = 1.0 / (dx * dx), .inv_dy2 = 1.0 / (dy * dy)}};

auto stepper = pfc::sim::steppers::create(std::tie(u_field, v_field), composite, adapter, dt);

double t = 0.0;
for (int i = 0; i < n_steps; ++i) {
  stepper.step(t, u_field.vec(), v_field.vec());
  t += dt;
}
```

`pfc::sim::steppers::create(std::tuple<LocalField&...>, Eval&, Model&, dt)` derives each field's `local_size` from `LocalField::size()` and returns a `MultiEulerStepper<Rhs, 2>` here (2 = `sizeof...(Ts)`), with one `du` buffer per field, entirely independent of the other field's buffer.

## Halo timing contract

FD and spectral backends handle "refresh dependencies before this step's derivatives are valid" at different points, and getting this backwards produces stale-data bugs that are easy to miss in a serial run and only show up under MPI decomposition:

| Backend | Who exchanges halos | When | `Eval::prepare()` |
|---|---|---|---|
| FD (`FDGradient<G>`) | Application (`stack.exchange_halos()`, `pfc::communication::exchange(halo)`, or a `prepare_parent` hook) | Before `stepper.step(...)` / before `for_each_interior` runs | No-op (unless a `halo_prepare_callback` was supplied to the constructor for multi-stage coordination) |
| Spectral (`SpectralGradient<G>`) | Nothing external — the evaluator itself | Inside `eval.prepare()`, called once per `for_each_interior` invocation | Runs the forward FFT and all requested spectral derivative multiplies |

For **explicit RK steppers**, `for_each_interior` (and therefore `eval.prepare()`) runs once per **stage**, not once per step. For FD models this means the application-level halo refresh must also happen once per stage if the RHS at an intermediate stage reads neighbor-dependent gradients — `FDGradient`'s constructor accepts an optional `halo_prepare_callback` specifically so a multi-stage stepper can trigger that refresh via `eval.prepare()` instead of requiring the driver loop to know about stages. Spectral evaluators need no such wiring: `eval.prepare()` already re-runs the FFT every stage automatically.

## `Simulator` integration: `step_with_physics()`

[`include/openpfc/kernel/simulation/simulator.hpp`](../../include/openpfc/kernel/simulation/simulator.hpp) is the layer above the stepper. `Simulator::step_with_physics()` wraps a caller-supplied nullary lambda with the prologue/epilogue that every model — legacy or stepper-composed — needs:

```cpp
template <class PhysicsFn> void step_with_physics(PhysicsFn &&physics_fn) {
  begin_integrator_step();
  std::forward<PhysicsFn>(physics_fn)();
  end_integrator_step();
}
```

`begin_integrator_step()`:

1. On the **first call only** (`get_increment() == 0`): applies initial conditions, applies boundary conditions, writes results if `Time::do_save()`.
2. On **every call**: advances `Time::next()`, then applies boundary conditions at the new time.

`Time` also exposes `get_step_count()` as a self-documenting alias for `get_increment()` (same underlying counter) — prefer it in integrator code that wants to read "how many steps have been taken" rather than the save-scheduling-flavored name `get_increment()`.

Your `physics_fn` (typically `t = stepper.step(t, u);`) then runs with boundary conditions already valid at the new time. `end_integrator_step()` writes results if the new time is a save point.

```cpp
pfc::Time time({0.0, 1.0, dt}, /*saveat=*/0.1);
pfc::Simulator sim(model, time);
sim.add_initial_conditions(/* ... */);
pfc::initialize(sim);   // pfc::initialize(model, dt) via Simulator::initialize()

double t = 0.0;
while (!sim.done()) {
  sim.step_with_physics([&]() {
    t = stepper.step(t, u);
  });
}
```

This is exactly `Simulator::step()`'s body, generalized: `step()` is `begin_integrator_step(); pfc::step(get_model(), pfc::time::current(get_time())); end_integrator_step();` — i.e. it hardcodes the legacy `Model::step()` call where `step_with_physics()` lets you substitute any nullary lambda, including one that drives a `pfc::sim::steppers::create(...)`-built stepper.

## Legacy path: `Model::step(double t)`

Spectral apps predating the stepper-composition design still use the legacy pattern, where the model owns the entire time update:

```cpp
class Diffusion : public Model {
  void step(double t) override {
    fft.forward(psi, psi_F);
    for (auto &v : psi_F) v = opL[v_index] * v;   // implicit-Fourier operator
    fft.backward(psi_F, psi);
  }
};
```

`Simulator sim(model, time); while (!sim.done()) sim.step();` calls `model.step(t)` internally. This path remains fully supported — it is efficient for constant-coefficient-diffusion-style problems where an implicit-Fourier update is available (2 FFTs/step, unconditionally stable) — but it is not swappable: changing from Euler to RK4 means rewriting the model's `step()` body. Model authors should prefer `rhs(t, g)` + stepper composition unless they specifically need the implicit-Fourier update.

## Known allocation cost

`ExplicitRKStepper::step()` allocates one `std::vector<double> u_temp(u)` per stage inside the loop body (see [`explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp)) — this is a per-step, per-stage heap allocation that `EulerStepper::step()` does not have (its `m_du` scratch is reused across steps with no reallocation). For a 4-stage RK4 this is 4 allocations per step in addition to the pre-allocated `m_k` stage buffers. This is a known cost of the current implementation, not a documented API contract; if it becomes a bottleneck for your problem size, this is the file to profile first.

## Contract summary (from ADR 0003)

1. **Integrator surface**: `begin_integrator_step()` / `end_integrator_step()` ordering is fixed; custom steppers plug into the gap between them via `step_with_physics()`.
2. **State access**: legacy models mutate fields directly in `step()`; stepper-composed models expose a pure `rhs(t, g) -> increment` and never touch `u`.
3. **Workspace ownership**: `du` / stage scratch buffers live in the stepper (`EulerStepper::m_du`, `ExplicitRKStepper::m_k`, ...), never in the model. The `pfc::field::FieldAccessor<F>` C++20 concept ([`field_accessor_concept.hpp`](../../include/openpfc/kernel/field/field_accessor_concept.hpp)) formalizes the narrower storage-only half of this: any field type a stepper reads/writes must provide a const-and-non-const `size()`/`data()` pair.
4. **Multi-field composition**: the tuple protocol (`as_tuple()` / `std::tuple`) lets `for_each_interior` scatter a model's bundled increments into per-field buffers without the driver loop naming any field.
5. **Halo timing**: FD exchanges before `for_each_interior`; spectral exchanges (via FFT) inside `eval.prepare()`, called by `for_each_interior` itself.
6. **Migration**: extract `rhs(t, g)` from the old `step()` body, pick an evaluator, call `pfc::sim::steppers::create(...)`, replace `sim.step()` with `sim.step_with_physics([&]{ t = stepper.step(t, u); })`.

## See also

- [Custom stepper integration](../user_guide/custom_stepper_integration.md) — task-oriented migration guide with a complete worked example
- [ADR 0003: Time integrator interface contracts](../adr/0003-time-integrator-interface.md) — the formal decision record this document expands on
- [ADR 0002: Gradient operators, FD vs. spectral](../adr/0002-gradient-operators-fd-vs-spectral.md) — spatial operator directionality
- [Per-point grads aggregates](../extending_openpfc/per_point_grads.md) — the grads-aggregate contract `rhs()` and the evaluators share
- [Halo exchange](../concepts/halo_exchange.md) — halo policies and exchange patterns referenced by the FD timing contract above
- [`include/openpfc/kernel/simulation/du_field.hpp`](../../include/openpfc/kernel/simulation/du_field.hpp) — the stack-friendly single-field alternative to the stepper classes, for compact `du.apply(...); u += dt * du;` loops
- [`docs/development/refactoring_roadmap.md`](refactoring_roadmap.md) — tracks remaining stepper/integrator work (IMEX, adaptive step-size control)
