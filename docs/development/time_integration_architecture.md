<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Time integration architecture

This guide explains OpenPFC's time-integration architecture for **model authors**:
how legacy `Model::step` physics differs from the replaceable
`rhs(t, g)` + stepper composition path, which factory signatures to call, who
owns scratch memory, and how to migrate. It bridges the architectural contracts
in [ADR 0003](../adr/0003-time-integrator-interface.md) with the grads-aggregate
details in [Per-point grads aggregates](../extending_openpfc/per_point_grads.md).

For a task-oriented walkthrough with a full worked example, see
[Custom stepper integration](../user_guide/custom_stepper_integration.md).
This document is the architecture reference those steps assume.

## 1. Two RHS evaluation patterns

### Legacy pattern

Spectral (and older) apps still implement time advance inside a virtual
`Model::step` override. The pure virtual declaration lives in
[`include/openpfc/kernel/simulation/model.hpp`](../../include/openpfc/kernel/simulation/model.hpp):

```cpp
class Model {
public:
  virtual void step(double t) = 0;
  virtual void initialize(double dt) = 0;
  // ...
};
```

A typical override owns the whole update (FFT, operators, field mutation):

```cpp
class Diffusion : public Model {
  void step(double t) override {
    fft.forward(psi, psi_F);
    for (auto &v : psi_F) v = opL[v_index] * v;   // implicit-Fourier operator
    fft.backward(psi_F, psi);
  }
};
```

`Simulator::step()` calls `model.step(t)` between `begin_integrator_step()` and
`end_integrator_step()`. This path remains supported and is efficient for
constant-coefficient implicit-Fourier updates, but it is **not** swappable:
changing Euler to RK4 means rewriting the model's `step()` body.

### Gradient-access pattern

Modern FD/spectral point-wise models expose a pure `rhs` that reads a grads
aggregate and returns an increment. The model does not own steppers or scratch
buffers. From
[`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp):

```cpp
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

struct HeatModel {
  [[nodiscard]] double rhs(double /*t*/, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};
```

The same idea scales to multi-field physics. From
[`apps/wave2d/include/wave2d/wave_model.hpp`](../../apps/wave2d/include/wave2d/wave_model.hpp):

```cpp
struct WaveIncrements {
  double du = 0.0;
  double dv = 0.0;
  auto as_tuple() { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};

struct WaveModel {
  double inv_dx2 = 1.0;
  double inv_dy2 = 1.0;

  [[nodiscard]] WaveIncrements rhs(double /*t*/, double v_val,
                                   const WaveLaplacian &lap) const noexcept {
    const double lap_u = inv_dx2 * lap.lxx + inv_dy2 * lap.lyy;
    return WaveIncrements{v_val, kC * kC * lap_u};
  }
};
```

A gradient evaluator (`pfc::gradient::FDGradient<G>` or spectral) turns field
storage into `G` at each interior point; `pfc::sim::for_each_interior` calls
`model.rhs(t, g)` and scatters increments into stepper-owned `du` buffers. The
model only ever sees `rhs` — never the integrator algorithm.

## 2. Writing physics with rhs callables

Contract for a point-wise model used with the stepper factories:

1. **Grads aggregate** — name only the derivatives you need from the catalog
   `{value, x, y, z, xx, yy, zz, xy, xz, yz}` (see
   [per_point_grads.md](../extending_openpfc/per_point_grads.md)).
2. **`rhs` is `const noexcept`** — no mutable time-stepping state; do not write
   field storage during evaluation.
3. **Return type** — `double` for single-field models (`HeatModel`), or a
   tuple-protocol bundle (`as_tuple()` / `std::tuple`) for multi-field models
   (`WaveIncrements`).
4. **Evaluator match** — the `G` (or composite aggregate) passed to `rhs` must
   be what your evaluator's `operator()(ix,iy,iz)` returns.

`HeatModel::rhs(t, g)` is the single-field reference. `WaveModel::rhs` takes
`(t, v_val, lap)` directly; apps that compose per-field evaluators usually
adapt it to a single composite aggregate before calling
`pfc::sim::steppers::create` (see
[`tests/unit/kernel/simulation/test_wave_model_multifield_integration.cpp`](../../tests/unit/kernel/simulation/test_wave_model_multifield_integration.cpp)).

Build the evaluator with the `pfc::field::create<G>(...)` family over
[`include/openpfc/kernel/field/fd_gradient.hpp`](../../include/openpfc/kernel/field/fd_gradient.hpp):

```cpp
namespace pfc::gradient {
template <class G> class FDGradient {
public:
  explicit FDGradient(const pfc::field::PaddedBrick<double> &u, int order = 2,
                      std::function<void()> halo_prepare_callback = {});
  void prepare();
  [[nodiscard]] G operator()(int ix, int iy, int iz) const noexcept;
  // imin/imax, jmin/jmax, kmin/kmax, idx(...)
};
} // namespace pfc::gradient

// Preferred construction (also available as pfc::field::create<G>(...)):
auto grad = pfc::field::create<heat3d::HeatGrads>(local_field, /*order=*/2);
```

`FDGradient<G>` rejects mixed second derivatives (`xy`, `xz`, `yz`) at compile
time; use spectral for those. For FD, the application must exchange halos
**before** `stepper.step(...)` because `prepare()` is a no-op unless a
`halo_prepare_callback` was supplied.

## 3. Using stepper factories

Factories live under
[`include/openpfc/kernel/simulation/steppers/`](../../include/openpfc/kernel/simulation/steppers/)
(there is no single `include/openpfc/kernel/stepper.hpp`). Most apps never
construct stepper classes directly.

From [`euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)
(`namespace pfc::sim::steppers`):

```cpp
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
```

The single-field overloads run `validate_rhs_signature<Model, Eval>()` and
`validate_spatial_compatibility<Eval>()` (defined in
[`stepper_validation.hpp`](../../include/openpfc/kernel/simulation/steppers/stepper_validation.hpp))
before returning. `eval` and `model` are captured **by reference** and must
outlive the stepper.

Matching RK factories in
[`explicit_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/explicit_rk.hpp)
take an extra `const ButcherTableau<double>& tableau`. Ready-made tableaus from
[`butcher_tableau.hpp`](../../include/openpfc/kernel/simulation/steppers/butcher_tableau.hpp):

| Factory | Method |
|---|---|
| `make_rk2_midpoint<T>()` | RK2 midpoint |
| `make_rk2_heun<T>()` | RK2 Heun |
| `make_rk4_classical<T>()` | Classical RK4 |
| `make_embedded_rk23<T>()` | Bogacki–Shampine 3(2) |
| `make_embedded_rk45<T>()` | Dormand–Prince 5(4) |

Standalone Heun steppers (`RK2HeunStepper`, `RK3HeunStepper` in
`rk2_heun.hpp` / `rk3_heun.hpp`) have **no** `create(...)` overload — construct
them with an explicit low-level RHS lambda if needed.

Config-driven selection:
[`integrator_method.hpp`](../../include/openpfc/kernel/simulation/steppers/integrator_method.hpp)
provides `RKIntegratorMethod` and `make_tableau(method)`. Those helpers
remain **identity / tableau** tools (`to_string`, `validate_method`,
`make_tableau`) — they do not construct a stepper.

The **composition boundary** is
[`method_composition.hpp`](../../include/openpfc/kernel/simulation/steppers/method_composition.hpp):
`compose_scalar` / `compose_multi` map a stable method id plus
`IntegratorComposeConfig` through the `register_method_composer` table to a
validated `IntegratorComposition` (Euler today, via the same
`EulerStepper` / `MultiEulerStepper` types as typed `create`). Prefer that
API when the driver must not switch on method-specific construction;
prefer typed `create` when the method type is fixed at the call site. See
[Custom stepper integration](../user_guide/custom_stepper_integration.md).

## 4. Workspace ownership contract

Integrators own their scratch; model fields are **read-only** during RHS
evaluation. From `EulerStepper` in `euler.hpp`:

```cpp
template <class Rhs> class EulerStepper {
public:
  EulerStepper(double dt, std::size_t local_size, Rhs rhs);
  double step(double t, std::vector<double> &u);   // u += dt * rhs(t, u)
  double dt() const noexcept;
  void save_state(const std::vector<double>& u);
  void restore_state(std::vector<double>& u);
  [[nodiscard]] bool can_rollback() const noexcept;

private:
  double m_dt{0.0};
  std::vector<double> m_du;             // scratch, sized local_size
  std::vector<double> m_u_checkpoint;   // scratch, sized local_size
  Rhs m_rhs;
};
```

`MultiEulerStepper<Rhs, N>` owns `std::array<std::vector<double>, N> m_du` and
`m_u_checkpoint` — one buffer per field.
`ExplicitRKStepper` additionally owns `std::vector<std::vector<double>> m_k`
(one buffer per RK stage). The caller passes field storage `u` by reference;
the stepper never owns it.

This is ADR 0003 contract area 3 (workspace ownership): stage state lives in
integrator allocations, not in `Model` fields. Reusable infrastructure for
custom multi-stage steppers:
[`StageWorkspace<T>`](../../include/openpfc/kernel/simulation/steppers/stage_workspace.hpp)
(`std::vector<std::vector<T>>`, move-only). The shipped `ExplicitRKStepper`
manages `m_k` itself today.

### Spectral exponential-action coefficients (CPU)

Diagonal integrating-factor / ETD1-style updates need per-mode coefficients
`exp(L*dt)` and `(exp(L*dt)-1)/L` from already-formed spectral samples `L`.
That construction lives in
[`spectral_exp_coefficients.hpp`](../../include/openpfc/kernel/integrator/spectral_exp_coefficients.hpp)
(`pfc::integrator`), not in app models:

- `spectral_exp_coeffs(L, dt)` — scalar evaluation; near `|L| < 1e-12` uses
  Taylor `dt + 0.5*L*dt*dt` for `phi1_L`, otherwise `expm1(L*dt)/L`.
- **Caller-owned:** `fill_spectral_exp_coeffs` writes into caller `std::span`s.
- **Method-owned:** `SpectralExpCoefficientCache` owns `std::vector` storage;
  `exp_Ldt()` / `phi1_L()` views remain valid until a resizing `ensure` or
  destruction/move. Rebuild when `SpectralExpOperatorId`, `SpectralExpDtId`,
  or `SpectralExpConfigId` (or mode count) changes.

Coefficients are transient/recomputable and are **not** checkpointed.
Tungsten no longer builds physics-specific `opN = expm1(arg)/opCk` inside
`Model` members. The app maps `L = k_laplacian * opCk` and owns method weights
in [`TungstenEtdWorkspace`](../../apps/tungsten/include/tungsten/common/tungsten_etd_workspace.hpp)
(`n_weight = k_laplacian * phi1_L`) over `SpectralExpCoefficientCache`.
`Model::step` delegates the exponential combine to that workspace (CPU) or to
`tungsten::ops::apply_time_integration` with workspace device buffers
(CUDA/HIP). Temporary adapter removal:
`TODO(remove-tungsten-etd-workspace): replace with Etd1Stepper after #169`.
Coverage:
[`test_spectral_exp_coefficients.cpp`](../../tests/unit/kernel/integrator/test_spectral_exp_coefficients.cpp)
(`[integrator][spectral_exp]`) and Tungsten
[`test_tungsten.cpp`](../../apps/tungsten/tests/test_tungsten.cpp)
(`[tungsten][spectral]`).

### ETD1 step attempts (`Etd1Stepper`)

CPU first-order exponential time-differencing lives in
[`etd1.hpp`](../../include/openpfc/kernel/simulation/steppers/etd1.hpp)
(`pfc::sim::steppers`). It consumes injectable diagonal coefficient spans
(`exp_Ldt`, `phi1_L` from the builder above or test-built views) and a
`StageFunction`-compatible nonlinear `N`:

```cpp
Etd1Stepper stepper(dt, local_size, rhs);
stepper.set_coefficients(exp_Ldt, phi1_L);  // or SpectralExpCoefficientCache
auto attempt = stepper.attempt_step(t, u_accepted);  // u_accepted never written
// candidate = exp_Ldt * u + phi1_L * N   (phi1_L already includes dt)
```

`attempt_step` copies accepted state into method-owned scratch before
evaluating `N`, writes an isolated `candidate()`, and returns
`Etd1StepAttempt` (`success` / `t_next`) — computational completion only,
not adaptive accept/reject. `MultiEtd1Stepper<Rhs, 2>` covers a two-field
pack with the same isolation per field. Transient coeff/scratch caches are
not checkpointable. Tungsten still uses the temporary
`TungstenEtdWorkspace` adapter until App/Simulator wires driver-owned
`Etd1Stepper` time advance (`TODO(remove-tungsten-etd-workspace)`). Coverage:
`tests/unit/kernel/simulation/steppers/test_etd1.cpp` (`[stepper][etd1]`).

### Embedded RK step attempts (`EmbeddedRKStepper`)

For adaptive controllers that need local truncation-error evidence without
mutating the accepted state, use
[`embedded_rk.hpp`](../../include/openpfc/kernel/simulation/steppers/embedded_rk.hpp):

```cpp
EmbeddedRKStepper stepper(local_size, make_embedded_rk45<double>(), rhs);
auto result = stepper.attempt(t, dt, u);  // u is const — never written
// result.u_high / result.u_low / result.error  (views into method-owned buffers)
// result.success == computational completion only (not accept/reject)
```

`attempt` evaluates shared stages once (`rhs_evals == tableau.stage_count()`),
accumulates with primary `b` into isolated `u_high` and with embedded `b_hat`
into `u_low`, and forms `error = u_high - u_low`. Accept/reject and next-`dt`
selection stay driver/controller-owned. FSAL stage reuse is deferred; any
future cache must invalidate on reject, restart, or configuration change.

### Driver-owned time attempt transactions (`Time`)

Adaptive control needs an accepted-time clock that does not move on reject,
plus a clipped attempt interval that cannot overshoot `t1` or skip past the
next `saveat` alignment point. That seam lives on
[`time.hpp`](../../include/openpfc/kernel/simulation/time.hpp):

- `get_accepted_time()` — read-only accepted clock; unchanged while an attempt
  is active.
- `clip_attempt_dt(candidate_dt)` — returns an attempted interval so
  `accepted + attempted <= t1`, and when `saveat > 0` lands on the next output
  alignment (most-restrictive `min` when both constraints apply; `saveat <= 0`
  skips alignment).
- `begin_attempt` / `commit_attempt` / `reject_attempt` — store the clipped
  interval, advance accepted time by exactly that interval on commit, or leave
  accepted time unchanged on reject. `do_save()` continues to key off accepted
  time only (rejected attempts never advance emission).

Prefer this transaction API over rewriting history with `set_dt` +
`set_increment`. When Simulator wires adaptive loops, every MPI rank must use
the **same** clipped `attempted_dt` for a given attempt (rank-consistency is
a driver obligation; this leaf does not orchestrate Simulator).

## 5. Future stepper integration

New steppers should follow the duck-typed surface already formalized in
[`stepper_concept.hpp`](../../include/openpfc/kernel/simulation/steppers/stepper_concept.hpp):

- `SingleFieldStepper<T>` — `step(double, std::vector<double>&) -> double` and
  `dt() -> double` (satisfied by `EulerStepper`, `RK2HeunStepper`,
  `ExplicitRKStepper`; see
  [`tests/unit/kernel/simulation/steppers/test_stepper_concept.cpp`](../../tests/unit/kernel/simulation/steppers/test_stepper_concept.cpp)).
- `MultiFieldStepper<T>` — multi-buffer `step` plus static `field_count`.

Recommended pattern:

1. Own all scratch (`du`, stage `k` buffers, optional checkpoint) inside the
   stepper class or via `StageWorkspace<T>`.
2. Accept an `Rhs` callable with the same low-level signature the factories
   generate: `rhs(t, u, du)` (single-field) or `rhs(t, u_pack, du_pack)`
   (multi-field).
3. Prefer a `create(Eval&, Model&, ...)` free function that closes over
   `pfc::sim::for_each_interior(model, eval, du, t)` so model authors keep
   writing only `rhs(t, g)`.
4. Plug into
   [`Simulator::step_with_physics`](../../include/openpfc/kernel/simulation/simulator.hpp)
   between `begin_integrator_step()` / `end_integrator_step()`.

Embedded pair evidence already exists under `steppers/`
(`EmbeddedRKStepper::attempt`, `make_embedded_rk45` / `make_embedded_rk23`).
<<<<<<< HEAD
The shared IMEX stage-composition seam is also landed
(`ImexEulerComposer` / `ImexStepAttemptResult` in
[`imex_stage_composition.hpp`](../../include/openpfc/kernel/simulation/steppers/imex_stage_composition.hpp)):
explicit operator evaluation then an implicit `SolveFunction` solve into an
isolated candidate, with driver commit via `apply_candidate`. First-order
IMEX Euler (`ImexEulerStepper` in
[`imex_euler.hpp`](../../include/openpfc/kernel/simulation/steppers/imex_euler.hpp))
is landed on CPU; higher-order IMEX-RK and adaptive *controller* policy
(accept/reject and next-`dt` selection) remain follow-on / driver-owned work
tracked in [`refactoring_roadmap.md`](refactoring_roadmap.md). The checkpoint
protocol on `EulerStepper` (`save_state` / `restore_state` / `can_rollback`)
is the hook those controller features are expected to use.
=======
Adaptive-control **policy configuration** (tolerances, growth/shrink limits,
min/max `dt`, rejection cap, fixed-vs-adaptive mode) lives in
[`adaptive_control_config.hpp`](../../include/openpfc/kernel/simulation/adaptive_control_config.hpp).
IMEX schemes and adaptive *controller* policy (accept/reject and next-`dt`
selection that *uses* that config) remain future/driver-owned work tracked in
[`refactoring_roadmap.md`](refactoring_roadmap.md). The checkpoint protocol on
`EulerStepper` (`save_state` / `restore_state` / `can_rollback`) is the hook
those controller features are expected to use.
>>>>>>> b263f973 (docs: document AdaptiveControlConfig in tour and architecture)

## 6. Migration path from virtual step methods to explicit stepper composition

These five steps reflect what actually exists in-tree today (heat3d / wave2d
apps and unit tests), not a hypothetical API:

1. **Extract point-wise physics** from the old `step()` body into an `rhs`
   callable with a grads parameter (or adapt multi-arg forms like
   `WaveModel::rhs` behind a thin adapter). Keep the model free of OpenPFC
   includes when possible (`HeatModel` is header-only physics).
2. **Create a gradient evaluator** with `pfc::field::create<G>(...)` from
   [`fd_gradient.hpp`](../../include/openpfc/kernel/field/fd_gradient.hpp)
   (or spectral / `create_composite` for multi-field).
3. **Replace hand-rolled advance** with
   `pfc::sim::steppers::create(...)` from `euler.hpp` or `explicit_rk.hpp`
   (pass a `ButcherTableau` for RK).
4. **Drive the stepper** from `Simulator::step_with_physics`: call
   `stack.exchange_halos()` (FD) then `stepper.step(t, u)` inside the lambda
   instead of `sim.step()` / `model.step(t)`.
5. **Remove the legacy `step()` override** once the app no longer needs the
   implicit-Fourier path; keep `Model::step` only when that spectral update is
   intentional.

Before/after references in the repository:

- After (single-field FD + Euler):
  [`apps/heat3d/tests/test_heat3d.cpp`](../../apps/heat3d/tests/test_heat3d.cpp)
- After (multi-field):
  [`tests/unit/kernel/simulation/test_wave_model_multifield_integration.cpp`](../../tests/unit/kernel/simulation/test_wave_model_multifield_integration.cpp)
- Legacy `Model::step` surface:
  [`include/openpfc/kernel/simulation/model.hpp`](../../include/openpfc/kernel/simulation/model.hpp)
- Narrative before/after in
  [Custom stepper integration](../user_guide/custom_stepper_integration.md)

## 7. Code examples

### (a) RHS callable definition

Source:
[`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp)

```cpp
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

struct HeatModel {
  [[nodiscard]] double rhs(double /*t*/, const HeatGrads &g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};
```

### (b) Stepper factory invocation

Source pattern from
[`apps/heat3d/tests/test_heat3d.cpp`](../../apps/heat3d/tests/test_heat3d.cpp);
factory signatures from
[`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)

```cpp
#include <heat3d/heat_model.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

heat3d::HeatModel model;
auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), /*order=*/2);
auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, /*dt=*/1.0e-3);
```

### (c) Full stepper usage in a simulation loop

Source pattern from the same heat3d test (with
[`FdCpuStack`](../../include/openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp)):

```cpp
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

pfc::sim::stacks::FdCpuStack stack(
    pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
    pfc::GridSpacing({1.0, 1.0, 1.0}), order, /*rank=*/0, /*nproc=*/1,
    MPI_COMM_WORLD);

heat3d::HeatModel model;
stack.u().apply(model.initial_condition);

auto grad = pfc::field::create<heat3d::HeatGrads>(stack.u(), order);
auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, /*dt=*/1.0e-3);

for (int step = 0; step < 5; ++step) {
  stack.exchange_halos();   // FD: halo exchange before the step
  (void)stepper.step(static_cast<double>(step) * 1.0e-3, stack.u().vec());
}
```

With a `Simulator`, wrap the advance in `step_with_physics` instead of calling
`sim.step()` (which hardcodes legacy `Model::step`).

## 8. Connections to existing documentation

| Document | Path | Role |
|---|---|---|
| ADR 0003 — Time integrator interface | [`docs/adr/0003-time-integrator-interface.md`](../adr/0003-time-integrator-interface.md) | Formal contracts (integrator surface, workspace ownership, halo timing, migration) |
| Per-point grads aggregates | [`docs/extending_openpfc/per_point_grads.md`](../extending_openpfc/per_point_grads.md) | Grads-aggregate catalog and evaluator contract that every `rhs()` depends on |
| Custom stepper integration | [`docs/user_guide/custom_stepper_integration.md`](../user_guide/custom_stepper_integration.md) | Task-oriented migration guide with a complete worked example |
| ADR 0002 — Gradient operators | [`docs/adr/0002-gradient-operators-fd-vs-spectral.md`](../adr/0002-gradient-operators-fd-vs-spectral.md) | FD vs spectral spatial operators |
| Halo exchange | [`docs/concepts/halo_exchange.md`](../concepts/halo_exchange.md) | Halo policies referenced by the FD timing contract |
| Refactoring roadmap | [`docs/development/refactoring_roadmap.md`](refactoring_roadmap.md) | Remaining stepper/integrator work (IMEX, adaptive control) |

Primary headers cited in this guide:

- [`include/openpfc/kernel/simulation/model.hpp`](../../include/openpfc/kernel/simulation/model.hpp)
- [`include/openpfc/kernel/simulation/steppers/`](../../include/openpfc/kernel/simulation/steppers/)
- [`include/openpfc/kernel/field/fd_gradient.hpp`](../../include/openpfc/kernel/field/fd_gradient.hpp)
- [`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp)
- [`apps/wave2d/include/wave2d/wave_model.hpp`](../../apps/wave2d/include/wave2d/wave_model.hpp)
