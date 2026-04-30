<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Per-point grads aggregates

OpenPFC's point-wise driver loop, [`pfc::sim::for_each_interior`](../../include/openpfc/kernel/simulation/for_each_interior.hpp), is the canonical way to evaluate a model's right-hand side cell by cell:

```
du[{i,j,k}] = model.rhs(t, eval(i,j,k))
```

The `eval(i,j,k)` call returns a **model-owned aggregate** that names exactly the partial derivatives the model needs, drawn from a fixed catalog. The kernel's per-point evaluators (`pfc::field::FdGradient<G>`, `pfc::field::SpectralGradient<G>`) are templated on that aggregate `G` and use C++20 concepts to fill **only the members `G` declares** — no wasted FFTs, no wasted stencil sweeps.

This page is the contract reference: which catalog members the kernel knows about, how a model declares its needs, what each backend can and cannot supply, and how to bundle multiple fields into one composite per-point view.

## The catalog

A grads aggregate is any C++ aggregate (`struct`) whose members are named from this fixed catalog:

| Member | Meaning | Detected by |
|--------|---------|--------------|
| `value` | The scalar field value at the point | `pfc::field::has_value<G>` |
| `x`, `y`, `z` | First partial derivatives | `pfc::field::has_x<G>`, ... |
| `xx`, `yy`, `zz` | Unmixed second partials | `pfc::field::has_xx<G>`, ... |
| `xy`, `xz`, `yz` | Mixed second partials | `pfc::field::has_xy<G>`, ... |

All members are `double`. Use C++20 designated initializers to construct in tests:

```cpp
struct HeatGrads { double xx{}, yy{}, zz{}; };
HeatGrads g{.xx = 1.0, .yy = -2.0, .zz = 0.5};
```

The default-zero initializers matter for any future evaluator that doesn't fill every requested member; nothing in the kernel today relies on that, but it keeps the contract robust.

A convenience default catalog struct, [`pfc::field::GradPoint`](../../include/openpfc/kernel/field/grad_point.hpp), enumerates every member. It is fine to use when minimizing kernel work isn't critical, but for the cheapest possible loop **declare a model-owned aggregate that names only what you read.** The introspection layer is per-member, so a smaller struct is genuinely cheaper.

## Single-field example: heat equation

The `apps/heat3d/` model is the canonical small example. The grads aggregate ([`apps/heat3d/include/heat3d/heat_grads.hpp`](../../apps/heat3d/include/heat3d/heat_grads.hpp)) names exactly three slots:

```cpp
namespace heat3d {
struct HeatGrads { double xx{}, yy{}, zz{}; };
}
```

The model ([`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp)) is **OpenPFC-free** — only `<cmath>` and `<functional>` are needed, plus the local `heat_grads.hpp`:

```cpp
struct HeatModel {
  double D = 1.0;
  inline double rhs(double, const HeatGrads& g) const noexcept {
    return D * (g.xx + g.yy + g.zz);
  }
};
```

The driver wires it to a backend with two lines:

```cpp
auto grad    = pfc::field::create<heat3d::HeatGrads>(stack.u(), order);
auto stepper = pfc::sim::steppers::create(stack.u(), grad, model, dt);
```

The `field::create<G>(...)` factory takes the grads type as an explicit template argument. The stepper deduces everything else from the field bundle.

## Backend capability matrix

Different backends can fulfill different subsets of the catalog. Asking for a member a backend cannot supply is a **compile-time error** (`static_assert`), not a silent zero.

| Backend | `value` | `x/y/z` | `xx/yy/zz` | `xy/xz/yz` |
|---------|---------|---------|------------|------------|
| `pfc::field::FdGradient<G>` | yes | not yet | yes | not yet (needs corner halos) |
| `pfc::field::SpectralGradient<G>` | yes | yes (via `i k_i`) | yes (via `-k_i^2`) | yes (via `-k_i k_j`) |

When new requirements show up:
- FD first derivatives need a 1st-order central stencil table and the same halo width as the existing 2nd-order pass.
- FD mixed seconds need corner halos in the exchanger; until then the spectral path is the only option.

## Multi-field models

For wave-equation-style problems with several coupled fields the model declares per-field grads structs and a composite aggregate, then OpenPFC's [`pfc::field::CompositeGradient`](../../include/openpfc/kernel/field/composite_gradient.hpp) bundles per-field evaluators into a single `eval(i,j,k) -> Composite` callable.

Sketch:

```cpp
namespace wave {

struct UGrads { double xx{}, yy{}, zz{}; };  // u needs Laplacian
struct VGrads { double value{}; };            // v needs only its value

struct WaveLocal {     // composite per-point view (model-owned)
  UGrads u;
  VGrads v;
};

struct WaveIncrements {                       // model-owned dU
  double du;
  double dv;
  // Opt in to the tuple protocol for scatter-into-buffers:
  auto as_tuple()       { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};

struct WaveModel {
  double c2 = 100.0;
  WaveIncrements rhs(double, const WaveLocal& g) const noexcept {
    return { .du = g.v.value,
             .dv = c2 * (g.u.xx + g.u.yy + g.u.zz) };
  }
};

} // namespace wave
```

The wiring then looks like:

```cpp
auto grad_u = pfc::field::create<wave::UGrads>(stack.u(), order);
auto grad_v = pfc::field::create<wave::VGrads>(stack.v(), order);

auto composite = pfc::field::create_composite<wave::WaveLocal>(grad_u, grad_v);

auto stepper = pfc::sim::steppers::create(
    std::tie(stack.u(), stack.v()),  // tuple of LocalField references
    composite, model, dt);
```

`for_each_interior` deduces that the model returns a multi-field bundle, normalizes both `du` (a `std::tuple<double*, double*>`) and `inc` (via `WaveIncrements::as_tuple()`) through the **tuple protocol** in [`tuple_protocol.hpp`](../../include/openpfc/kernel/field/tuple_protocol.hpp), and scatters element by element. The scatter is `static_assert`-checked for arity match, so missing or extra fields surface as a clean diagnostic.

## The tuple protocol — how multi-field bundles fan out without Boost

OpenPFC does not depend on `boost::pfr` or any other external introspection library. To opt a model-owned aggregate into the multi-field path, add **one line per type**:

```cpp
struct WaveIncrements {
  double du, dv;
  auto as_tuple()       { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};
```

Alternatively, just use `std::tuple<double, double>` directly (loses the `inc.du` / `inc.dv` ergonomics; valid).

Detection is encoded in two trait/concepts in `pfc::field::detail` — see the source if you need to extend the protocol (e.g. to accept `std::array`).

## Storage stays SoA; per-point views are AoS-on-demand

Each field is a separate contiguous `LocalField<double>` (so FFTs and halo exchange stay cheap). The composite aggregate is **only built per point**, lives in registers across the `model.rhs` call, and never lands in memory. There is no allocation cost for per-point AoS bundling — the optimizer collapses it.

If you find yourself wanting interleaved (`u1, v1, w1, u2, v2, w2, ...`) storage, the cost calculus probably no longer holds: you would give up FFT-friendliness for a small per-point cache benefit. Stay with SoA at storage and let the per-point view be transient.

## Where this lives

- Concepts: [`include/openpfc/kernel/field/grad_concepts.hpp`](../../include/openpfc/kernel/field/grad_concepts.hpp)
- Default catalog struct: [`include/openpfc/kernel/field/grad_point.hpp`](../../include/openpfc/kernel/field/grad_point.hpp)
- FD evaluator: [`include/openpfc/kernel/field/fd_gradient.hpp`](../../include/openpfc/kernel/field/fd_gradient.hpp)
- Spectral evaluator: [`include/openpfc/kernel/field/spectral_gradient.hpp`](../../include/openpfc/kernel/field/spectral_gradient.hpp)
- Composite multi-field evaluator: [`include/openpfc/kernel/field/composite_gradient.hpp`](../../include/openpfc/kernel/field/composite_gradient.hpp)
- Tuple protocol: [`include/openpfc/kernel/field/tuple_protocol.hpp`](../../include/openpfc/kernel/field/tuple_protocol.hpp)
- Driver loop: [`include/openpfc/kernel/simulation/for_each_interior.hpp`](../../include/openpfc/kernel/simulation/for_each_interior.hpp)
- Stepper factories (single + multi-field): [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)
- End-to-end single-field example: [`apps/heat3d/`](../../apps/heat3d/)
- Composite multi-field test: [`tests/unit/kernel/field/test_composite_gradient.cpp`](../../tests/unit/kernel/field/test_composite_gradient.cpp)
