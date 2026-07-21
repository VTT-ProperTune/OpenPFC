<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Per-point grads aggregates

OpenPFC's point-wise driver loop, [`pfc::sim::for_each_interior`](../../include/openpfc/kernel/simulation/for_each_interior.hpp), is the canonical way to evaluate a model's right-hand side cell by cell:

```
du[{i,j,k}] = model.rhs(t, eval(i,j,k))
```

The `eval(i,j,k)` call returns a **model-owned aggregate** that names exactly the partial derivatives the model needs, drawn from a fixed catalog. The kernel's per-point evaluators (`pfc::gradient::FDGradient<G>`, `pfc::field::SpectralGradient<G>`; `pfc::field::FdGradient<G>` is a deprecated alias retained for source compatibility) are templated on that aggregate `G` and use C++20 concepts to fill **only the members `G` declares** — no wasted FFTs, no wasted stencil sweeps.

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

The `apps/heat3d/` model is the canonical small example. The grads aggregate, the diffusion constant, and the model itself all live in one self-consistent file ([`apps/heat3d/include/heat3d/heat_model.hpp`](../../apps/heat3d/include/heat3d/heat_model.hpp)). The grads aggregate names exactly three slots:

```cpp
namespace heat3d {
struct HeatGrads { double xx{}, yy{}, zz{}; };
}
```

The model is **OpenPFC-free** — only `<cmath>` and `<functional>` are needed:

```cpp
inline constexpr double kD = 1.0;

struct HeatModel {
  inline double rhs(double, const HeatGrads& g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};
```

The **spectral** compact driver wires the evaluator behind `pfc::sim::DuField` so the user-facing time loop reads like the math. **`heat3d_spectral_pointwise`** uses `SpectralCpuStack`:

```cpp
auto& u  = stack.u();
auto  du = stack.du<heat3d::HeatGrads>();   // spectral path

for (int step = 0; step < n_steps; ++step) {
  du.apply([](const heat3d::HeatGrads& g) { return heat3d::kD * (g.xx + g.yy + g.zz); });
  u += dt * du;                              // explicit Euler, on the page
  t  += dt;
}
```

The **FD** compact driver **`heat3d_fd`** instead spells the three primitives — halo, gradient, sweep — out in `main`, on top of two `pfc::field::PaddedBrick<double>` buffers (`u`, `du`) plus a `pfc::communication::PaddedHaloExchanger<double> halo(u, MPI_COMM_WORLD)` and a `pfc::gradient::FDGradient<heat3d::HeatGrads> grad(u, fd_order)` bound to the same `u`:

```cpp
for (int step = 0; step < n_steps; ++step) {
  pfc::communication::exchange(halo);
  pfc::field::for_each(du, [&](const auto& idx) {
    const auto g = pfc::gradient::evaluate(grad, idx);
    du[idx] = g.xx + g.yy + g.zz;   // D = 1 in this driver
  });
  u += dt * du;
  t += dt;
}
```

`pfc::communication::exchange` is the blocking one-shot; `start_exchange` / `finish_exchange` live in the same namespace for overlap. `pfc::gradient::evaluate` and the dimension-agnostic `pfc::Int3` callback keep the inner loop readable without hiding the halo or the evaluator. Programmatic code that needs an FFT-safe unpadded core continues to use **`FdCpuStack`** (`stack.du<G>()` wires `SparseHaloExchanger` + face buffers + `FDGradient` behind `DuField`). For non-trivial multi-field models, `pfc::sim::steppers::create(...)` plus `pfc::field::CompositeGradient<...>` remains the right tool (see `apps/kobayashi`).

## Backend capability matrix

Different backends can fulfill different subsets of the catalog. Asking for a member a backend cannot supply is a **compile-time error** (`static_assert`), not a silent zero.

| Backend | `value` | `x/y/z` | `xx/yy/zz` | `xy/xz/yz` |
|---------|---------|---------|------------|------------|
| `pfc::gradient::FDGradient<G>` | yes | yes — D1 orders 2..14 | yes — D2 orders 2..20 | not yet (host 26-fill via `pfc::communication::FullPaddedHaloExchanger`; member enablement is a follow-up — see also `pfc::cuda::FullPaddedDeviceHalo`) |
| `pfc::cuda::FdGradientDevice<G>` / `pfc::hip::FdGradientDevice<G>` | yes | yes — D1 orders 2..14 | yes — D2 orders 2..20 | yes — D1⊗D1 when paired with [`FullPaddedDeviceHalo`](../../include/openpfc/runtime/cuda/full_padded_device_halo.hpp) (or HIP twin / equivalent corner fill) |
| `pfc::field::SpectralGradient<G>` | yes | yes (via `i k_i`) | yes (via `-k_i^2`) | yes (via `-k_i k_j`) |

`FDGradient<G>`'s constructor consults `pfc::field::has_*<G>` for every member individually and throws `std::invalid_argument` if the requested `order` falls outside the tabulated range for any declared member (so a model that asks for `g.x` at order 16 surfaces as a clean error at construction time, not as silent zeros at runtime). The same fail-closed posture applies when `halo_width` is strictly less than the stencil half-width required by those members (`order/2` for tabulated even central D1/D2): both the CPU evaluator and the CUDA/HIP `FdGradientDevice` twins throw `std::invalid_argument` before any stencil read or kernel launch. Brick Laplacians (`laplacian_interior` and siblings) likewise throw instead of silently returning when `halo_width < Order/2`.

When new requirements show up:
- **FD higher-order first derivatives**: extend `EvenCentralD1<Order>` in [`fd_stencils.hpp`](../../include/openpfc/kernel/field/fd_stencils.hpp) — the closed form `c_k = (-1)^{k+1} (M!)^2 / (k (M-k)! (M+k)!)` produces the rational coefficients; build the integer table with their lowest common denominator and add the matching `lookup_even_central_d1` case.
- **FD mixed seconds (`xy/xz/yz`)**: device evaluators ([`pfc::cuda::FdGradientDevice`](../../include/openpfc/runtime/cuda/fd_gradient_device.hpp) / HIP twin) already populate them via separable D1⊗D1 when the padded buffer has corner-filled ghosts ([`FullPaddedDeviceHalo`](../../include/openpfc/runtime/cuda/full_padded_device_halo.hpp)). Host plumbing for the matching CPU path is [`pfc::communication::FullPaddedHaloExchanger`](../../include/openpfc/kernel/decomposition/full_padded_halo_exchange.hpp) (3-pass widening, 26-direction); host `FDGradient<G>` still compile-rejects mixed seconds until that follow-up lands. Until then, `SpectralGradient<G>` is the right CPU path for models that need cross terms.

## Custom stencils — Sobel, CNN-style filters, anisotropic FD

`FDGradient<G>` is the **PDE-specialised** evaluator: it consumes the central-difference tables in [`fd_stencils.hpp`](../../include/openpfc/kernel/field/fd_stencils.hpp) and returns the standard partial derivatives the model expects. For applications that want **arbitrary** stencils — Sobel-style edge detection, learned convolutional kernels, anisotropic FD on a non-uniform grid, separable Gaussian smoothing — OpenPFC ships a parallel **generic stencil layer** in [`stencil_apply.hpp`](../../include/openpfc/kernel/field/stencil_apply.hpp):

| Primitive | Use when... |
|---|---|
| `pfc::field::stencil::apply_1d_along<Axis>(coeffs, half_width, core, c, sx, sy, sz)` | The stencil acts along a single axis and the weights are arbitrary (asymmetric OK — first-derivative central FD, anisotropic 1D filter, ...). |
| `pfc::field::stencil::apply_separable(cx, Hx, cy, Hy, cz, Hz, core, c, sx, sy, sz)` | The 3D stencil factors as `cx ⊗ cy ⊗ cz` (3D Sobel, separable Gaussian, mixed seconds built from two 1D D1 tables, ...). |
| `pfc::field::stencil::apply_dense<Nz, Ny, Nx>(weights, core, c, sx, sy, sz)` | The 3D stencil does **not** factor (rotationally-invariant FCC Laplacian, fully connected CNN filter, multiscale corner-emphasis filter). Compile-time extents `Nz x Ny x Nx` (each odd, each `>= 1`) so the loop unrolls. |

The contract is identical to `apply_d2_along`: the caller passes a row-major buffer, a centre linear index, and three strides; the primitive returns the **already-scaled** weighted sum. Custom evaluators are easy to write — see the [`examples/16_sobel_edge_detection.cpp`](../../examples/) walkthrough for the full pattern (`prepare()` is a no-op, `operator()(i,j,k)` calls `apply_dense` once and stuffs the result into the model-owned aggregate).

This is the laboratory layer: nothing in `apply_dense` is PDE-specific. If you need to plug a custom stencil through `for_each_interior`, write a small `MyStencilGradient<G>` evaluator that wraps it (~30 lines) and pass it to `pfc::sim::steppers::create(...)` exactly the same way you would `pfc::gradient::FDGradient<G>`.

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
