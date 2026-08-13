<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Functional Field Operations (IC / BC)

This page shows how to use the coordinate-space functional API to set initial and boundary conditions without writing manual nested loops.

## Contents

- [Basics](#basics)
- [Gaussian pulse](#gaussian-pulse)
- [Time-varying boundary-like pattern](#time-varying-boundary-like-pattern)
- [In-place updates (partial modifications)](#in-place-updates-partial-modifications)
- [Migration from legacy loop-based patterns](#migration-from-legacy-loop-based-patterns)
- [Notes](#notes)

## Basics

- `pfc::field::apply(field, world, fft, Fn)` applies `Fn(const Real3&) -> double` over the local FFT inbox
- `pfc::field::apply_with_time(field, world, fft, t, Fn)` applies `Fn(const Real3&, double)` with a time parameter
- `pfc::field::apply_inplace(field, world, fft, Fn)` applies `Fn(const Real3&, double current) -> double` for partial updates
- `pfc::field::apply_inplace_with_time(field, world, fft, t, Fn)` applies `Fn(const Real3&, double current, double t)` with time

From a `Model`, pass `get_real_field(m, name)`, `get_world(m)`, and `get_fft(m)`.

### Constant initial condition

```cpp
using namespace pfc;
field::apply(u, world, fft, [](const Real3 &) { return 0.5; });
```

## Gaussian pulse

```cpp
field::apply(u, world, fft, [](const Real3 &x) {
  const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  return std::exp(-r2 / 2.0);
});
```

## Time-varying boundary-like pattern

```cpp
const double freq = 1.0;
field::apply_with_time(u, world, fft, t, [freq](const Real3 &x, double tt) {
  return std::sin(2.0 * M_PI * freq * tt) * (x[0] > 10.0 ? 1.0 : 0.0);
});
```

## In-place updates (partial modifications)

Use `apply_inplace` when you need to modify only certain regions, or when the new value depends on the current value.

### Boundary band with smooth transition

```cpp
const double xwidth = 20.0;
const double xpos = 100.0;
const double alpha = 1.0;
const double rho_low = 0.0;
const double rho_high = 1.0;

field::apply_inplace(u, world, fft, [=](const Real3 &x, double current) {
  if (std::abs(x[0] - xpos) < xwidth) {
    double S = 1.0 / (1.0 + std::exp(-alpha * (x[0] - xpos)));
    return rho_low * S + rho_high * (1.0 - S);
  }
  return current; // outside band: preserve value
});
```

### Masked update (modify only where condition is true)

```cpp
field::apply_inplace(u, world, fft, [](const Real3 &x, double current) {
  if (x[0] > 0.0 && x[2] < 10.0) {
    return 0.5; // set value in region
  }
  return current; // keep existing value elsewhere
});
```

### Accumulate or blend

```cpp
field::apply_inplace(u, world, fft, [](const Real3 &x, double current) {
  const double perturbation = 0.01 * std::sin(x[0]);
  return current + perturbation; // additive update
});
```

## Migration from legacy loop-based patterns

Before (manual nested loops):

```cpp
void apply(Model &m, double) override {
  const FFT &fft = m.get_fft();
  Field &field = m.get_real_field(get_field_name());
  const Domain &w = m.get_domain();
  Int3 low = get_inbox(fft).low;
  Int3 high = get_inbox(fft).high;
  auto [dx, dy, dz] = get_spacing(w);
  auto [x0, y0, z0] = get_origin(w);

  long int idx = 0;
  for (int k = low[2]; k <= high[2]; k++) {
    for (int j = low[1]; j <= high[1]; j++) {
      for (int i = low[0]; i <= high[0]; i++) {
        double x = x0 + i * dx;
        double y = y0 + j * dy;
        double z = z0 + k * dz;
        field[idx++] = compute_value(x, y, z);
      }
    }
  }
}
```

After (functional):

```cpp
void apply(Model &m, double) override {
  pfc::field::apply(pfc::get_real_field(m, get_field_name()),
                    pfc::get_world(m), pfc::get_fft(m),
                    [](const pfc::Real3 &X) {
    return compute_value(X[0], X[1], X[2]);
  });
}
```

Benefits:

- No manual index management; coordinates come from the Domain/FFT inbox.
- Clearer intent: focus on the computation, not the iteration.
- Less boilerplate (many lines → a short lambda).

## Notes

- Operates over the local inbox only (MPI-friendly).
- Prefer pure functions without side effects for clarity and performance.
- Use `apply_inplace` when the new value depends on the current value or for partial updates.
- Related headers live under `include/openpfc/kernel/simulation/` (e.g. initial conditions and modifiers used by the simulator).

## See also

- [`../learning_paths.md`](../learning_paths.md) — extend track links functional IC/BC patterns
- [`../class_tour.md`](../reference/class_tour.md) — `FieldModifier`, `Simulator`
- [`../extending_openpfc/README.md`](../extending_openpfc/README.md) — extension checklist
