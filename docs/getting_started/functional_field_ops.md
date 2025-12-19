#Functional Field Operations(IC / BC)

This page shows how to use the new coordinate-space functional API to set initial and boundary conditions without writing manual nested loops.

## Basics

- `pfc::field::apply(model, field_name, Fn)` applies `Fn(const Real3&) -> double` over the local FFT inbox
- `pfc::field::apply_with_time(model, field_name, t, Fn)` applies `Fn(const Real3&, double)` with a time parameter
- `pfc::field::apply_inplace(model, field_name, Fn)` applies `Fn(const Real3&, double current) -> double` for partial updates
- `pfc::field::apply_inplace_with_time(model, field_name, t, Fn)` applies `Fn(const Real3&, double current, double t)` with time

### Constant initial condition

```cpp
using namespace pfc;
field::apply(model, "psi", [](const Real3 &) { return 0.5; });
```

    ## #Gaussian pulse

```cpp field::apply(model, "psi", [](const Real3 &x) {
      const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
      return std::exp(-r2 / 2.0);
    });
```

    ## #Time -
    varying boundary -
    like pattern

```cpp const double freq = 1.0;
field::apply_with_time(model, "psi", t, [freq](const Real3 &x, double tt) {
  return std::sin(2.0 * M_PI * freq * tt) * (x[0] > 10.0 ? 1.0 : 0.0);
});
```

        ##In -
        place Updates(Partial Modifications)

            Use `apply_inplace` when you need to modify only certain regions
    or when the new value depends on the current value.

       ## #Boundary band with smooth transition

```cpp const double xwidth = 20.0;
const double xpos = 100.0;
const double alpha = 1.0;
const double rho_low = 0.0;
const double rho_high = 1.0;

field::apply_inplace(model, "psi", [=](const Real3 &x, double current) {
  if (std::abs(x[0] - xpos) < xwidth) {
    double S = 1.0 / (1.0 + std::exp(-alpha * (x[0] - xpos)));
    return rho_low * S + rho_high * (1.0 - S);
  }
  return current; // outside band: preserve value
});
```

    ## #Masked update(modify only where condition is true)

```cpp field::apply_inplace(model, "psi", [](const Real3 &x, double current) {
      if (x[0] > 0.0 && x[2] < 10.0) {
        return 0.5; // set value in region
      }
      return current; // keep existing value elsewhere
    });
```

    ## #Accumulate or
    blend

```cpp field::apply_inplace(model, "psi", [](const Real3 &x, double current) {
      const double perturbation = 0.01 * std::sin(x[0]);
      return current + perturbation; // additive update
    });
```

    ##Backward compatibility via adapter

        You can wrap a lambda into a `FieldModifier` that works with existing
            Simulator APIs
    :

```cpp auto mod =
        field::make_legacy_modifier("psi", [](const Real3 &) { return 0.5; });
simulator.add_initial_condition(std::move(mod));
```

    ##Migration from Legacy Loop -
    Based Patterns

    Before(manual nested loops)
    :
```cpp void apply(Model & m, double) override {
  const FFT &fft = m.get_fft();
  Field &field = m.get_real_field(get_field_name());
  const World &w = m.get_world();
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

    After(functional)
    :
```cpp void apply(Model & m, double) override {
  pfc::field::apply(m, get_field_name(), [](const pfc::Real3 &X) {
    return compute_value(X[0], X[1], X[2]);
  });
}
```

    Benefits : -No manual index management -
               Coordinates automatically computed via `world::to_coords()` -
               Clearer intent : focus on the computation,
    not the iteration -
        Less boilerplate(15 lines → 5 lines)

            ##Notes
        - Operates over the local inbox only(MPI - friendly) - Header - only,
    no allocations; suitable for hot paths when function is simple
- Prefer pure functions without side-effects for clarity and performance
- Use `apply_inplace` when new value depends on current value or for partial updates
- All migrated ICs/BCs in OpenPFC now use this functional API (see `include/openpfc/initial_conditions/` and `include/openpfc/boundary_conditions/`)
