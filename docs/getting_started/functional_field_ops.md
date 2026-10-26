<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Functional field operations (IC / BC)

Coordinate-space callbacks set initial conditions and modify selected regions
without repeating index loops. The current API takes a writable host buffer,
a `Domain` describing physical coordinates, and a `Box3i` describing the global
indices owned by that buffer. It does not require the removed `Model` or `World`
classes.

## Basics

The functions in `openpfc/kernel/field/operations.hpp` have these contracts:

| Function | Callback returns a new value from |
|---|---|
| `pfc::field::apply(u, domain, box, fn)` | `fn(const Real3 &x)` |
| `pfc::field::apply_with_time(u, domain, box, t, fn)` | `fn(const Real3 &x, double t)` |
| `pfc::field::apply_inplace(u, domain, box, fn)` | `fn(const Real3 &x, double current)` |
| `pfc::field::apply_inplace_with_time(u, domain, box, t, fn)` | `fn(const Real3 &x, double current, double t)` |

Here is a complete setup for a single owned box, followed by a constant
initial condition. `std::vector<double>` converts to the non-owning
`FieldOutput<double>` expected by these functions. Its length must equal
`box.count()`; a mismatch throws `std::invalid_argument`.

```cpp
#include <cmath>
#include <numbers>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>

using pfc::Real3;
namespace field = pfc::field;

int main() {
  const auto domain = pfc::domain::create({32, 16, 8});
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {31, 15, 7});
  std::vector<double> u(static_cast<std::size_t>(box.count()));

  // Inside the setup or initial-condition callback:
  field::apply(u, domain, box, [](const Real3 &) { return 0.5; });
}

```

Place the following fragments inside `main` after the setup; they reuse
`domain`, `box` and `u`.
Physical coordinates are `origin + global_index * spacing`. The box has
inclusive lower and upper indices, and storage is contiguous with x varying
fastest. For an MPI run use the rank's owned box, not the global box, and
allocate only its local cell count.

## Gaussian pulse

```cpp
field::apply(u, domain, box, [](const Real3 &x) {
  const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  return std::exp(-r2 / 2.0);
});
```

This is a Gaussian centred at the coordinate origin with unit width. It is
not automatically periodized; choose the domain origin and width to suit the
problem. The [Gaussian heat-spreading lesson](https://vtt-propertune.github.io/OpenPFC/learn/heat/gaussian-pulse/)
explains the distinction between a free-space Gaussian and a periodic profile.

## Time-varying boundary-like pattern

```cpp
const double frequency = 1.0;
const double t = 0.25;
field::apply_with_time(u, domain, box, t,
    [frequency](const Real3 &x, double time) {
      return std::sin(2.0 * std::numbers::pi * frequency * time)
             * (x[0] > 10.0 ? 1.0 : 0.0);
    });
```

This replaces every owned value. It illustrates a prescribed spatial band;
it does not impose a physical boundary condition on an FFT, whose axes remain
periodic. See [boundary transforms](../api/boundary_transforms.md) for bounded
axes, and [implicit interfaces](../concepts/implicit_interface.md) for geometry.

## In-place updates (partial modifications)

Return the current value outside the region you want to change:

```cpp
const double xwidth = 5.0;
const double xpos = 10.0;
const double alpha = 1.0;
const double rho_low = 0.0;
const double rho_high = 1.0;

field::apply_inplace(u, domain, box, [=](const Real3 &x, double current) {
  if (std::abs(x[0] - xpos) < xwidth) {
    const double blend = 1.0 / (1.0 + std::exp(-alpha * (x[0] - xpos)));
    return rho_low * blend + rho_high * (1.0 - blend);
  }
  return current;
});
```

An additive update follows the same contract:

```cpp
field::apply_inplace(u, domain, box, [](const Real3 &x, double current) {
  return current + 0.01 * std::sin(x[0]);
});
```

Callbacks update one value at a time. They are suitable for pointwise
initialization and prescribed bands. Neighbour-dependent stencils need the
[gradient and halo interfaces](../extending_openpfc/per_point_grads.md).

## Migration from legacy loop-based patterns

Replace the nested loop with a coordinate callback, and pass current objects
explicitly. Code written against `get_real_field`, `get_world`, or a `Model`
subclass needs an ownership update as well as a shorter loop. Follow
[the migration guide](../MIGRATION_0.1_to_0.2.md) and
[custom stepper integration](../user_guide/custom_stepper_integration.md).

The FFT convenience overload accepts a `const pfc::fft::IHostFFT &` in place
of `box` and forwards to its real-space inbox. Finite-difference applications
can use `apply_subdomain(vector, decomposition, rank, fn)`, whose callback
receives three scalar coordinates and whose output vector is resized to the
rank's owned cell count.

## Notes

- These helpers operate on host-accessible contiguous owned values. Do not pass
  a device pointer or a padded allocation containing halo cells.
- Each rank handles its local values; the helpers perform no halo exchange or
  collective communication.
- Keep callbacks free of shared mutable state so iteration order does not
  affect the result.
- Existing [field-operation unit tests](../../tests/unit/kernel/field/test_field_operations.cpp)
  exercise constants, time dependence, selective updates and coordinate offsets.

## See also

- [Learning paths](../learning_paths.md) — routes for application and library authors.
- [Tour of main types](../reference/class_tour.md) — current ownership and API roles.
- [Extending OpenPFC](../extending_openpfc/README.md) — extension contracts.
