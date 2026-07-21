<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Time Integration Architecture Contract

This document defines the formal contract between time integrators and the OpenPFC simulator framework. It specifies the responsibilities of integrator implementations, state access patterns, boundary preparation ordering, and output scheduling guarantees. All integrator implementations must adhere to this contract to ensure correct behavior across CPU and accelerator backends.

## 1. Contract scope and responsibilities

### Integrator role

A time integrator is responsible for managing the advancement of simulation time through one or more computational stages. The integrator coordinates physics evaluation with boundary condition application and respects output scheduling. It acts as the orchestration layer between the temporal evolution logic and the spatial discretization implemented by the model.

The formal contract boundary is defined by two hooks in the `Simulator` class:

- `Simulator::begin_integrator_step()` — prologue called before physics evaluation
- `Simulator::end_integrator_step()` — epilogue called after physics evaluation

These hooks are implemented in [`simulator_integrator::begin_integrator_step()`](../../include/openpfc/kernel/simulation/simulator_integrator.hpp) and [`simulator_integrator::end_integrator_step()`](../../include/openpfc/kernel/simulation/simulator_integrator.hpp).

### When to use `step_with_physics()` vs direct hooks

For simple single-stage integrators, use `Simulator::step_with_physics()` which internally calls the begin/end hooks:

```cpp
simulator.step_with_physics([&]() {
    model.step(time.get_current());
});
```

For multi-stage integrators or custom orchestration, call the hooks explicitly to maintain control over intermediate computations:

```cpp
simulator.begin_integrator_step();
// Perform multi-stage computations here
simulator.end_integrator_step();
```

The `step_with_physics()` method is equivalent to `begin_integrator_step(); physics_fn(); end_integrator_step()` and should be preferred when the physics body is a single function call.

## 2. Simulator hook semantics

### Full ordering contract

The `simulator_integrator::begin_integrator_step()` function implements the following ordering contract:

1. **Initial condition path** (when `pfc::time::increment(time) == 0`):
   - Apply initial conditions via `Simulator::apply_initial_conditions()`
   - Apply boundary conditions via `Simulator::apply_boundary_conditions()`
   - Optionally write results if `pfc::time::do_save(time)` returns true
   - Advance time via `pfc::time::next(time)`
   - Apply boundary conditions again at the new time

2. **Steady-state path** (when `pfc::time::increment(time) > 0`):
   - Advance time via `pfc::time::next(time)`
   - Apply boundary conditions at the new time

The `simulator_integrator::end_integrator_step()` function:

- Optionally writes results if `pfc::time::do_save(time)` returns true

### State access patterns

Integrators may read and write model fields through the `Model` API:

- `Model::get_real_field(std::string_view name)` — access real-valued fields
- `Model::get_complex_field(std::string_view name)` — access complex-valued fields

These methods return references to field storage that can be modified directly by the integrator. All field access must respect the halo exchange requirements specified in [Section 4](#4-boundary-and-halo-preparation).

The `Time` object provides temporal state queries:

- `Time::get_current()` — current physical time
- `Time::get_increment()` — current step counter (0-based)
- `Time::do_save()` — whether output should be written at this time
- `Time::next()` — advance to next time step

## 3. Physics evaluation patterns

### Single-stage integrators

Single-stage integrators (e.g., explicit Euler) call `Model::step(double t)` exactly once per `begin_integrator_step()`/`end_integrator_step()` pair:

```cpp
simulator.begin_integrator_step();
model.step(time.get_current());
simulator.end_integrator_step();
```

The existing [`pfc::sim::steppers::EulerStepper`](../../include/openpfc/kernel/simulation/steppers/euler.hpp) provides a reference implementation for single-stage integration with the signature:

```cpp
double step(double t, std::vector<double>& u);
```

### Multi-stage integrators

**Critical contract requirement:** `begin_integrator_step()` and `end_integrator_step()` hooks bracket the **full timestep**, not each substage. All intermediate substage computations occur between these two calls with no additional hook invocations.

For example, a 4th-order Runge-Kutta integrator performs all four slope evaluations between `begin_integrator_step()` and `end_integrator_step()`:

```cpp
simulator.begin_integrator_step();

// All RK4 substages occur here, with no hook calls
double t = time.get_current();
auto& u = model.get_real_field("density");

auto k1 = compute_rhs(t, u);
auto k2 = compute_rhs(t + 0.5*dt, u + 0.5*dt*k1);
auto k3 = compute_rhs(t + 0.5*dt, u + 0.5*dt*k2);
auto k4 = compute_rhs(t + dt, u + dt*k3);

u += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4);

simulator.end_integrator_step();
```

This design ensures that boundary conditions and output scheduling are applied only at the physical time level, not at intermediate substage times that may not correspond to meaningful physical states.

### Physics evaluation API

The physics evaluation is performed through the `Model::step(double t)` method:

```cpp
virtual void step(double t) = 0;
```

This pure virtual method must be implemented by concrete model classes. It receives the current physical time and is responsible for evolving the model state by one time step. The implementation typically involves:

1. Computing spatial derivatives (spectral or finite difference)
2. Applying nonlinear terms
3. Updating field values using the chosen time integration scheme

## 4. Boundary and halo preparation

### Halo exchange architecture

The halo exchange architecture is specified in [`docs/concepts/halo_exchange.md`](halo_exchange.md). Integrators must respect the halo exchange requirements when using real-space stencils on distributed domains.

### Finite difference integrators

Finite difference integrators **must** exchange halos before physics evaluation when using real-space stencils. The halo exchange synchronizes ghost cells between neighboring ranks to ensure stencil operations have access to valid neighbor data:

```cpp
// Example: FD integrator with halo exchange
auto& u = model.get_real_field("density");
auto& lap = model.get_real_field("laplacian");

// Exchange halos before physics evaluation
exchanger.exchange_halos(u.data(), u.size());
halo::copy_to_face_layout(exchanger, face_halos);

// Compute physics with valid ghost data
field::fd::laplacian_periodic_separated<2>(
    u.data(), face_ptrs, lap.data(),
    nx, ny, nz, inv_dx2, inv_dx2, inv_dx2, halo_width
);
```

### Spectral integrators

Spectral integrators typically skip halo exchange because the FFT-based operations operate on global k-space representations where neighbor communication is handled implicitly by the FFT decomposition.

### Boundary condition application

Boundary conditions are applied in two places:

1. **In `begin_integrator_step()`** — at `increment==0` (after ICs) and after `time.next()`
2. **Post-physics** — required for multi-stage schemes where intermediate states violate boundary constraints

**Single-stage integrators** generally do not require post-physics BC reapplication because the physics update preserves boundary-enforced values.

**Multi-stage integrators** may require post-physics BC application depending on the scheme:

- **Predictor-corrector methods** that compute temporary states outside valid bounds must reapply BCs after each substage that produces a physically invalid intermediate state
- **Runge-Kutta methods** typically do not require intermediate BC application unless the problem has strong boundary constraints that must be enforced at all times
- **Split-step methods** may require BC application after each substep if the substep modifies boundary values

Example of post-physics BC application in a predictor-corrector scheme:

```cpp
simulator.begin_integrator_step();

// Predictor step (may violate BCs)
auto& u = model.get_real_field("density");
auto u_pred = u + dt * compute_rhs(t, u);

// Reapply BCs if predictor violates constraints
simulator.apply_boundary_conditions();

// Corrector step
auto rhs_corrector = compute_rhs(t + dt, u_pred);
u += 0.5 * dt * (compute_rhs(t, u) + rhs_corrector);

simulator.end_integrator_step();
```

## 5. Output scheduling guarantees

### Save point detection

Output scheduling is controlled by the `Time` object through `Time::do_save()`. This method returns true when the current time corresponds to a scheduled save point based on the `saveat` interval:

```cpp
if (time.do_save()) {
    simulator.write_results();
}
```

The `Time::saveat()` interval uses floating-point modulo with a tolerance of 1e-6 to determine save points. A `saveat` value of 0.0 disables automatic saving.

### Result writing mechanism

Result writing is performed through `Simulator::write_results()`, which internally calls `write_scheduled_simulator_results()`. This function:

1. Dispatches all registered `ResultsWriters` with the current field state
2. Increments the internal result counter via `simulator.set_result_counter(file_num + 1)`

The result counter is used by writers to generate incrementing filenames (e.g., `field_0000.bin`, `field_0001.bin`).

### Custom integrator responsibilities

Custom integrators must respect `Time::saveat()` semantics or call `Simulator::write_results()` explicitly at save points. The recommended approach is to rely on the hook semantics:

```cpp
simulator.begin_integrator_step();  // Handles initial save if needed
// Perform physics
simulator.end_integrator_step();    // Handles final save if needed
```

For integrators that need custom output timing, explicitly check `time.do_save()` and call `write_results()`:

```cpp
if (time.do_save()) {
    simulator.write_results();
}
```

## 6. Restart and checkpoint semantics

### Checkpoint writing

Scheduled field dumps still go through `ResultsWriters` / frontend
`BinaryWriter` as **headerless** MPI-IO bricks (periodic output and
post-processing). That raw layout is specified in
[`docs/reference/binary_field_io_spec.md`](../reference/binary_field_io_spec.md).

For a **durable accepted-state restart bundle**, use
`pfc::checkpoint::publish_checkpoint_directory` (kernel headers under
`include/openpfc/kernel/checkpoint/`). It stages versioned `metadata.json`
plus accepted field bricks, then atomically renames the staging directory to
the final path so incomplete writes are never loadable. See
[`docs/development/checkpoint_publish.md`](../development/checkpoint_publish.md).
Restore / migration validation is not claimed here.

Key properties of the **headerless scheduled dump** format:

- **Layout:** Single global 3D array in Fortran (column-major) order
- **Element type:** `double` for real fields, `std::complex<double>` for complex fields
- **Byte order:** Native to the writing machine
- **No header:** Files contain only raw payload data
- **Per-rank data:** Each rank writes its local brick, together covering the global grid

### Resume expectations

Restart logic expects:

1. **Consistent field state** — All fields must contain values matching the checkpoint time
2. **Matching `Time` state** — The `Time` object must be configured with the same `t0`, `t1`, `dt`, and `saveat` values used when the checkpoint was written
3. **Increment alignment** — The `Time::get_increment()` value should match the increment number encoded in the checkpoint filename

When reading a checkpoint, the `BinaryReader` uses the same MPI-IO subarray view as the writer, requiring the same communicator, decomposition, and data type configuration.

## 7. CPU and accelerator numerical contract

### Numerical equivalence requirement

Both CPU and GPU paths must produce identical results within floating-point tolerance. This requirement ensures that:

- Simulations can be developed and debugged on CPU workstations
- Results are reproducible across different hardware backends
- Validation studies are not tied to a specific platform

### Architecture layering

The kernel/runtime separation described in [`docs/concepts/architecture.md`](architecture.md) enforces this contract:

- **Kernel layer** — Backend-agnostic numerical algorithms and data structures
- **Runtime layer** — Backend-specific implementations (CPU, CUDA, HIP)

Numerical behavior is defined at the kernel layer and must be preserved by all runtime implementations.

### GPU implementation obligations

GPU implementations of steppers (via `runtime/cuda` or `runtime/hip`) must:

1. **Obey the same stage ordering** as CPU implementations
2. **Use equivalent numerical algorithms** — same order of operations, same stencil coefficients
3. **Handle boundary conditions identically** — same BC application points and logic
4. **Respect the same halo exchange contract** — same ghost cell synchronization patterns

Deviations from these requirements are acceptable only when documented as platform-specific limitations with clear justification.

## 8. Example integration patterns

### Single-stage Euler using `step_with_physics()`

```cpp
#include <openpfc/kernel/simulation/simulator.hpp>

void run_euler_simulation(Simulator& simulator, Model& model) {
    auto& time = simulator.get_time();

    while (!time.done()) {
        simulator.step_with_physics([&]() {
            model.step(time.get_current());
        });
    }
}
```

This pattern is used in the time stepping demo at [`docs/api/examples/04_time_stepping.cpp`](../../docs/api/examples/04_time_stepping.cpp).

### Multi-stage integrator with explicit `begin/end` calls

```cpp
#include <openpfc/kernel/simulation/simulator.hpp>

class RK4Integrator {
public:
    void step(Simulator& simulator, Model& model, double dt) {
        auto& time = simulator.get_time();
        auto& u = model.get_real_field("density");

        simulator.begin_integrator_step();

        double t = time.get_current();
        auto u_backup = u;  // Save initial state

        // Stage 1
        auto k1 = compute_rhs(model, t, u);

        // Stage 2
        u = u_backup + 0.5 * dt * k1;
        auto k2 = compute_rhs(model, t + 0.5*dt, u);

        // Stage 3
        u = u_backup + 0.5 * dt * k2;
        auto k3 = compute_rhs(model, t + 0.5*dt, u);

        // Stage 4
        u = u_backup + dt * k3;
        auto k4 = compute_rhs(model, t + dt, u);

        // Final combination
        u = u_backup + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

        simulator.end_integrator_step();
    }

private:
    std::vector<double> compute_rhs(Model& model, double t,
                                    const std::vector<double>& u);
};
```

### Custom integrator with halo exchange

```cpp
#include <openpfc/kernel/decomposition/sparse_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

class FDHeatIntegrator {
public:
    FDHeatIntegrator(SparseHaloExchanger<double>& exchanger,
                     std::array<std::vector<double>, 6>& face_halos,
                     int nx, int ny, int nz, double dx, double D)
        : m_exchanger(exchanger), m_face_halos(face_halos),
          m_nx(nx), m_ny(ny), m_nz(nz),
          m_inv_dx2(1.0 / (dx * dx)), m_D(D) {}

    void step(Simulator& simulator, Model& model, double dt) {
        auto& u = model.get_real_field("temperature");
        auto& lap = model.get_real_field("laplacian");

        simulator.begin_integrator_step();

        // Exchange halos before physics evaluation
        m_exchanger.exchange_halos(u.data(), u.size());
        halo::copy_to_face_layout(m_exchanger, m_face_halos);

        // Build face pointer array for Laplacian
        std::array<const double*, 6> face_ptrs;
        for (int i = 0; i < 6; ++i) {
            face_ptrs[i] = m_face_halos[i].data();
        }

        // Compute Laplacian using finite differences
        field::fd::laplacian_periodic_separated<2>(
            u.data(), face_ptrs, lap.data(),
            m_nx, m_ny, m_nz,
            m_inv_dx2, m_inv_dx2, m_inv_dx2, 1
        );

        // Explicit Euler update
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] += dt * m_D * lap[i];
        }

        simulator.end_integrator_step();
    }

private:
    SparseHaloExchanger<double>& m_exchanger;
    std::array<std::vector<double>, 6>& m_face_halos;
    int m_nx, m_ny, m_nz;
    double m_inv_dx2;
    double m_D;
};
```

This pattern extends the finite difference heat example at [`examples/15_finite_difference_heat.cpp`](../../examples/15_finite_difference_heat.cpp) into a full integrator with simulator hook integration.

---

**See also:** [`docs/concepts/architecture.md`](architecture.md) for overall system architecture, [`docs/concepts/halo_exchange.md`](halo_exchange.md) for halo exchange patterns, [`docs/development/checkpoint_publish.md`](../development/checkpoint_publish.md) for atomic accepted-state publication, and [`docs/reference/binary_field_io_spec.md`](../reference/binary_field_io_spec.md) for headerless scheduled field dumps.
