<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Extending OpenPFC

OpenPFC is designed so application-specific physics can live outside the
framework repository. A downstream project can add models, field modifiers,
writers, and spatial setup while linking the installed `OpenPFC::openpfc`
target.

Read [Architecture](../concepts/architecture.md) first. It defines the stable
kernel, runtime, and frontend boundaries used by this guide. For a complete
out-of-tree executable, follow the
[Minimal custom application tutorial](../tutorials/custom_app_minimal.md).

## Choose the extension point

| Goal | Primary extension point | Starting point |
|------|-------------------------|----------------|
| Add a PDE or phase-field model | physics `step(t)` on a stack / `SimulationState` | `examples/04_diffusion_model.cpp`, `examples/12_cahn_hilliard.cpp` |
| Add a stiff spectral-ETD (PFC) model | `pfc::sim::SpectralETDPhysics` consumed by `SpectralETDSystem` | [Add a spectral-ETD physics](#add-a-spectral-etd-physics), `tests/fixtures/swift_hohenberg.hpp` |
| Add a config-selected initial or boundary condition | `pfc::FieldModifier` and a modifier catalog | `examples/10_ui_register_ic.cpp`, `examples/14_custom_field_initializer.cpp` |
| Apply programmatic field operations | Namespace free functions and field iteration helpers | [Functional field operations](../getting_started/functional_field_ops.md) |
| Add an output format | `pfc::ResultsWriter` or a writer catalog | `examples/11_write_results.cpp` |
| Add custom spatial interpretation | Domain and coordinate helper functions | `examples/17_custom_coordinate_system.cpp` |
| Build a JSON/TOML-driven binary | `make_simulation_session` / `pfc::sim::run` | [Minimal custom application](../tutorials/custom_app_minimal.md) |
| Add point-wise gradients or finite-difference physics | Field/gradient primitives and halo policies | [Per-point gradients](per_point_grads.md), [Halo exchange](../concepts/halo_exchange.md) |
| Couple an external solver | `pfc::coupling::FieldHandle` + `Time::clip_attempt_dt` | [External coupling](external_coupling.md), `examples/22_external_coupling.cpp` |
| Restart from a checkpoint bundle | `CheckpointService` / `restart_from` | [Checkpoint publication](../development/checkpoint_publish.md) |

The [Examples catalog](../reference/examples_catalog.md) is the authoritative
inventory of runnable examples.


## Add a spectral-ETD physics

Stiff PFC-type models (tungsten, aluminum, Swift-Hohenberg) do not write a
time loop. They describe themselves to `pfc::sim::SpectralETDSystem<Physics,
MemorySpace>` (`include/openpfc/kernel/simulation/spectral_etd_system.hpp`),
which runs on the host and on CUDA/HIP from the same physics source. A physics
type provides:

| Member | Required | Meaning |
|--------|----------|---------|
| `declare_fields(SimulationState&)` | yes | Allocate the primary field `psi` (see `pfc::sim::add_declared_field`). |
| `linear_symbol(double k_laplacian)` | yes | Real diagonal symbol \(L(k)\) from OpenPFC's spectral Laplacian \(-|k|^2\). |
| `pointwise()` | yes | Returns a trivially copyable functor with `OPENPFC_HD double nonlinearity(const pfc::sim::SpectralCell&) const`. |
| `nonlinear_symbol(double k_laplacian)` | optional | Multiplier \(M(k)\) on \(\hat N\) (default 1; PFC models return `k_laplacian`). |
| `filter_mf(double k_laplacian)` | optional | Mean-field filter \(\chi(k)\); the driver then fills `cell.psi_mf`. |
| `correlation_kernel(double k_laplacian)` | optional | Kernel \(P(k)\); the driver then fills `cell.p_star`. |
| functor `free_energy_density(const SpectralCell&)` | optional | Per-cell observable reduced into `last_free_energy()`. |

The `SpectralCell` (`kernel/simulation/spectral_pointwise.hpp`) carries
`psi`, `psi_mf`, `p_star`, the cell coordinates `x`, `y`, `z`, and the time
`t`. The functor must be `OPENPFC_HD` and self-contained (all constants by
value) because the driver launches it inside a GPU kernel. The concepts live in
`kernel/simulation/physics_concepts.hpp` (`SpectralETDPhysics`,
`HasMeanFieldFilter`, `HasCorrelationKernel`, `HasNonlinearSymbol`).

The minimal example is
[`tests/fixtures/swift_hohenberg.hpp`](../../tests/fixtures/swift_hohenberg.hpp)
with its functor in
[`tests/fixtures/spectral_etd_toys_pointwise.hpp`](../../tests/fixtures/spectral_etd_toys_pointwise.hpp);
production examples are
[`apps/tungsten/include/tungsten/tungsten_physics.hpp`](../../apps/tungsten/include/tungsten/tungsten_physics.hpp)
and
[`apps/aluminumNew/include/aluminum/aluminum_physics.hpp`](../../apps/aluminumNew/include/aluminum/aluminum_physics.hpp).

To drive the physics from JSON, add `static Physics from_json(const
nlohmann::json& params, const Domain&, const Box3i&)` and use
`pfc::ui::SpectralETDSession<Physics, Stack>`
(`include/openpfc/frontend/ui/json_spectral_etd_session.hpp`) with
`pfc::ui::run_json_session_main<Session>` as `main()`. The session wires
initial and boundary conditions from the `FieldModifier` catalog, writers from
the `ResultsWriter` catalog, `CheckpointService`, and the JSON `profiling`
section.

GPU builds need exactly one CUDA or HIP translation unit that instantiates the
functor for the device launcher; keep it free of JSON and session headers:

```cpp
// src/gpu/my_pointwise.inc, stamped into my_pointwise.cu and my_pointwise.hip
#include <my_app/my_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(my_app::MyPointwise)
```

Add that source to every GPU target that instantiates
`SpectralETDSystem<Physics, CUDASpace | HIPSpace>`; a missing instantiation
fails at link time with the functor's name in the undefined symbol. The tungsten
and aluminum `CMakeLists.txt` show the pattern.

## API style

OpenPFC favors data-centric types and namespace free functions. Use inheritance
where the framework needs a runtime extension seam, such as `FieldModifier` or
`ResultsWriter`; keep the implementation behind that seam in ordinary functions
and small data types.

This keeps physics code testable and avoids deep class hierarchies. The complete
conventions are in the [Style guide](../development/styleguide.md).

## Minimal config-driven project

A downstream application commonly contains:

| File | Responsibility |
|------|----------------|
| `your_physics.hpp` and implementation files | Define fields, initialization, and the time step |
| `main.cpp` | Build a session or stack and call `pfc::sim::run`; register optional catalogs |
| `CMakeLists.txt` | Find OpenPFC and link `OpenPFC::openpfc` |
| JSON or TOML input | Define domain, time integration, planner options, modifiers, and writers |

The minimal CMake shape uses the same target as the packaging smoke test:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_openpfc_app LANGUAGES C CXX)

find_package(OpenPFC REQUIRED)

add_executable(my_openpfc_app main.cpp your_model.cpp)
target_link_libraries(my_openpfc_app PRIVATE OpenPFC::openpfc)
```

Frontend headers may require additional packages used directly by the
application, such as `nlohmann_json`. The complete wiring and run command belong
in the [custom application tutorial](../tutorials/custom_app_minimal.md), not in
this overview.

## Configuration and registration

The frontend converts JSON or TOML into a domain, decomposition, FFT stack,
model, simulator, modifiers, and writers. The ownership and wiring order are
described in [Application pipeline](../user_guide/app_pipeline.md). Exact keys
belong in the
[Spectral App configuration reference](../reference/spectral_app_config_reference.md).

For custom initial and boundary conditions, prefer an explicit local
`FieldModifierCatalog` in tests and reusable libraries. Process-wide registration
is convenient for simple applications but introduces shared mutable state.

## Validation

A model can validate required parameters, types, ranges, units, and typical
values before time integration begins. See
[Parameter validation](../user_guide/parameter_validation.md) and the tungsten
application for a larger metadata-driven example.

## Backend-specific work

Kernel code must remain independent of CUDA and HIP implementation headers.
Backend-specific execution, memory, and FFT functionality belongs under the
runtime layer. Review [Architecture](../concepts/architecture.md) and the
[GPU path guide](../hpc/gpu_path_decision.md) before adding a new backend path.

## Review checklist

Before proposing an extension:

- identify the layer that owns the new behavior;
- reuse an existing extension point when one matches;
- avoid copying complete option or configuration references into examples;
- add a focused test and a runnable example when the behavior is user-facing;
- update the canonical guide or reference page for any new public contract;
- record release-visible changes under `[Unreleased]` in the changelog.
