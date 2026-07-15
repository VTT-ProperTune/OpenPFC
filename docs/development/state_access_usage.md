// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

# State Access and Storage Usage Guide

This guide provides usage examples for the state access and storage primitives in OpenPFC.

## Scalar Field Usage Examples

### Creating a FieldView for Read-Only Access

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <vector>

using namespace pfc::field;

// Create field data
std::vector<double> u_data(64, 1.0);

// Define geometry
pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

// Create read-only view
FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);

// Access data
const double* u_ptr = u_view.data();
std::size_t u_size = u_view.size();

// Query geometry
pfc::types::Int3 u_extents = u_view.extents();
pfc::types::Real3 u_spacing = u_view.spacing();
pfc::types::Real3 u_origin = u_view.origin();
```

### Creating FieldOutput for Mutable Access

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <vector>

using namespace pfc::field;

// Create output storage
std::vector<double> du_data(64, 0.0);

// Create mutable output
FieldOutput<double> du_output(du_data.data(), du_data.size());

// Access mutable data
double* du_ptr = du_output.data();
std::size_t du_size = du_output.size();

// Write to output
for (std::size_t i = 0; i < du_size; ++i) {
    du_ptr[i] = 0.1 * i;  // Write RHS values
}
```

### Validating Aliasing

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>
#include <vector>

using namespace pfc::field;

std::vector<double> input_data(64, 1.0);
std::vector<double> output_data(64, 0.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> input(input_data.data(), input_data.size(), extents, spacing, origin);
FieldOutput<double> output(output_data.data(), output_data.size());

// Validate no aliasing
output.validate_no_alias(input);  // Throws if aliasing detected

// Or use the free function
validate_no_alias(output, input);
```

### Validating Shape Compatibility

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>
#include <vector>

using namespace pfc::field;

std::vector<double> data1(64, 1.0);
std::vector<double> data2(64, 2.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> field1(data1.data(), data1.size(), extents, spacing, origin);
FieldView<double> field2(data2.data(), data2.size(), extents, spacing, origin);

// Validate shape compatibility
validate_shape_compatibility(field1, field2);  // Throws if incompatible

// Or use the member function
if (field1.is_compatible_with(field2)) {
    // Fields are compatible, can use together
}
```

## Multi-Field System Examples

### Creating a FieldBundle for Coupled Fields

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <vector>

using namespace pfc::field;

std::vector<double> u_data(64, 1.0);
std::vector<double> v_data(64, 2.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

// Create bundle
FieldBundle<FieldView<double>, FieldView<double>> fields(u_view, v_view);

// Access individual fields
const auto& u = fields.get<0>();
const auto& v = fields.get<1>();

// Validate shapes across all fields
if (fields.validate_shapes()) {
    // All fields have compatible shapes
}
```

### Multi-Field Output Bundle

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <vector>

using namespace pfc::field;

std::vector<double> du_data(64, 0.0);
std::vector<double> dv_data(64, 0.0);

FieldOutput<double> du_output(du_data.data(), du_data.size());
FieldOutput<double> dv_output(dv_data.data(), dv_data.size());

// Create output bundle
FieldBundle<FieldOutput<double>, FieldOutput<double>> outputs(du_output, dv_output);

// Access individual outputs
auto& du = outputs.get<0>();
auto& dv = outputs.get<1>();

// Write to outputs
for (std::size_t i = 0; i < du_output.size(); ++i) {
    du.data()[i] = 0.1;  // du value
    dv.data()[i] = 0.2;  // dv value
}
```

## Validation Usage

### Shape Compatibility Validation

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>
#include <vector>

using namespace pfc::field;

// Create two fields with compatible shapes
std::vector<double> u_data(64, 1.0);
std::vector<double> v_data(64, 2.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

// Validate shape compatibility
try {
    validate_shape_compatibility(u_view, v_view);
    // Fields are compatible
} catch (const std::invalid_argument& e) {
    // Handle incompatible shapes
    std::cerr << "Shape validation failed: " << e.what() << std::endl;
}
```

### Aliasing Validation

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>
#include <vector>

using namespace pfc::field;

std::vector<double> u_data(64, 1.0);
std::vector<double> du_data(64, 0.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
FieldOutput<double> du_output(du_data.data(), du_data.size());

// Validate no aliasing
try {
    du_output.validate_no_alias(u_view);
    // No aliasing, safe to proceed
} catch (const std::invalid_argument& e) {
    // Handle aliasing
    std::cerr << "Aliasing detected: " << e.what() << std::endl;
}
```

### Multi-Field Shape Validation

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <vector>

using namespace pfc::field;

std::vector<double> u_data(64, 1.0);
std::vector<double> v_data(64, 2.0);
std::vector<double> lap_data(64, 0.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
FieldView<double> lap_view(lap_data.data(), lap_data.size(), extents, spacing, origin);

// Create bundle
FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> fields(u_view, v_view, lap_view);

// Validate shapes across all fields
if (fields.validate_shapes()) {
    // All fields have compatible shapes
} else {
    // Some fields have incompatible shapes
}
```

## Workspace Usage

### Creating Integrator Workspace

```cpp
#include <openpfc/kernel/integrator/workspace.hpp>
#include <openpfc/types.hpp>

using namespace pfc::integrator;

// Define field extents
pfc::types::Int3 extents{4, 4, 4};

// Create workspace for RK4 (4 stages)
Workspace<double> workspace(extents, 4);

// Access stage storage
double* k1 = workspace.stage(0);
double* k2 = workspace.stage(1);
double* k3 = workspace.stage(2);
double* k4 = workspace.stage(3);

// Access scratch buffer
double* scratch = workspace.scratch();

// Clear all buffers
workspace.clear();
```

### Using Workspace in Time Stepping

```cpp
#include <openpfc/kernel/integrator/workspace.hpp>
#include <openpfc/types.hpp>
#include <vector>

using namespace pfc::integrator;

class RungeKutta4Integrator {
public:
    RungeKutta4Integrator(const pfc::types::Int3& extents)
        : m_workspace(extents, 4) {}

    void step(double dt) {
        // RK4 stages
        double* k1 = m_workspace.stage(0);
        double* k2 = m_workspace.stage(1);
        double* k3 = m_workspace.stage(2);
        double* k4 = m_workspace.stage(3);
        double* scratch = m_workspace.scratch();

        // Stage 1
        compute_rhs(k1);
        for (std::size_t i = 0; i < m_workspace.stage_size(); ++i) {
            scratch[i] = m_state[i] + 0.5 * dt * k1[i];
        }

        // Stage 2
        compute_rhs_from_state(k2, scratch);
        for (std::size_t i = 0; i < m_workspace.stage_size(); ++i) {
            scratch[i] = m_state[i] + 0.5 * dt * k2[i];
        }

        // Stage 3
        compute_rhs_from_state(k3, scratch);
        for (std::size_t i = 0; i < m_workspace.stage_size(); ++i) {
            scratch[i] = m_state[i] + dt * k3[i];
        }

        // Stage 4
        compute_rhs_from_state(k4, scratch);

        // Combine stages
        for (std::size_t i = 0; i < m_workspace.stage_size(); ++i) {
            m_state[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        }
    }

private:
    Workspace<double> m_workspace;
    std::vector<double> m_state;

    void compute_rhs(double* rhs);
    void compute_rhs_from_state(double* rhs, const double* state);
};
```

## Stage Context Usage

### Creating Stage Context

```cpp
#include <openpfc/kernel/integrator/stage_context.hpp>

using namespace pfc::integrator;

// Create stage context
StageContext ctx{
    .time = 0.0,           // Current evaluation time
    .dt = 0.01,            // Timestep being attempted
    .stage_index = 0,      // RK stage index
    .region_kind = StageContext::RegionKind::Interior,  // Interior access
    .needs_boundary_update = false,  // No BC update needed
    .needs_halo_exchange = true      // Halo exchange needed
};
```

### Using Stage Context for MPI Coordination

```cpp
#include <openpfc/kernel/integrator/stage_context.hpp>

using namespace pfc::integrator;

void driver_orchestration(const StageContext& ctx) {
    // Check if halo exchange is needed
    if (ctx.needs_halo_exchange) {
        // Start non-blocking halo exchange
        halo_exchanger.start_exchange(field_data);

        // Overlap with interior computation
        if (ctx.region_kind == StageContext::RegionKind::Interior ||
            ctx.region_kind == StageContext::RegionKind::All) {
            compute_interior(field_data, scratch);
        }

        // Finish halo exchange
        halo_exchanger.finish_exchange(field_data);
    }

    // Check if boundary update is needed
    if (ctx.needs_boundary_update) {
        if (ctx.region_kind == StageContext::RegionKind::Boundary ||
            ctx.region_kind == StageContext::RegionKind::All) {
            apply_boundary_conditions(field_data, ctx.time);
        }
    }
}
```

## Backend-Agnostic Usage

### Same API Works for CPU and GPU

```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <vector>

using namespace pfc::field;

// CPU storage
std::vector<double> cpu_data(64, 1.0);

// GPU storage (when implemented)
// pfc::gpu::GPUVector<double> gpu_data(64, 1.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

// Same API for both backends
FieldView<double> cpu_view(cpu_data.data(), cpu_data.size(), extents, spacing, origin);
// FieldView<double> gpu_view(gpu_data.data(), gpu_data.size(), extents, spacing, origin);

// Same operations work for both
std::size_t size = cpu_view.size();
pfc::types::Int3 ext = cpu_view.extents();
pfc::types::Real3 sp = cpu_view.spacing();
pfc::types::Real3 orig = cpu_view.origin();
```

### Backend-Specific Validation (Deferred)

```cpp
// Backend-specific validation is deferred to separate headers
// when GPU implementation is pursued.

// For example, in include/openpfc/kernel/field/validation_cuda.hpp:
/*
namespace pfc::field {

template<typename T>
void validate_backend_compatibility(const FieldView<T>& field1,
                                     const FieldView<T>& field2) {
    // CUDA-specific backend checks
    // Throws if mixing CPU and GPU storage
}

} // namespace pfc::field
*/
```

## Migration Guide

### Migrating from LocalField to FieldView

**Old pattern (LocalField)**:
```cpp
#include <openpfc/kernel/field/local_field.hpp>

using namespace pfc::field;

LocalField<double> u = LocalField<double>::from_subdomain(decomp, rank, halo_width);
const double* u_data = u.data();
std::size_t u_size = u.size();
pfc::types::Int3 u_extents = u.size3();
pfc::types::Real3 u_spacing = u.spacing();
pfc::types::Real3 u_origin = u.origin();
```

**New pattern (FieldView)**:
```cpp
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/state_access.hpp>

using namespace pfc::field;

// Create LocalField as before
LocalField<double> u_local = LocalField<double>::from_subdomain(decomp, rank, halo_width);

// Create FieldView from LocalField
FieldView<double> u_view(u_local.data(), u_local.size(), 
                         u_local.size3(), u_local.spacing(), u_local.origin());

// Access data and geometry
const double* u_data = u_view.data();
std::size_t u_size = u_view.size();
pfc::types::Int3 u_extents = u_view.extents();
pfc::types::Real3 u_spacing = u_view.spacing();
pfc::types::Real3 u_origin = u_view.origin();
```

### Migrating from WaveIncrements to FieldBundle

**Old pattern (WaveIncrements)**:
```cpp
struct WaveIncrements {
    double du = 0.0;
    double dv = 0.0;
    auto as_tuple() { return std::tie(du, dv); }
};

WaveIncrements increments = model.rhs(t, v, lap);
auto [du, dv] = increments.as_tuple();
```

**New pattern (FieldBundle)**:
```cpp
#include <openpfc/kernel/field/state_access.hpp>

using namespace pfc::field;

std::vector<double> du_data(64, 0.0);
std::vector<double> dv_data(64, 0.0);

FieldOutput<double> du_output(du_data.data(), du_data.size());
FieldOutput<double> dv_output(dv_data.data(), dv_data.size());

FieldBundle<FieldOutput<double>, FieldOutput<double>> outputs(du_output, dv_output);

// Access outputs
auto& du = outputs.get<0>();
auto& dv = outputs.get<1>();

// Write to outputs
for (std::size_t i = 0; i < du_output.size(); ++i) {
    du.data()[i] = du_value;
    dv.data()[i] = dv_value;
}
```

## Common Error Patterns and Resolution

### Error: Incompatible Shapes

```cpp
// Error: fields have different extents
std::vector<double> u_data(64, 1.0);
std::vector<double> v_data(128, 2.0);

pfc::types::Int3 extents1{4, 4, 4};
pfc::types::Int3 extents2{8, 4, 4};

FieldView<double> u_view(u_data.data(), u_data.size(), extents1, spacing, origin);
FieldView<double> v_view(v_data.data(), v_data.size(), extents2, spacing, origin);

validate_shape_compatibility(u_view, v_view);  // Throws std::invalid_argument

// Resolution: Ensure fields have matching geometry
```

### Error: Aliasing Detected

```cpp
// Error: output storage aliases input storage
std::vector<double> u_data(64, 1.0);

FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
FieldOutput<double> u_output(u_data.data(), u_data.size());

u_output.validate_no_alias(u_view);  // Throws std::invalid_argument

// Resolution: Use separate storage for output
std::vector<double> du_data(64, 0.0);
FieldOutput<double> du_output(du_data.data(), du_data.size());
du_output.validate_no_alias(u_view);  // OK
```

### Error: Backend Mismatch (Future)

```cpp
// Error: mixing CPU and GPU storage (when GPU is implemented)
std::vector<double> cpu_data(64, 1.0);
pfc::gpu::GPUVector<double> gpu_data(64, 1.0);

FieldView<double> cpu_view(cpu_data.data(), cpu_data.size(), extents, spacing, origin);
FieldView<double> gpu_view(gpu_data.data(), gpu_data.size(), extents, spacing, origin);

validate_backend_compatibility(cpu_view, gpu_view);  // Throws std::invalid_argument

// Resolution: Use storage from same backend
```

## Backend-Specific Validation (Deferred)

Backend-specific validation is deferred to separate headers when GPU implementation is pursued:

```cpp
// Future: include/openpfc/kernel/field/validation_cuda.hpp
namespace pfc::field {

template<typename T>
void validate_backend_compatibility(const FieldView<T>& field1,
                                     const FieldView<T>& field2) {
    // CUDA-specific backend checks
    // Throws if mixing CPU and GPU storage
}

} // namespace pfc::field

// Future: include/openpfc/kernel/field/validation_hip.hpp
namespace pfc::field {

template<typename T>
void validate_backend_compatibility(const FieldView<T>& field1,
                                     const FieldView<T>& field2) {
    // HIP-specific backend checks
    // Throws if mixing CPU and GPU storage
}

} // namespace pfc::field
```

Note: FieldView<T> is intentionally backend-agnostic. Backend-specific validation is provided in separate headers to avoid contaminating generic code with backend-specific dependencies.
