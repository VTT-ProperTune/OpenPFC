// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

# State Access and Storage Usage Guide

This guide provides examples and patterns for using the state access primitives, workspace management, and validation utilities in OpenPFC.

## Scalar Field Usage

### Creating FieldView for Input

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

// Access field data (read-only)
const double* data = u_view.data();
std::size_t size = u_view.size();
pfc::types::Int3 ext = u_view.extents();
pfc::types::Real3 sp = u_view.spacing();
pfc::types::Real3 orig = u_view.origin();
```

### Creating FieldOutput for Results

```cpp
#include <openpfc/kernel/field/state_access.hpp>

using namespace pfc::field;

// Create output storage
std::vector<double> du_data(64, 0.0);

// Create mutable output view
FieldOutput<double> du_output(du_data.data(), du_data.size());

// Write results
double* du_data_ptr = du_output.data();
std::size_t du_size = du_output.size();

// Example: write RHS computation
for (std::size_t i = 0; i < du_size; ++i) {
    du_data_ptr[i] = /* RHS computation */;
}
```

### Validation Usage

```cpp
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;

// Validate shape compatibility
FieldView<double> u_view1(/* ... */);
FieldView<double> u_view2(/* ... */);

try {
    validate_shape_compatibility(u_view1, u_view2);
    // Fields are compatible, proceed with computation
} catch (const std::invalid_argument& e) {
    // Handle incompatibility
    std::cerr << "Shape error: " << e.what() << std::endl;
}

// Validate no aliasing
FieldOutput<double> du_output(/* ... */);

try {
    du_output.validate_no_alias(u_view1);
    // No aliasing, safe to proceed
} catch (const std::invalid_argument& e) {
    // Handle aliasing
    std::cerr << "Aliasing error: " << e.what() << std::endl;
}
```

## Multi-Field System Usage

### Creating FieldBundle

```cpp
#include <openpfc/kernel/field/state_access.hpp>

using namespace pfc::field;

// Create multiple fields with same geometry
std::vector<double> u_data(64, 1.0);
std::vector<double> v_data(64, 2.0);

pfc::types::Int3 extents{4, 4, 4};
pfc::types::Real3 spacing{1.0, 1.0, 1.0};
pfc::types::Real3 origin{0.0, 0.0, 0.0};

FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

// Create multi-field bundle
FieldBundle<FieldView<double>, FieldView<double>> wave_bundle(u_view, v_view);

// Access individual fields
auto& u = wave_bundle.get<0>();
auto& v = wave_bundle.get<1>();

// Validate shapes across bundle
if (wave_bundle.validate_shapes()) {
    // All fields have compatible shapes
}
```

### Coordinated Validation

```cpp
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;

// Validate all fields in bundle are compatible
FieldBundle<FieldView<double>, FieldView<double>> bundle(/* ... */);

if (bundle.validate_shapes()) {
    // Safe to use fields together in computation
} else {
    // Handle shape incompatibility
}

// Validate cross-field operations
FieldView<double> u_view = bundle.get<0>();
FieldView<double> v_view = bundle.get<1>();

validate_shape_compatibility(u_view, v_view);  // Throws if incompatible
```

## Workspace Usage

### Creating Workspace for Integrator

```cpp
#include <openpfc/kernel/integrator/workspace.hpp>

using namespace pfc::integrator;

// Define field geometry
pfc::types::Int3 extents{32, 32, 32};
std::size_t num_stages = 4;  // For RK4

// Create workspace
Workspace<double> workspace(extents, num_stages);

// Access stage storage
double* stage0 = workspace.stage(0);
double* stage1 = workspace.stage(1);
double* stage2 = workspace.stage(2);
double* stage3 = workspace.stage(3);

// Access scratch buffer
double* scratch = workspace.scratch();

// Clear all buffers
workspace.clear();
```

### Workspace Lifetime Management

```cpp
class ExplicitEulerIntegrator {
public:
    ExplicitEulerIntegrator(const pfc::types::Int3& extents)
        : m_workspace(extents, 1)  // 1 stage for Euler
    {}

    void step(FieldView<double> u_view, FieldOutput<double> du_output, double dt) {
        // Validate no aliasing
        du_output.validate_no_alias(u_view);

        // Use workspace for intermediate computations
        double* scratch = m_workspace.scratch();

        // Compute RHS using scratch
        compute_rhs(u_view, scratch);

        // Write result to output
        for (std::size_t i = 0; i < du_output.size(); ++i) {
            du_output.data()[i] = scratch[i];
        }
    }

private:
    Workspace<double> m_workspace;
};
```

## Stage Context Usage

### Creating Stage Context for MPI Coordination

```cpp
#include <openpfc/kernel/integrator/stage_context.hpp>

using namespace pfc::integrator;

// Create stage context
StageContext ctx;
ctx.time = 0.0;  // Current simulation time
ctx.dt = 0.01;   // Timestep being attempted
ctx.stage_index = 0;  // RK stage index
ctx.region_kind = StageContext::RegionKind::All;  // Field region needed
ctx.needs_boundary_update = true;  // BCs need updating
ctx.needs_halo_exchange = true;  // Halo exchange needed

// Driver uses context to coordinate MPI
if (ctx.needs_halo_exchange) {
    exchange_halos(field);
}

if (ctx.needs_boundary_update) {
    apply_boundary_conditions(field);
}
```

### Stage Context in Integrator

```cpp
class RK4Integrator {
public:
    StepAttempt step(FieldView<double> u_view, double t, double dt) {
        StepAttempt result;

        // Stage 1
        StageContext ctx1;
        ctx1.time = t;
        ctx1.dt = dt;
        ctx1.stage_index = 0;
        ctx1.region_kind = StageContext::RegionKind::All;
        ctx1.needs_halo_exchange = true;

        // Driver coordinates MPI based on ctx1
        // ... compute k1 ...

        // Stage 2
        StageContext ctx2;
        ctx2.time = t + dt / 2.0;
        ctx2.dt = dt;
        ctx2.stage_index = 1;
        ctx2.region_kind = StageContext::RegionKind::Interior;
        ctx2.needs_halo_exchange = false;

        // ... compute k2 ...

        return result;
    }
};
```

## Backend-Agnostic Usage

### Same API for CPU and GPU

```cpp
#include <openpfc/kernel/field/state_access.hpp>

using namespace pfc::field;

// CPU storage
std::vector<double> cpu_data(64, 1.0);

// GPU storage (when implemented)
// pfc::gpu::GPUVector<double> gpu_data(64, 1.0);

// Define geometry
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

### Backend Memory-Space Compatibility Validation

```cpp
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;

// Define backend tags
struct CPUBackendTag {};
struct CUDABackendTag {};
struct HIPBackendTag {};

// CPU fields are compatible with each other
FieldView<double> cpu_field1(/* ... */);
FieldView<double> cpu_field2(/* ... */);

validate_backend_compatibility<double, CPUBackendTag, CPUBackendTag>(
    cpu_field1, cpu_field2);  // OK: same backend

// Mixing CPU and GPU fields is not allowed (compile-time error)
// FieldView<double> gpu_field(/* ... */);
// validate_backend_compatibility<double, CPUBackendTag, CUDABackendTag>(
//     cpu_field1, gpu_field);  // Static assertion failure

// CUDA fields are compatible with each other
// FieldView<double> cuda_field1(/* ... */);
// FieldView<double> cuda_field2(/* ... */);
// validate_backend_compatibility<double, CUDABackendTag, CUDABackendTag>(
//     cuda_field1, cuda_field2);  // OK: same backend
```

### Backend-Specific Validation (Future)

When GPU backends are implemented, backend-specific validation can be added in separate headers:

```cpp
// Future: include/openpfc/kernel/field/validation_cuda.hpp
namespace pfc::field {

// Specialization for CUDA backend
template<typename T>
void validate_backend_compatibility<const FieldView<T>&,
                                     CUDABackendTag,
                                     CUDABackendTag>(const FieldView<T>& field1,
                                                      const FieldView<T>& field2) {
    // CUDA-specific backend checks
    // Verify both fields are on the same device
    // Verify compatible CUDA streams
    // etc.
}

} // namespace pfc::field
```

Note: FieldView<T> is intentionally backend-agnostic. Backend-specific validation is provided through compile-time backend tags and can be extended with backend-specific implementations in separate headers.

## Migration Guide

### Migrating from LocalField to FieldView

**Old pattern (LocalField)**:
```cpp
#include <openpfc/field/local_field.hpp>

using namespace pfc;

LocalField<double> u = LocalField<double>::from_subdomain(decomp, rank, halo_width);
const double* u_data = u.data();
std::size_t u_size = u.size();
Int3 u_size3 = u.size3();
Real3 u_spacing = u.spacing();
Real3 u_origin = u.origin();

// Use u_data for computation
```

**New pattern (FieldView)**:
```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/field/local_field.hpp>

using namespace pfc;
using namespace pfc::field;

// Create LocalField as before
LocalField<double> u_local = LocalField<double>::from_subdomain(decomp, rank, halo_width);

// Create backend-agnostic view
FieldView<double> u_view(u_local.data(), u_local.size(), u_local.size3(),
                         u_local.spacing(), u_local.origin());

// Use view for computation (same API)
const double* u_data = u_view.data();
std::size_t u_size = u_view.size();
pfc::types::Int3 u_size3 = u_view.extents();
pfc::types::Real3 u_spacing = u_view.spacing();
pfc::types::Real3 u_origin = u_view.origin();
```

**Benefits of migration**:
1. Backend-agnostic view (works with CPU and GPU storage)
2. Explicit read-only semantics (const access)
3. Shape validation via `is_compatible_with()`
4. Aliasing detection for output storage

### Migrating from WaveIncrements to FieldBundle

**Old pattern (WaveIncrements)**:
```cpp
struct WaveIncrements {
    PaddedBrick<double>& du;
    PaddedBrick<double>& dv;
};

WaveIncrements increments{du, dv};

// Access via members
increments.du(i, j, k) = /* ... */;
increments.dv(i, j, k) = /* ... */;
```

**New pattern (FieldBundle)**:
```cpp
#include <openpfc/kernel/field/state_access.hpp>

using namespace pfc::field;

// Create views for each field
FieldView<double> du_view(/* ... */);
FieldView<double> dv_view(/* ... */);

// Create bundle
FieldBundle<FieldView<double>, FieldView<double>> wave_bundle(du_view, dv_view);

// Access via get<I>()
auto& du = wave_bundle.get<0>();
auto& dv = wave_bundle.get<1>();

// Validate shapes across bundle
if (!wave_bundle.validate_shapes()) {
    throw std::invalid_argument("Wave fields have incompatible shapes");
}
```

**Benefits of migration**:
1. Coordinated access to multiple fields
2. Shape validation across all fields in bundle
3. Type-safe indexed access via `get<I>()`
4. Backend-agnostic views for each field

### Migrating from PaddedBrick to FieldView

**Old pattern (PaddedBrick)**:
```cpp
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc::field;

PaddedBrick<double> u(decomp, rank, halo_width);

// Access via for_each_owned
for_each_owned(u, [&](int i, int j, int k) {
    double val = u(i, j, k);
    // ... computation ...
});
```

**New pattern (FieldView)**:
```cpp
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc::field;

PaddedBrick<double> u_padded(decomp, rank, halo_width);

// Extract owned region
std::vector<double> u_owned(u_padded.owned_size());
extract_owned_region(u_padded, u_owned.data());

// Create view
FieldView<double> u_view(u_owned.data(), u_owned.size(),
                         u_padded.owned_extents(),
                         u_padded.spacing(),
                         u_padded.origin());

// Access via direct indexing
for (std::size_t i = 0; i < u_view.size(); ++i) {
    double val = u_view.data()[i];
    // ... computation ...
}
```

**Benefits of migration**:
1. Backend-agnostic view
2. Explicit geometry metadata
3. Compatibility with validation utilities
4. Works with any contiguous storage

## Common Error Patterns and Resolution

### Error: Incompatible Shapes

```cpp
FieldView<double> field1(/* extents: {4, 4, 4} */);
FieldView<double> field2(/* extents: {8, 4, 4} */);

try {
    validate_shape_compatibility(field1, field2);
} catch (const std::invalid_argument& e) {
    // Error: fields have incompatible shapes
    // Resolution: Ensure fields have matching extents, spacing, and origin
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### Error: Aliased Storage

```cpp
std::vector<double> data(64, 0.0);

FieldView<double> input_view(data.data(), data.size(), /* ... */);
FieldOutput<double> output_alias(data.data(), data.size());  // Same storage!

try {
    output_alias.validate_no_alias(input_view);
} catch (const std::invalid_argument& e) {
    // Error: output storage aliases input storage
    // Resolution: Use separate storage for output
    std::vector<double> output_data(64, 0.0);
    FieldOutput<double> output_distinct(output_data.data(), output_data.size());
}
```

### Error: Backend Mismatch

```cpp
struct CPUBackendTag {};
struct CUDABackendTag {};

FieldView<double> cpu_field(/* ... */);
// FieldView<double> gpu_field(/* ... */);

// Compile-time error: backend mismatch
// validate_backend_compatibility<double, CPUBackendTag, CUDABackendTag>(
//     cpu_field, gpu_field);

// Resolution: Use fields from same backend
FieldView<double> cpu_field2(/* ... */);
validate_backend_compatibility<double, CPUBackendTag, CPUBackendTag>(
    cpu_field, cpu_field2);  // OK
```

## Verification Summary

This section summarizes how verification is performed for each semantic requirement:

| Requirement | Verification Method | Example |
|-------------|-------------------|---------|
| Const input semantics | FieldView<T>::data() const | `const double* data = view.data();` |
| Mutable output semantics | FieldOutput<T>::data() non-const | `double* data = output.data();` |
| Aliasing detection | validate_no_alias() pointer comparison | `output.validate_no_alias(input);` |
| Shape compatibility | validate_shape_compatibility() structural check | `validate_shape_compatibility(f1, f2);` |
| Backend compatibility | validate_backend_compatibility() static_assert | `validate_backend_compatibility<double, CPU, CPU>(f1, f2);` |
| Workspace ownership | Workspace<T> private members | `Workspace<double> ws(extents, stages);` |
| Stage context coordination | StageContext struct fields | `ctx.needs_halo_exchange = true;` |

All verification is performed at compile time (static_assert, type system) or at binding time (validation functions called once during construction/binding), ensuring no per-evaluation overhead.
