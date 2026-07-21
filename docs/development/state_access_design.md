// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

# State Access and Storage Design

This document describes the design rationale for evidence-driven state access primitives, workspace management, and validation utilities in OpenPFC.

## Rationale for Value-Semantic Approach

The implementation uses value-semantic types (FieldView<T>, FieldOutput<T>, FieldBundle<Ts...>) rather than type erasure or virtual interfaces for the following reasons:

### No Virtual Dispatch

- **Performance**: Virtual function calls have overhead that matters in tight computational loops
- **Inlinability**: Value-semantic types can be inlined by the compiler
- **Predictability**: No runtime indirection, easier to reason about performance

### No Heap Allocation

- **Efficiency**: FieldView and FieldOutput are thin wrappers around pointers
- **Cache locality**: Better memory access patterns
- **No ownership complexity**: Clear ownership semantics (caller owns storage)

### Copyable and Moveable

- **Pass by value**: Natural C++ semantics for small types
- **Standard containers**: Can be stored in std::vector, std::tuple, etc.
- **Template-friendly**: Works well with generic programming

### Evidence from Existing Code

Heat3D and Wave2D both use value-semantic patterns:

```cpp
// Heat3D: PaddedBrick is value-semantic
PaddedBrick<double> u(decomp, rank, halo_width);
PaddedBrick<double> du(decomp, rank, halo_width);
// Direct member access, no virtual functions
for_each_owned(du, [&](auto idx) { du[idx] = /* RHS */; });

// Wave2D: Multiple PaddedBrick instances
PaddedBrick<double> u(decomp, rank, halo_width);
PaddedBrick<double> v(decomp, rank, halo_width);
// Tuple-like access patterns
```

## Evidence from Existing Patterns

### Heat3D Scalar Field Pattern

**Source**: `apps/heat3d/src/cpu/heat3d_fd.cpp`

**Pattern**:
- Single scalar field u (temperature)
- PaddedBrick<double> for state storage
- FDGradient<HeatGrads> for Laplacian evaluation
- Explicit Euler time integration: u += dt * du

**Key observations**:
- Contiguous storage with halo padding
- Geometry metadata (size, spacing, origin)
- Read-only access to input state
- Separate output storage for RHS
- Clear separation between state and workspace

**Design implications**:
- FieldView<T> provides const access to field data and geometry
- FieldOutput<T> provides mutable output storage
- Workspace<T> owns scratch buffers for intermediate computations

### Wave2D Multi-Field Pattern

**Source**: `apps/wave2d/src/cpu/wave2d_fd.cpp`

**Pattern**:
- Multiple fields: u (displacement), v (velocity), lap (Laplacian)
- PaddedBrick<double> for each field
- Coupled time integration: u += dt * v, v += dt * k^2 * lap
- Coordinated halo exchange across fields

**Key observations**:
- Multiple fields with same geometry
- Tuple-like access patterns (u, v, lap)
- Shape compatibility validation required
- Coordinated validation across field bundle

**Design implications**:
- FieldBundle<Ts...> groups multiple fields
- validate_shape_compatibility() ensures geometric compatibility
- Coordinated access via get<I>() interface

### Complex Field Usage

**Source**: Spectral method implementations

**Pattern**:
- ComplexField = std::vector<std::complex<double>>
- Same access patterns as real fields
- Used in Fourier space computations

**Key observations**:
- No code duplication for complex vs real
- Same geometric queries
- Same read/write semantics

**Design implications**:
- Template types parameterized by T (double or std::complex<double>)
- Single implementation works for both types

### Backend Differences

**Source**: CPU vs GPU implementations

**CPU pattern**:
```cpp
std::vector<double> m_data;
double* data() noexcept { return m_data.data(); }  // Host pointer
```

**GPU pattern** (when implemented):
```cpp
pfc::gpu::GPUVector<double> m_data;
double* data() noexcept { return m_data.data(); }  // Device pointer
```

**Evidence**:
- Both backends provide `data()` and `size()` methods
- Geometry metadata is backend-agnostic
- `OPENPFC_HD` macro for host/device callable annotations

**Design implications**:
- `FieldView<T>` is backend-agnostic (const T* + geometry)
- Backend-specific validation uses compile-time backend tags
- No shared abstraction layer required
- Backend allocation APIs permitted

## MPI Coordination Patterns

**Source**: Existing drivers with halo exchange

**Pattern**: Non-blocking halo exchanges with timing control
```cpp
for (int step = 0; step < n_steps; ++step) {
    exchange_halos(u);  // MPI halo exchange
    for_each_owned(u, [&](auto idx) {
        du[idx] = /* RHS using halo data */;
    });
    u += dt * du;
}
```

**Key observations**:
- Halo exchange before evaluation
- Driver orchestrates timing
- Integrator doesn't manage MPI

**Design implications**:
- `pfc::integrator::StageContext` carries timing and region requirements
- Driver reads context and schedules MPI operations
- Integrator focuses on algorithm, not communication

## Workspace vs StageWorkspace

Two similarly named workspace types exist; they must not be conflated:

| | `pfc::integrator::Workspace<T>` | `pfc::sim::steppers::StageWorkspace<T>` |
|---|---|---|
| Header | `include/openpfc/kernel/integrator/workspace.hpp` | `include/openpfc/kernel/simulation/steppers/stage_workspace.hpp` |
| Constructor | `(extents, num_stages)` | `(num_stages, local_size)` |
| Reclaim | `clear()` | `reset()` |
| Role in this slice | Integrator-owned stage + scratch for the state-access contract | Existing stepper helper under `kernel/simulation/steppers/`; out of scope here |

This design slice specifies and tests only `pfc::integrator::Workspace<T>`. Do not edit or merge `StageWorkspace`.

## Integrator StageContext vs solver StageContext

Two different `StageContext` types share a short name:

| | `pfc::integrator::StageContext` | `pfc::sim::StageContext` |
|---|---|---|
| Header | `include/openpfc/kernel/integrator/stage_context.hpp` | `include/openpfc/kernel/simulation/solver_contract.hpp` |
| Fields | `time`, `dt`, `stage_index`, `region_kind`, `needs_boundary_update`, `needs_halo_exchange` | `evaluation_time`, `ExecutionService& execution_service` |
| Role | MPI / BC coordination flags from integrators to drivers | Solver evaluation context with execution service |

Always qualify as `pfc::integrator::StageContext` in this slice. Do not merge or replace the solver_contract type.

## Migration examples

### Scalar path (Heat3D-style)

```cpp
FieldView<double> u_view(u.data(), u.size(), extents, spacing, origin);
FieldOutput<double> du_out(du.data(), du.size());
du_out.validate_no_alias(u_view);  // distinct RHS buffer
// heat3d::HeatOperator::evaluate(u_view, du_out, ...);
// In-place Euler may use LocalField: u += dt * du (ScaledField; bypasses validate_no_alias)
```

Evidence: `tests/integration/scenarios/field_operations/test_heat3d_state_access.cpp`
(`Heat3D numerical equivalence test`, `Heat3D time integration pattern`,
`Heat3D migration path from LocalField to FieldView`).

### Multi-field path (Wave2D-style)

```cpp
FieldBundle<FieldView<double>, FieldView<double>> wave(u_view, v_view);
REQUIRE(wave.validate_shapes());
auto& u = wave.get<0>();
auto& v = wave.get<1>();
// wave2d::WaveOperatorResult::as_tuple() scatters coupled (du, dv) outputs
```

Evidence: `tests/integration/scenarios/field_operations/test_wave2d_state_access.cpp`
(`Wave2D multi-field bundle pattern`, `Wave2D coupled time integration pattern`,
`Wave2D migration path from multi-field to FieldBundle`).

### Workspace and MPI coordination

- Integrators own `pfc::integrator::Workspace<T>`; models never access it.
- Drivers read `pfc::integrator::StageContext` flags to schedule halo/BC work.

## Why FieldView is Backend-Agnostic

`FieldView<T>` is intentionally backend-agnostic, holding only:
- `const T* m_data`: Pointer to field data (any contiguous storage)
- `std::size_t m_size`: Number of elements
- `pfc::types::Int3 m_extents`: Grid dimensions
- `pfc::types::Real3 m_spacing`: Physical spacing
- `pfc::types::Real3 m_origin`: Physical origin

This design enables:

1. **Single header usage**: Same `FieldView<T>` works for CPU and GPU backends
2. **No backend-specific code**: Generic algorithms work with any backend
3. **Clear separation**: Backend details isolated in allocation/source code
4. **Future extensibility**: New backends (e.g., SYCL) can use same interface

## Backend Compatibility Enforcement Strategy

Backend memory-space compatibility validation uses compile-time backend tag checks:

### Current Implementation

```cpp
// include/openpfc/kernel/field/validation.hpp
template<typename T, typename BackendTag1, typename BackendTag2>
void validate_backend_compatibility(const FieldView<T>& field1,
                                     const FieldView<T>& field2) {
    // Compile-time check: require same backend tag type
    static_assert(std::is_same_v<BackendTag1, BackendTag2>,
                  "validate_backend_compatibility: fields must use the same backend memory space");
    // FieldView parameters allow backend-specific implementations to access metadata
    (void)field1;
    (void)field2;
}
```

### Usage Pattern

```cpp
struct CPUBackendTag {};
struct CUDABackendTag {};

// CPU fields are compatible with each other
validate_backend_compatibility<double, CPUBackendTag, CPUBackendTag>(field1, field2); // OK

// Mixing CPU and GPU fields is not allowed (compile-time error)
validate_backend_compatibility<double, CPUBackendTag, CUDABackendTag>(field1, field2); // Static assertion failure
```

### Why Backend Tags Are Template Parameters

1. **FieldView<T> is backend-agnostic**:
   - `FieldView<T>` holds only `const T*` and geometry
   - Backend type information is not available from FieldView alone
   - Backend tags must be provided by calling code

2. **Compile-time enforcement**:
   - Static assertion catches backend mismatches at compile time
   - No runtime overhead for backend validation
   - Clear error messages at compile time

3. **Extensibility for GPU implementations**:
   - Backend-specific headers can specialize or override the default implementation
   - Specializations can add runtime checks (e.g., CUDA device ID compatibility)
   - The generic interface remains the same across backends

### Future GPU Implementation Path

When GPU backends are implemented, backend-specific validation can be added:

```cpp
// include/openpfc/kernel/field/validation_cuda.hpp (future)
namespace pfc::field {

template<typename T>
void validate_backend_compatibility<const FieldView<T>&,
                                     CUDABackendTag,
                                     CUDABackendTag>(const FieldView<T>& field1,
                                                      const FieldView<T>& field2) {
    // CUDA-specific checks
    // Verify both fields are on the same device
    // Verify compatible CUDA streams
    // etc.
}

} // namespace pfc::field
```

This approach maintains:
- Semantic equivalence across backends
- Zero overhead for CPU-only code
- Extensibility for backend-specific checks
- No virtual dispatch required

## Backend Extensibility Strategy

The design enables CUDA/HIP implementation without changing semantic contracts:

### No Virtual Dispatch

- Value-semantic types work with any backend
- Compiler can inline and optimize
- No runtime indirection

### No Return-by-Value Allocations

- FieldView and FieldOutput are non-owning views
- No heap allocation in accessors
- Caller owns all storage

### Device Pointer/Copy APIs Not Required

- Backend-specific allocation handled at storage level
- FieldView works with any contiguous pointer
- No host/device copy APIs in semantic layer

### Backend-Specific Headers Allowed

- validation_cuda.hpp, validation_hip.hpp can provide backend-specific validation
- No changes to generic interface required
- Backend code isolated in separate headers

### Example CUDA Path

```cpp
// GPU storage allocation (backend-specific)
pfc::gpu::GPUVector<double> u_gpu(size);

// Create backend-agnostic view
FieldView<double> u_view(u_gpu.data(), u_gpu.size(), extents, spacing, origin);

// Use in generic algorithm (same as CPU)
auto laplacian = compute_laplacian(u_view);  // Works on GPU!

// Backend-specific validation (when needed)
validate_backend_compatibility<double, CUDABackendTag, CUDABackendTag>(
    u_view, v_view);  // Compile-time check
```

## Non-Scope Items

The following items are explicitly out of scope for this design slice:

- Virtual `TimeIntegrator` hierarchy: Not required for value-semantic approach
- `FieldView` class as virtual interface: Implemented as concrete value-semantic type
- Return-by-value field moves: FieldView/FieldOutput are non-owning views
- Device pointer/copy APIs: Backend-specific, not in semantic layer
- Runtime string-based factory methods: Not needed for value-semantic types
- Single unified container: Different types for different purposes is acceptable
- One fixed layout: Backend-specific layouts permitted
- Preservation of `GPUVector` or other legacy symbols: Evidence only, can be refactored
- Integration with driver orchestration: Covered in later work items
- Operator evaluation seam implementation: Covered in later work items
- Method invocation seam: Covered in later work items
- Step-attempt semantic implementation: Covered in later work items

## Acceptance Criteria Verification

Mapped to the current work-item acceptance criteria (not legacy numeric ids):

- `FieldView<T>` provides const access to field data and geometry for operator inputs (`state_access.hpp`; unit case `FieldView const access`).
- `FieldOutput<T>` provides mutable caller-owned output storage with `validate_no_alias` (`state_access.hpp`; unit cases `FieldOutput mutable access`, `Field aliasing detection`).
- `FieldBundle<Fields...>` groups fields with `get<I>()` and `validate_shapes()` (`state_access.hpp`; unit case `FieldBundle multi-field`).
- `pfc::integrator::Workspace<T>` provides integrator-owned stage and scratch storage (unit cases `Workspace stage storage and scratch`, `Workspace clear resets buffers`).
- `pfc::integrator::StageContext` carries `time`, `dt`, `stage_index`, `region_kind`, `needs_boundary_update`, `needs_halo_exchange` (unit case `StageContext MPI coordination fields`).
- Validation free functions cover shape, aliasing, and backend-tag checks (`validation.hpp`; unit cases `Shape compatibility validation`, `Backend compatibility validation`). Backend mismatch is a compile-time `static_assert`, not a runtime throw.
- Unit tests in `tests/unit/kernel/field/test_state_access.cpp` exercise the contracts without CUDA/HIP dependencies (including the documented ScaledField in-place exception).
- Integration evidence: `test_heat3d_state_access.cpp` (scalar) and `test_wave2d_state_access.cpp` (multi-field).
- This document covers value semantics, backend compatibility, validation strategy, Workspace vs StageWorkspace, and integrator vs solver StageContext distinctions, plus migration examples above.

## Summary

The value-semantic approach provides:

1. **Clear contracts**: Const views for input, mutable outputs for results
2. **No overhead**: Thin wrappers around pointers, no virtual dispatch
3. **Backend extensibility**: Same interface works for CPU and GPU
4. **Type safety**: Compile-time checks for shape and backend compatibility
5. **Evidence-based**: Derived from existing Heat3D and Wave2D patterns

This design enables physics-model-independent time integration while maintaining backend neutrality and performance.
