// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

# State Access and Storage Design

This document describes the design rationale for the evidence-driven state access and storage implementation in OpenPFC.

## Rationale for Value-Semantic Approach

The implementation uses value-semantic types (`FieldView<T>`, `FieldOutput<T>`, `FieldBundle<Ts...>`) rather than type erasure or virtual interfaces. This choice is justified by:

### Why Not Virtual Interfaces

- **No runtime overhead**: Virtual function calls add indirection and prevent inlining
- **Predictable performance**: Compiler can optimize value-semantic types more aggressively
- **Simple ownership**: Clear copy/move semantics without shared_ptr complexity
- **Header-only implementation**: No vtable or RTTI requirements

### Why Not Type Erasure

- **No heap allocation**: Value-semantic types can be stack-allocated
- **Type safety**: Template parameters catch errors at compile time
- **Debuggability**: Easier to inspect and debug concrete types
- **No type erasure overhead**: Avoids std::function-style small buffer optimization

### Why Value Semantics

- **Natural C++ idiom**: Familiar to C++ developers
- **Copyable and movable**: Easy to pass by value or reference
- **No hidden allocation**: All storage is explicit and caller-provided
- **Const correctness**: Clear distinction between const views and mutable outputs

## Evidence from Existing Patterns

### Heat3D Scalar Field Patterns

**Source**: `apps/heat3d/src/cpu/heat3d_fd.cpp`

**Pattern**: Single scalar field `u` with PaddedBrick storage
```cpp
pfc::field::PaddedBrick<double> u(decomp, rank, hw);
const double* u_data = u.data();
std::size_t u_size = u.size();
```

**Evidence**:
- Direct data access via `data()` and `size()`
- Geometry metadata (extents, spacing, origin) for coordinate transforms
- Separate halo storage managed by `PaddedHaloExchanger`
- In-place update: `u += dt * du` via `ScaledField`

**Design Implications**:
- `FieldView<T>` provides const access to data and geometry
- `FieldOutput<T>` provides mutable access for RHS computation
- Aliasing validation needed for in-place operations
- Backend-agnostic design allows CPU and GPU storage

### Wave2D Multi-Field Patterns

**Source**: `apps/wave2d/src/cpu/wave2d_fd.cpp`

**Pattern**: Multiple coupled fields (u, v, lap) with tuple-based increments
```cpp
field::PaddedBrick<double> u(decomp, rank, hw);
field::PaddedBrick<double> v(decomp, rank, hw);
field::PaddedBrick<double> lap(decomp, rank, hw);

// Per-point Laplacian aggregate
struct WaveLaplacian {
    double lxx = 0.0;
    double lyy = 0.0;
};

// Tuple-based increments
struct WaveIncrements {
    double du = 0.0;
    double dv = 0.0;
    auto as_tuple() { return std::tie(du, dv); }
};
```

**Evidence**:
- Multiple fields with identical geometry
- Per-point aggregate structures (WaveLaplacian, WaveIncrements)
- Coordinated halo exchange across fields
- Tuple-based access patterns

**Design Implications**:
- `FieldBundle<Ts...>` groups multiple fields with coordinated validation
- Shape validation across all fields in bundle
- Support for heterogeneous field types (double, complex)
- Indexed access via `get<I>()`

### Complex Field Usage in Spectral Methods

**Source**: Spectral apps using `ModelFieldRegistry`

**Pattern**: Complex fields for Fourier transforms
```cpp
using ComplexField = std::vector<std::complex<double>>;
ComplexField u_hat(N);
```

**Evidence**:
- Same access patterns as real fields
- Used in spectral methods with FFT
- No special handling needed for complex types

**Design Implications**:
- `FieldView<T>` works with `double` and `std::complex<double>`
- No code duplication for type handling
- Template-based implementation handles both types uniformly

### Backend Differences

**CPU Backend**: `std::vector<T>` storage in `LocalField<T>`
```cpp
std::vector<double> m_data;
T* data() noexcept { return m_data.data(); }
```

**GPU Backend**: `pfc::gpu::GPUVector<T>` with device memory
```cpp
// Future implementation
pfc::gpu::GPUVector<double> m_data;
double* data() noexcept { return m_data.data(); }  // Device pointer
```

**Evidence**:
- Both backends provide `data()` and `size()` methods
- Geometry metadata is backend-agnostic
- `OPENPFC_HD` macro for host/device callable annotations

**Design Implications**:
- `FieldView<T>` is backend-agnostic (const T* + geometry)
- Backend-specific validation in separate headers (e.g., validation_cuda.hpp)
- No shared abstraction layer required
- Backend allocation APIs permitted

### MPI Coordination Patterns

**Source**: Existing drivers with halo exchange

**Pattern**: Non-blocking halo exchanges with timing control
```cpp
PaddedHaloExchanger<double> halo(decomp, rank, hw, comm, tag);
halo.start_exchange(u.data(), u.size());
// Overlap with computation
halo.finish_exchange(u.data(), u.size());
```

**Evidence**:
- Halo exchange orchestrated by driver, not integrator
- Non-blocking exchanges for overlap with computation
- Timing controlled by application driver
- Boundary conditions applied at specific stages

**Design Implications**:
- `StageContext` carries timing information (time, dt, stage_index)
- Region requirements (interior vs boundary vs all)
- Flags for boundary update and halo exchange needs
- Driver reads context and schedules MPI operations

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

Backend memory-space compatibility validation is deferred to separate backend-specific headers:

### Current Implementation (Generic)

```cpp
// include/openpfc/kernel/field/validation.hpp
template<typename T>
void validate_shape_compatibility(const FieldView<T>& field1,
                                   const FieldView<T>& field2);

template<typename T, typename InputView>
void validate_no_alias(const FieldOutput<T>& output,
                        const InputView& input);
```

### Future Backend-Specific Validation

```cpp
// include/openpfc/kernel/field/validation_cuda.hpp (when GPU is implemented)
namespace pfc::field {

template<typename T>
void validate_backend_compatibility(const FieldView<T>& field1,
                                     const FieldView<T>& field2) {
    // Check if fields use compatible CUDA storage
    // Throws if mixing CPU and GPU storage
}

} // namespace pfc::field
```

### Why Backend Validation is Deferred

1. **BackendTag parameters cannot be obtained from FieldView<T>**:
   - `FieldView<T>` holds only `const T*` and geometry
   - Backend type information is not available at validation time
   - Backend checks require access to backend type information

2. **Separate headers enable backend-specific implementations**:
   - CUDA validation can use CUDA-specific APIs
   - HIP validation can use HIP-specific APIs
   - Generic validation remains backend-agnostic

3. **Avoids contaminating generic code**:
   - Generic algorithms don't need backend-specific includes
   - CPU-only builds don't need CUDA/HIP headers
   - Clear separation of concerns

## Backend Extensibility Strategy

The design enables CUDA/HIP implementation without changing semantic contracts:

### 1. Backend-Agnostic FieldView

```cpp
// Works with any contiguous storage
std::vector<double> cpu_storage;
FieldView<double> cpu_view(cpu_storage.data(), cpu_storage.size(), /* geometry */);

pfc::gpu::GPUVector<double> gpu_storage;
FieldView<double> gpu_view(gpu_storage.data(), gpu_storage.size(), /* geometry */);
```

### 2. Backend-Specific Allocation

```cpp
// CPU: Use std::vector
std::vector<double> cpu_data(64, 1.0);

// GPU: Use GPUVector (when implemented)
pfc::gpu::GPUVector<double> gpu_data(64, 1.0);
```

### 3. No Virtual Dispatch

All types are value-semantic templates:
```cpp
template<typename T>
void evaluate_operator(const FieldView<T>& input, FieldOutput<T>& output);
```

Compiler can inline and optimize without virtual function overhead.

### 4. No Return-by-Value Allocations

Output storage is caller-provided:
```cpp
std::vector<double> output_data(64);
FieldOutput<double> output(output_data.data(), output_data.size());
evaluate_operator(input, output);
```

### 5. Device-Specific APIs Permitted

Backend-specific allocation and synchronization APIs are permitted:
```cpp
// CUDA-specific (when implemented)
cudaMemcpyAsync(...);
cudaDeviceSynchronize();
```

Semantic contracts remain identical across backends.

## Non-Scope Items

The following items are explicitly out of scope for this design slice:

### Not Implemented Here

- **Virtual `TimeIntegrator` hierarchy**: Intended for future work items
- **`FieldView` class as virtual interface**: Value-semantic approach chosen instead
- **Return-by-value field moves**: Output storage is caller-provided
- **Device pointer/copy APIs**: Backend-specific, deferred to GPU work items
- **Runtime string-based factory methods**: Not needed for value-semantic types
- **Single unified container**: Multiple types for different use cases
- **One fixed layout**: Backend-specific layouts permitted
- **Preservation of `GPUVector`**: Legacy symbol, evidence only

### Deferred to Future Work Items

- **Integration with driver orchestration**: Covered in later work items
- **Operator evaluation seam**: Covered in later work items
- **Method invocation seam**: Covered in later work items
- **Step-attempt semantic**: Covered in later work items
- **CUDA/HIP implementation**: Backend-specific, deferred to GPU work items

## Architecture Summary

### Value-Semantic Field Accessors

```cpp
// Read-only view for operator inputs
template<typename T>
class FieldView {
    const T* data() const noexcept;
    std::size_t size() const noexcept;
    pfc::types::Int3 extents() const noexcept;
    pfc::types::Real3 spacing() const noexcept;
    pfc::types::Real3 origin() const noexcept;
    bool is_compatible_with(const FieldView& other) const noexcept;
};

// Mutable output for operator results
template<typename T>
class FieldOutput {
    T* data() noexcept;
    std::size_t size() const noexcept;
    template<typename InputView>
    void validate_no_alias(const InputView& input) const;
};

// Multi-field bundle for coupled systems
template<typename... Fields>
class FieldBundle {
    template<std::size_t I> auto& get() noexcept;
    bool validate_shapes() const noexcept;
};
```

### Integrator-Owned Workspace

```cpp
template<typename T>
class Workspace {
    explicit Workspace(const pfc::types::Int3& extents, std::size_t num_stages);
    T* stage(std::size_t stage_index) noexcept;
    T* scratch() noexcept;
    void clear() noexcept;
};
```

### MPI Coordination Context

```cpp
struct StageContext {
    double time;
    double dt;
    int stage_index;
    enum class RegionKind { Interior, Boundary, All } region_kind;
    bool needs_boundary_update;
    bool needs_halo_exchange;
};
```

### Validation Utilities

```cpp
// Generic validation (backend-agnostic)
template<typename T>
void validate_shape_compatibility(const FieldView<T>& field1,
                                   const FieldView<T>& field2);

template<typename T, typename InputView>
void validate_no_alias(const FieldOutput<T>& output,
                        const InputView& input);

// Backend-specific validation (future)
// template<typename T>
// void validate_backend_compatibility(const FieldView<T>& field1,
//                                      const FieldView<T>& field2);
```

## Testing Strategy

### CPU Unit Tests

- **FieldView const access**: Verify read-only semantics
- **FieldOutput mutable access**: Verify write semantics
- **Aliasing detection**: Verify alias rejection
- **Shape compatibility**: Verify shape validation
- **Multi-field bundles**: Verify coordinated access

### Contract Tests

- **Read/write contract**: Verify observational non-mutation
- **Const/mutable separation**: Verify const correctness
- **Backend semantic equivalence**: Verify CPU/GPU contract match

### Evidence Tests

- **Heat3D scalar field**: Verify numerical equivalence with existing pattern
- **Wave2D multi-field**: Verify numerical equivalence with existing pattern

### Compile-Time Tests

- **FieldView concept**: Verify type requirements
- **FieldOutput concept**: Verify type requirements
- **FieldBundle concept**: Verify type requirements
- **Compatible fields concept**: Verify compatibility constraints

## Conclusion

The evidence-driven state access and storage design provides:

1. **Clear semantic contracts**: Read-only inputs, mutable outputs, no aliasing
2. **Backend extensibility**: CPU and GPU use same semantic contracts
3. **Minimal overhead**: Value-semantic types, no virtual dispatch
4. **Validation at binding time**: Shape and aliasing checks once, not per evaluation
5. **MPI coordination**: Stage context enables driver-level orchestration

The design is justified by evidence from existing Heat3D and Wave2D patterns, enables CUDA/HIP implementation without changing semantic contracts, and provides clear migration paths from existing code.
