// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

# State Access and Storage Requirements

This document specifies the evidence-based semantic requirements for scalar and coupled real/complex field state and integrator workspace access in OpenPFC.

## Evidence Sources

The requirements are derived from existing field implementations and usage patterns:

- **Scalar field patterns** (`apps/heat3d/`): `pfc::field::LocalField<double>`, `pfc::field::PaddedBrick<double>`, `pfc::field::Field<double>`
- **Multi-field patterns** (`apps/wave2d/`): Multiple `PaddedBrick<double>` instances (u, v, lap), tuple-based increments (`WaveIncrements{du, dv}`), per-point Laplacian aggregate (`WaveLaplacian{lxx, lyy}`)
- **Complex field patterns**: `ComplexField` = `std::vector<std::complex<double>>`, used in spectral methods via `ModelFieldRegistry`
- **Backend differences**: CPU `std::vector<T>` storage in `LocalField<T>` vs GPU `pfc::gpu::GPUVector<T>`, `OPENPFC_HD` macro for host/device callable annotations
- **MPI coordination**: `PaddedHaloExchanger<T>` for non-blocking halo exchanges, face-only exchange patterns (6-direction), timing controlled by application driver

## State Representation Requirements

### Read Access for Operator Inputs

- **Const access to field data**: Operators must read input fields through const interfaces
- **Geometry queries**: Operators must query size, spacing, origin, and global bounds
- **No mutation of input state**: Input fields must remain observationally unchanged during evaluation

**Verification**: FieldView<T> provides const-only access to field data through `data()` const member function. The const qualifier prevents mutation through the view.

**Example from Heat3D**:
```cpp
const auto& u = /* current state */;
const double* u_data = u.data();
std::size_t u_size = u.size();
pfc::Int3 u_extents = u.extents();
// Read-only access for gradient evaluation
```

### Write Access for Operator Outputs

- **Mutable output storage provided by caller**: Operators write into caller-provided storage
- **Separate from input state**: Output storage must be distinct from input state (no in-place mutation unless explicitly supported)
- **Shape compatibility with input fields**: Output must have matching size and layout

**Verification**: FieldOutput<T> provides mutable access through `data()` non-const member function. Aliasing validation ensures output storage does not alias input storage.

**Example from Heat3D**:
```cpp
auto du = /* output storage */;  // Caller-provided, separate from u
double* du_data = du.data();
// Write residual into du
```

### Real and Complex Type Support

- **Single interface for both types**: Same access patterns work for `double` and `std::complex<double>`
- **No code duplication**: Templates provide generic programming support

**Verification**: FieldView<T>, FieldOutput<T>, and FieldBundle<Ts...> are all template types parameterized by value type T, supporting both `double` and `std::complex<double>` without code duplication.

## Aliasing Validation Requirements

### Input/Output Alias Detection

- **Runtime check**: Output storage must not alias input fields
- **Exception thrown**: Aliasing detected before mutation
- **Documented in-place patterns**: Explicitly supported patterns (e.g., `u += dt*du` via `ScaledField`)

**Verification**: `validate_no_alias()` function performs pointer comparison to detect memory overlap. It throws `std::invalid_argument` if aliasing is detected.

**Implementation details**:
```cpp
template<typename T, typename InputView>
void validate_no_alias(const FieldOutput<T>& output, const InputView& input) {
    // Pointer range overlap check
    const void* output_ptr = static_cast<const void*>(output.data());
    const void* input_ptr = static_cast<const void*>(input.data());
    // Check if ranges overlap
    if (!(output_end <= input_start || input_end <= output_start)) {
        throw std::invalid_argument("output storage aliases input storage");
    }
}
```

### Const Correctness

- **Input parameters**: Passed as `const&` or by value
- **Output parameters**: Non-const pointers/references
- **Const alone insufficient**: Need pointer comparison for alias detection

**Verification**: FieldView<T> only provides const access to data. FieldOutput<T> provides mutable access but requires explicit validation against input views.

## Shape/Layout Compatibility Requirements

### Size Validation

- **Matching size()**: Fields used together must have matching size
- **Checked at construction/binding**: Validation performed once, not per-evaluation
- **Clear error messages**: Indicate which fields are incompatible

**Verification**: `validate_shape_compatibility()` function checks that extents, spacing, and origin match between two fields. Throws `std::invalid_argument` with detailed error message if incompatible.

**Implementation details**:
```cpp
template<typename T>
void validate_shape_compatibility(const FieldView<T>& field1,
                                   const FieldView<T>& field2) {
    if (!field1.is_compatible_with(field2)) {
        throw std::invalid_argument(
            "validate_shape_compatibility: fields have incompatible shapes\n"
            "  Field1 extents: [" + std::to_string(field1.extents()[0]) + ", " + ... + "]\n"
            "  Field2 extents: [" + std::to_string(field2.extents()[0]) + ", " + ... + "]");
    }
}
```

### Layout Compatibility

- **Same row-major ordering**: x-fastest memory layout
- **Matching halo widths**: For padded fields
- **Compatible spacing/origin**: For coordinate transforms

**Verification**: `FieldView::is_compatible_with()` checks structural compatibility including extents, spacing, and origin. Halo width compatibility is the responsibility of the calling code (different from shape compatibility).

### Validation Boundary

- **Construction/binding time**: Shape checks performed when fields are bound to operators
- **Single validation point**: One check per operator-field binding
- **No per-evaluation overhead**: Validation done once, not per call

**Verification**: Validation functions are designed to be called at construction/binding time. Evidence tests demonstrate calling validation once before computation loops.

## Backend Memory-Space Compatibility Requirements

### CPU and GPU Semantic Equivalence

- **Same read/write contracts**: Both backends obey identical semantic contracts
- **Backend-specific allocation APIs**: Each backend uses its own allocation mechanisms
- **No shared abstraction layer**: No requirement for a unified memory abstraction

**Verification**: FieldView<T> is backend-agnostic, working with any contiguous storage (CPU std::vector, GPU GPUVector, etc.). The same read/write contracts apply to both backends.

**Example**:
```cpp
// CPU: std::vector storage
std::vector<double> cpu_storage;
FieldView<double> cpu_view(cpu_storage.data(), cpu_storage.size(), /* geometry */);

// GPU: GPUVector storage (when implemented)
pfc::gpu::GPUVector<double> gpu_storage;
FieldView<double> gpu_view(gpu_storage.data(), gpu_storage.size(), /* geometry */);

// Same read/write contracts for both
```

### Backend Memory-Space Verification

**Current implementation (CPU-only slice)**:
- **Compile-time backend tag checks**: `validate_backend_compatibility()` uses template parameters to enforce backend compatibility at compile time
- **Static assertion**: Requires `BackendTag1` and `BackendTag2` to be the same type
- **Extensibility**: Backend-specific implementations can provide runtime checks (e.g., CUDA device ID compatibility, HIP stream compatibility)

**Verification method**:
```cpp
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

**Usage example**:
```cpp
struct CPUBackendTag {};
struct CUDABackendTag {};

// CPU fields are compatible with each other
validate_backend_compatibility<double, CPUBackendTag, CPUBackendTag>(field1, field2); // OK

// Mixing CPU and GPU fields is not allowed (compile-time error)
validate_backend_compatibility<double, CPUBackendTag, CUDABackendTag>(field1, field2); // Static assertion failure
```

**Future GPU implementation path**:
- Backend-specific headers (e.g., `validation_cuda.hpp`, `validation_hip.hpp`) can specialize or override the default implementation
- Specializations can add runtime checks for device ID compatibility, stream compatibility, etc.
- The generic interface remains the same, ensuring semantic equivalence across backends

**Note**: FieldView<T> is intentionally backend-agnostic (const T* + geometry metadata) to allow single header usage across CPU and GPU backends. Backend tags are provided by the calling code, not extracted from FieldView, enabling compile-time enforcement without virtual dispatch.

## Workspace Storage Requirements

### Integrator-Owned Allocation

- **Scratch buffers owned by integrator**: Integrator objects allocate and own scratch buffers
- **Lifetime options**: Per-method-object, per-step, or pooled lifetime
- **Not exposed to models or drivers**: Workspace is internal to integrators

**Verification**: Workspace<T> class owns all storage (stage buffers and scratch buffer). Storage is allocated in constructor and freed in destructor. No access to workspace storage is provided outside the integrator.

**Example**:
```cpp
// Integrator allocates workspace
Workspace<double> workspace(extents, num_stages);

// Integrator uses workspace internally
double* stage1 = workspace.stage(0);
double* scratch = workspace.scratch();

// Workspace is not exposed to physics models or drivers
```

### Stage Storage

- **Intermediate RK stages**: Stored in workspace
- **Reused across steps**: Avoids allocation overhead
- **Size matches field dimensions**: Stage buffers sized to field extents

**Verification**: Workspace<T>::stage() returns pointers to pre-allocated stage buffers. Stage count and size are fixed at construction time.

### Scratch Buffers

- **Temporary workspace**: For operator evaluations
- **Cleared/reclaimed between uses**: No persistent state across steps
- **Lifetime**: Managed by integrator, not exposed externally

**Verification**: Workspace<T>::scratch() returns pointer to scratch buffer. Workspace<T>::clear() resets all buffers to zero.

## MPI Coordination Requirements

### Stage Context

- **Evaluation time t**: Current simulation time
- **Timestep dt**: Attempted timestep size
- **Stage index**: RK stage or method-specific stage identifier
- **Field region requirements**: Interior vs boundary vs all
- **Boundary condition requirements**: Whether BCs need updating
- **Halo exchange requirements**: Whether halo exchange is needed

**Verification**: StageContext struct carries all required fields. Integrators populate StageContext before each evaluation. Drivers read StageContext to coordinate MPI operations.

**Example**:
```cpp
StageContext ctx;
ctx.time = t;
ctx.dt = dt;
ctx.stage_index = stage;
ctx.region_kind = StageContext::RegionKind::Interior;
ctx.needs_boundary_update = false;
ctx.needs_halo_exchange = true;

// Driver uses context to coordinate MPI
if (ctx.needs_halo_exchange) {
    exchange_halos(field);
}
```

### Halo Exchange Timing

- **Orchestrated by driver**: Not by integrator
- **Triggered by stage context**: Based on region requirements
- **Non-blocking exchanges**: Overlap with computation when possible

**Verification**: StageContext::needs_halo_exchange flag indicates when halo exchange is required. Drivers check this flag and perform exchange before evaluation if needed.

## Summary of Verification Methods

| Requirement | Verification Method | Location |
|-------------|-------------------|----------|
| Const input semantics | FieldView<T>::data() const | state_access.hpp |
| Mutable output semantics | FieldOutput<T>::data() non-const | state_access.hpp |
| Aliasing detection | validate_no_alias() pointer comparison | validation.hpp |
| Shape compatibility | validate_shape_compatibility() structural check | validation.hpp |
| Backend compatibility | validate_backend_compatibility() static_assert | validation.hpp |
| Workspace ownership | Workspace<T> private member variables | workspace.hpp |
| Stage context coordination | StageContext struct fields | stage_context.hpp |

All verification is performed at compile time (static_assert, type system) or at binding time (validation functions called once during construction/binding), ensuring no per-evaluation overhead.
