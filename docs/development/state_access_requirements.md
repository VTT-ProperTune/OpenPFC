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

**Example from Heat3D**:
```cpp
auto du = /* output storage */;  // Caller-provided, separate from u
double* du_data = du.data();
// Write residual into du
```

### Real and Complex Type Support

- **Single interface for both types**: Same access patterns work for `double` and `std::complex<double>`
- **No code duplication**: Templates or type erasure handle both types uniformly
- **Type erasure or templates**: Use compile-time polymorphism or type erasure for generic programming

**Example from spectral methods**:
```cpp
// Same access patterns for real and complex fields
template<typename T>
void evaluate_operator(const FieldView<T>& input, FieldOutput<T>& output);
```

### Multi-Field Bundles

- **Group related fields**: Coupled systems (e.g., u/v for wave equation) require coordinated access
- **Named field access**: Fields may be accessed by name or index (e.g., `ModelFieldRegistry` pattern)
- **Coordinated shape validation**: All fields in a bundle must have compatible shapes

**Example from Wave2D**:
```cpp
// Multi-field bundle for wave equation
FieldBundle<FieldView<double>, FieldView<double>> fields(u_view, v_view);
fields.validate_shapes();  // Ensure u and v have compatible geometry
```

## Aliasing Validation Requirements

### Input/Output Alias Detection

- **Runtime check**: Output storage must not alias input fields
- **Exception on aliasing**: Throw exception if aliasing detected before mutation
- **Documented in-place patterns**: Explicitly document supported in-place patterns (e.g., `u += dt*du` via `ScaledField`)

**Example**:
```cpp
FieldOutput<double> output(/* storage */);
FieldView<double> input(/* storage */);

// Must throw if output storage aliases input storage
output.validate_no_alias(input);
```

### Const Correctness

- **Input parameters as const**: Inputs passed as `const&` or by value
- **Output parameters as non-const**: Outputs as non-const pointers/references
- **Const alone insufficient**: Const references alone do not detect aliasing; need pointer comparison
- **Verified by C++ type system**: Const correctness is enforced at compile time through const member functions and const/non-const return types; FieldView<T>::data() returns const T* (read-only) while FieldOutput<T>::data() returns T* (mutable)

**Example**:
```cpp
// Const alone does not prevent aliasing
void evaluate(const FieldView<double>& input, FieldOutput<double>& output);

// Need explicit alias check
output.validate_no_alias(input);
```

## Shape/Layout Compatibility Requirements

### Size Validation

- **Matching size()**: Fields used together must have matching `size()`
- **Check at binding time**: Validation occurs at construction/binding time, not during evaluation
- **Clear error messages**: Indicate which fields and which dimensions mismatch

**Example**:
```cpp
FieldView<double> field1(/* size=64 */);
FieldView<double> field2(/* size=128 */);

// Must throw at binding time, not during evaluation
validate_shape_compatibility(field1, field2);  // throws std::invalid_argument
```

### Layout Compatibility

- **Same row-major ordering**: All fields use x-fastest row-major ordering
- **Matching halo widths**: Padded fields must have matching halo widths
- **Compatible spacing/origin**: Coordinate transforms require compatible spacing and origin

**Example**:
```cpp
FieldView<double> field1(/* extents={8,8,8}, spacing={1.0,1.0,1.0} */);
FieldView<double> field2(/* extents={8,8,8}, spacing={1.0,1.0,1.0} */);

// Must check extents, spacing, and origin
field1.is_compatible_with(field2);  // returns true
```

### Validation Boundary

- **Single validation point**: Shape checks performed when fields are bound to operators
- **No per-evaluation overhead**: Validation happens once at binding, not per evaluation
- **Clear ownership**: Each operator-field binding has a clear validation boundary

**Example**:
```cpp
// Validate once at construction
class LaplacianOperator {
public:
    LaplacianOperator(const FieldView<double>& input, FieldOutput<double>& output) {
        validate_shape_compatibility(input, output);
        // Validation complete; no per-evaluation overhead
    }
};
```

## Backend Memory-Space Compatibility Requirements

### CPU and GPU Semantic Equivalence

- **Same read/write contracts**: Both backends obey identical semantic contracts
- **Backend-specific allocation APIs**: Each backend uses its own allocation mechanisms
- **No shared abstraction layer**: No requirement for a unified memory abstraction

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

- **Compile-time separation**: Backend-specific headers provide backend-aware validation
- **Runtime checks for mixing**: Disallow mixing CPU and GPU storage in same operation
- **Clear error messages**: Indicate backend mismatch when detected

**Example**:
```cpp
// Backend-specific validation in separate headers
// include/openpfc/kernel/field/validation_cuda.hpp (future)
template<typename T>
void validate_backend_compatibility(const FieldView<T>& field1,
                                     const FieldView<T>& field2);

// Throws if fields use different backends
```

**Note**: FieldView<T> is intentionally backend-agnostic (const T* + geometry metadata) to allow single header usage across CPU and GPU backends. Backend-specific validation will be provided in separate headers (e.g., validation_cuda.hpp, validation_hip.hpp) when GPU implementation is pursued.

## Workspace Storage Requirements

### Integrator-Owned Allocation

- **Scratch buffers owned by integrator**: Integrator objects allocate and own scratch buffers
- **Lifetime options**: Per-method-object, per-step, or pooled lifetime
- **Not exposed to models or drivers**: Workspace is internal to integrators

**Example**:
```cpp
// Integrator owns workspace
class RungeKuttaIntegrator {
public:
    RungeKuttaIntegrator(const pfc::Int3& extents, std::size_t num_stages)
        : m_workspace(extents, num_stages) {}
    
    // Workspace not exposed to physics models
private:
    Workspace<double> m_workspace;
};
```

### Stage Storage

- **Intermediate RK stages**: RK stages stored in workspace
- **Reuse across steps**: Reused across time steps to avoid allocation
- **Size matches field dimensions**: Stage storage sized to match field extents

**Example**:
```cpp
// Workspace provides stage storage
Workspace<double> workspace(extents, /* num_stages=4 */);

// Access stage storage for RK4
double* k1 = workspace.stage(0);
double* k2 = workspace.stage(1);
double* k3 = workspace.stage(2);
double* k4 = workspace.stage(3);
```

### Scratch Buffers

- **Temporary workspace**: Scratch buffers for operator evaluations
- **Cleared/reclaimed**: Buffers cleared or reclaimed between uses
- **No persistent state**: No persistent state across time steps

**Example**:
```cpp
// Scratch buffer for temporary computation
double* scratch = workspace.scratch();

// Use for intermediate results
compute_laplacian(input, scratch);
apply_rhs(scratch, output);

// Clear for next use
workspace.clear();
```

## MPI Coordination Requirements

### Stage Context

- **Evaluation time and timestep**: Context carries current time `t` and timestep `dt`
- **Stage index**: RK stage index or method-specific stage identifier
- **Field region patterns**: Interior vs boundary access patterns

**Example**:
```cpp
StageContext ctx{
    .time = 0.0,
    .dt = 0.01,
    .stage_index = 0,
    .region_kind = RegionKind::Interior,
    .needs_boundary_update = false,
    .needs_halo_exchange = true
};
```

### Halo Exchange Timing

- **Orchestrated by driver**: Driver controls halo exchange timing
- **Triggered by context requests**: Integrator indicates need via stage context
- **Non-blocking exchanges**: Overlap halo exchange with computation

**Example**:
```cpp
// Driver reads context and schedules halo exchange
if (ctx.needs_halo_exchange) {
    halo_exchanger.start_exchange(field);
    // Overlap with interior computation
    compute_interior(field, scratch);
    halo_exchanger.finish_exchange(field);
}
```

## Validation Contracts

### Compile-Time Concepts (Where Possible)

- **Field compatibility concepts**: Concepts for field type requirements
- **Backend tag concepts**: Concepts for backend-specific constraints
- **Value semantic requirements**: Concepts for copyable/movable types

**Example**:
```cpp
template<typename T>
concept FieldViewLike = requires(T v) {
    { v.data() } -> std::convertible_to<const double*>;
    { v.size() } -> std::convertible_to<std::size_t>;
    { v.extents() } -> std::convertible_to<pfc::Int3>;
};
```

### Runtime Contracts

- **Shape compatibility checks**: Runtime validation of field geometry
- **Aliasing checks**: Runtime pointer comparison for alias detection
- **Backend compatibility checks**: Runtime validation of backend mixing

**Example**:
```cpp
// Runtime validation throws on violation
validate_shape_compatibility(field1, field2);
validate_no_alias(output, input);
validate_backend_compatibility(field1, field2);  // Backend-specific
```

### CPU-Focused Unit Tests

- **FieldView const access**: Verify read-only semantics
- **FieldOutput mutable access**: Verify write semantics
- **Aliasing detection**: Verify alias rejection
- **Shape compatibility**: Verify shape validation
- **Multi-field bundles**: Verify coordinated access

### Contract Tests

- **Read/write contract**: Verify observational non-mutation
- **Const/mutable separation**: Verify const correctness
- **Backend semantic equivalence**: Verify CPU/GPU contract match

## Non-Scope Items

The following items are explicitly out of scope for this design slice:

- Virtual `TimeIntegrator` hierarchy
- `FieldView` class as virtual interface
- Return-by-value field moves
- Device pointer/copy APIs between host and device
- Runtime string-based factory methods
- Single unified container for all field types
- One fixed layout enforced across all backends
- Preservation of `GPUVector` or other legacy symbols (they are evidence only)
- Integration with driver orchestration (covered in later work items)
- Operator evaluation seam implementation (covered in later work items)
- Method invocation seam (covered in later work items)
- Step-attempt semantic implementation (covered in later work items)

## Backend Extensibility Strategy

The design enables CUDA/HIP implementation without changing semantic contracts:

1. **Backend-agnostic FieldView**: `FieldView<T>` holds only `const T*` and geometry metadata, allowing single header usage across backends
2. **Backend-specific validation**: Separate headers (e.g., `validation_cuda.hpp`, `validation_hip.hpp`) provide backend-aware checks
3. **No virtual dispatch**: Value-semantic design avoids virtual function calls
4. **No return-by-value allocations**: Output storage is caller-provided
5. **Device-specific APIs**: Backend-specific allocation and synchronization APIs are permitted

Backend-specific validation is deferred because:
- `BackendTag` template parameters cannot be obtained from `FieldView<T>` arguments
- Backend checks require access to backend type information not available in generic code
- Separate headers allow backend-specific implementations without contaminating generic code
