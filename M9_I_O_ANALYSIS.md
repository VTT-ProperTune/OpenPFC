# M9 I/O and Output Management Analysis

## Overview
Analysis of OpenPFC I/O patterns for M9 consolidation work completed 2026-08-01 (30/30 tests passing baseline).

## Current I/O Patterns

### 1. ResultsWriter Interface (`results_writer.hpp`)
**Purpose:** Abstract base class defining unified output writer interface
**Scope:** Output file format abstraction, parallel I/O coordination
**Key Responsibilities:**
- Define output filename pattern support (e.g., "output_%04d.bin")
- Configure domain decomposition (global size, local size, local offset)
- Write real fields (double precision data)
- Write complex fields (Fourier coefficients)
- Handle time step indexing for output files
- Coordinate parallel I/O across MPI ranks

**Key Interface Methods:**
- `set_domain(global_size, local_size, offset)` - Configure parallel I/O layout
- `write(increment, RealField)` - Write real-valued field at timestep
- `write(increment, ComplexField)` - Write complex-valued field at timestep
- `<T> write()` - Template overload for generic data vectors

**Design Features:**
- **Interface segregation:** Minimal abstract interface (set_domain, write)
- **Format flexibility:** Subclasses implement specific file formats
- **Parallel-aware:** Built for MPI from the ground up
- **Composable:** Multiple writers can be used simultaneously

**Example Usage:**
```cpp
auto writer = std::make_unique<BinaryWriter>("output_%04d.bin");
writer->set_domain(global_size, local_size, local_offset);
for (int step = 0; step < num_steps; ++step) {
    writer->write(step, field);  // Creates output_0000.bin, output_0001.bin, ...
}
```

---

### 2. BinaryWriter (`binary_writer.hpp`)
**Purpose:** Binary format writer for raw field data output
**Scope:** High-performance checkpointing and restart operations
**Key Responsibilities:**
- Raw binary format output (optimal for checkpointing)
- MPI-IO collective file operations
- Filetype/etype management for subarray views
- Buffer-size validation (fail-closed policy)
- Parallel write coordination across ranks

**MPI-IO Collective Operations:**
- Uses `MPI_File_open`, `MPI_File_set_size`, `MPI_File_set_view`
- `MPI_Type_create_subarray` for distributed data layout
- `MPI_File_write_all` for collective writes
- Communicator-wide error handling and validation

**Critical Safety Features:**
- **Buffer validation:** Collective checks ensure buffer size matches set_domain
- **Fail-closed policy:** MPI errors throw exceptions via `throw_on_mpi_error`
- **Filetype management:** Automatic MPI_Type_free in destructor
- **Domain validation:** Subarray bounds checking via `validate_subarray_domain`

**Example Usage:**
```cpp
BinaryWriter writer("checkpoint_%04d.bin", MPI_COMM_WORLD);
writer.set_domain(global_size, local_size, offset);
writer.write(step, field);  // Collective MPI write
```

**Optimal Use Cases:**
- Checkpointing and restart (exact data preservation)
- Large-scale simulations (minimal storage overhead)
- Fast I/O performance (no parsing or conversion)

---

### 3. VTKWriter (`vtk_writer.hpp`)
**Purpose:** VTK ImageData format writer for visualization
**Scope:** Scientific visualization output (ParaView, VisIt compatible)
**Key Responsibilities:**
- VTK ImageData format generation (.vti files)
- Parallel output support (.pvti master + .vti pieces per rank)
- Real and complex field writing (complex writes magnitude)
- Domain origin/spacing metadata
- Extension-based filename generation

**Visualization Support:**
- **Serial output:** Single .vti file for standalone runs
- **Parallel output:** .pvti master file coordinating per-rank .vti pieces
- **Base64 encoding:** Binary format for compact output
- **Field metadata:** Origin, spacing, custom field names

**Example Usage:**
```cpp
VTKWriter writer("results_%04d.vti", MPI_COMM_WORLD);
writer.set_domain(global_size, local_size, offset);
writer.set_origin({0.0, 0.0, 0.0});
writer.set_spacing({1.0, 1.0, 1.0});
writer.set_field_name("density");
writer.write(step, field);  // Creates .vti or .pvti + pieces
```

**Design Features:**
- **Format standard:** VTK ImageData specification compliant
- **Tool integration:** Direct compatibility with ParaView/VisIt
- **MPI awareness:** Automatic .pvti generation for parallel runs
- **Complex handling:** Computes magnitude for complex field visualization

---

### 4. Checkpoint System (`state_capture.hpp`)
**Purpose:** Capture and validate-before-mutate restore for checkpoint payloads
**Scope:** Versioned state management with safety guarantees
**Key Responsibilities:**
- Field capture from contiguous buffers into FieldPayload
- Component capture for integrator state
- Validate-before-mutate restore pattern
- Comprehensive error checking and rejection

**Validation Framework:**
- **Version checking:** Payload version compatibility validation
- **Field ID matching:** Ensures correct field restored to correct destination
- **Data type verification:** Float64 vs Complex128 type matching
- **Shape validation:** Extent consistency across source and destination
- **Coordinate order:** XFastest layout enforcement
- **Decomposition metadata:** MPI layout consistency (optional)
- **Buffer sizing:** Byte-length capacity checking

**Restore Safety Guarantees:**
```cpp
// Validate BEFORE any write operation
if (payload.version != expected_version) {
    return RestoreError::VersionMismatch;  // Destination unchanged
}
// All validations pass, then write
std::memcpy(destination.data(), payload.bytes.data(), nbytes);
```

**Exclusions (Intentionally Not Captured):**
- Stage buffers (Workspace stages)
- FFT plans
- Operator caches
- Halo rings
- Driver-owned Time/increment/config identity

**Test-Friendly Architecture:**
- Template-based `capture_field<T>` works with any contiguous buffer
- `std::as_bytes` enables testing with std::vector<double>
- Injectable PublishedFieldBrick for Catch2 without payload carrier dependency

---

### 5. Checkpoint Metadata (`checkpoint_metadata.hpp`)
**Purpose:** Versioned metadata for filesystem checkpoint publication
**Scope:** Checkpoint bundle identity and provenance
**Key Responsibilities:**
- Record accepted simulation time and increment
-Document domain geometry (dimensions, origin, spacing)
- Optional MPI decomposition descriptors
- Integrator method identity

**Metadata Schema:**
```json
{
  "format_version": 1,
  "accepted_time": 12.5,
  "accepted_increment": 125,
  "domain": {
    "global_dimensions": [256, 256, 256],
    "physical_origin": [0.0, 0.0, 0.0],
    "grid_spacing": [1.0, 1.0, 1.0]
  },
  "decomposition": {
    "mpi_size": 8,
    "local_size": [128, 256, 256],
    "local_offset": [0, 0, 0]
  },
  "method_identity": "RK3Heun"
}
```

**Design Contracts:**
- **Caller fills time/increment:** Uses `Time::get_current()` / `get_increment()`
- **JSON serialization:** Automatic `to_json()` metadata.json generation
- **Decomposition optional:** Single-rank runs omit decomposition field

---

### 6. Atomic Publication (`publish.hpp`)
**Purpose:** Atomic filesystem publication of accepted checkpoint bundles
**Scope:** Transactional checkpoint directory creation
**Key Responsibilities:**
- Stage checkpoint under `<final_dir>.publishing/`
- Write versioned metadata.json
- Write field bricks under `fields/<field_id>.bin`
- Atomic rename to final directory
- Rollback on any failure

**Atomic Publication Pattern:**
```
staging: checkpoint_125.publishing/
  metadata.json
  fields/density.bin
  fields/order_parameter.bin
→ atomic rename →
final: checkpoint_125/
  metadata.json
  fields/density.bin
  fields/order_parameter.bin
```

**Failure Safety:**
- **Never expose incomplete checkpoints:** Staging directory renamed atomically
- **Proven rollback:** `best_effort_remove_all` cleans staging on failures
- **Exception-safe:** Try-catch with rollback on any exception
- **Testing hooks:** Optional `PublishWriteHook` for fault injection tests

**Testing Support:**
- Injectable `PublishedFieldBrick` spans work with `std::as_bytes(std::vector<double>)`
- Test doubles can be injected without payload carrier dependency
- Comprehensive failure modes testable via pre-write hooks

---

### 7. ResultsWriterCatalog (`results_writer_catalog.hpp`)
**Purpose:** Type string → ResultsWriter factory for JSON wiring
**Scope:** Driver-level output format registration and instantiation
**Key Responsibilities:**
- Map JSON `"writer": "<type>"` strings to factory functions
- Built-in "binary" → BinaryWriter registration
- Custom writer type registration for applications
- Process-wide default catalog with extension capability

**JSON Integration:**
```json
{
  "fields": [
    {
      "name": "density",
      "writer": "binary",
      "output_pattern": "results_%04d.bin"
    },
    {
      "name": "visualization",
      "writer": "vtk",
      "output_pattern": "vis_%04d.vti"
    }
  ]
}
```

**Catalog extensibility:**
```cpp
ResultsWriterCatalog catalog = make_builtin_results_writer_catalog();
catalog.register_writer_type("vtk", [](std::string path, MPI_Comm comm) {
    return std::make_unique<VTKWriter>(std::move(path), comm);
});
```

**Design Philosophy:**
- **No inference:** Required catalog argument prevents hidden defaults
- **Application control:** Custom catalogs for format extensions
- **Test ease:** Built-in catalog available for tests

---

## Clear Separation of Concerns

| Pattern | Primary Concern | Context Level |
|---------|-----------------|---------------|
| **ResultsWriter** | Output abstraction interface | Library kernel |
| **BinaryWriter** | High-performance checkpointing | Frontend implementation |
| **VTKWriter** | Scientific visualization | Frontend implementation |
| **State Capture** | Versioned state management | Kernel checkpointing |
| **Checkpoint Metadata** | Checkpoint identity and provenance | Kernel checkpointing |
| **Atomic Publication** | Transactional filesystem operations | Kernel checkpointing |
| **ResultsWriterCatalog** | JSON-driven writer selection | Frontend wiring |

**No Overlapping Responsibilities:**
- ResultsWriter defines interface, not format implementation
- BinaryWriter handles binary format, not visualization
- VTKWriter handles visualization, not checkpointing
- State capture handles serialization, not filesystem operations
- Atomic publication handles filesystem, not validation
- ResultsWriterCatalog handles JSON integration, not I/O operations

---

## I/O Workflow Integration

### Simulation Output Flow
```
Simulator::end_integrator_step()
  → Iterates over ResultsWriterMap
    → For each ResultsWriter:
      → writer->write(increment, field)  [Collective MPI]
        → BinaryWriter: MPI_File_write_all
        → VTKWriter: std::ofstream + .pvti coordination
```

### Checkpoint/Restart Flow
```
**Checkpoint:**
→ Capture accepted field state via capture_field<T>()
→ Validate buffers match set_domain layout
→ Create versioned CheckpointMetadata (time, increment, method)
→ Stage under checkpoint_<increment>.publishing/
→ Publish atomically via rename
→ Expose: checkpoint_<increment>/metadata.json + fields/*.bin

**Restart:**
→ Read CheckpointMetadata from checkpoint_<increment>/
→ Validate version, time, domain, method match expectations
→ restore_field() with full validation (version, ID, dtype, shape, layout)
→ Validate-before-mutate: destination unchanged on any rejection
→ Resume simulation from accepted state
```

### JSON Configuration Flow
```
App::main()
  → Load JSON/TOML settings
  → wire_simulator_from_settings()
    → ResultsWriterCatalog provides writer factories
    → Create writers from "fields[].writer" type strings
    → Register in Simulator::results_writers()
  → Simulator::run()
    → end_integrator_step() iterates registered writers
```

---

## M9 Consolidation Assessment

### Current Design Quality: ✅ EXCELLENT
- Clear separation of concerns across 7 patterns
- No overlapping responsibilities
- Consistent interface patterns (set_domain, write)
- Comprehensive safety features (validation, atomic operations)
- Excellent testability (injectable components, hooks)

### Interface Consistency Assessment

#### ✅ ResultsWriter Interface [ALREADY CONSOLIDATED]
**Status:** Excellent unified interface
**Evidence:**
- Single abstract base class defining all output writers
- Consistent `set_domain()` + `write(increment, data)` pattern
- Built-in RealField and ComplexField support
- Template overload for generic vectors

#### ✅ Checkpoint/Restart Unification [ALREADY CONSOLIDATED]
**Status:** Excellent versioned state management
**Evidence:**
- Unified FieldPayload/ComponentPayload for state capture
- Comprehensive validation-before-mutate safety
- Versioned metadata with provenance tracking
- Atomic publication with rollback safety

#### ✅ Visualization Output Standardization [ALREADY CONSOLIDATED]
**Status:** VTK format standard with MPI awareness
**Evidence:**
- Single VTKWriter class for all visualization output
- Automatic parallel .pvti generation
- VTK specification compliant
- Compatible with ParaView, VisIt, other VTK tools

#### ✅ JSON Integration [ALREADY CONSOLIDATED]
**Status:** Excellent type string → factory pattern
**Evidence:**
- ResultsWriterCatalog for JSON-driven writer selection
- Built-in "binary" registration
- Extension API for custom writer types
- No hidden defaults (explicit catalog required)

### Completeness Verification

#### 1. Output Writer Interface Consolidation ✅
**Status:** Complete - ResultsWriter provides unified interface
**Rationale:** All output writers inherit from ResultsWriter with consistent patterns

#### 2. Checkpoint/Restart Unification ✅
**Status:** Complete - Full capture/validation/publication pipeline
**Analysis:**
- State capture with comprehensive validation
- Versioned metadata for provenance
- Atomic publication with safety guarantees
- Restore with validate-before-mutate semantics

#### 3. Visualization Output Standardization ✅
**Status:** Complete - VTKWriter provides standard format
**Analysis:**
- Single VTKWriter for all visualization
- Parallel output automatically handled
- VTK specification compliant
- Direct ParaView/VisIt compatibility

---

## Conclusion

**M9 I/O and Output Management is SUBSTANTIALLY COMPLETE** due to excellent existing design:

- ✅ Clear separation of concerns (7 distinct patterns)
- ✅ Unified ResultsWriter interface with consistent patterns
- ✅ Comprehensive checkpoint/restart with safety guarantees
- ✅ Standard visualization output (VTK) with MPI awareness
- ✅ JSON-driven writer catalog with extension capability
- ✅ Green baseline maintained (30/30 tests passing)

**Recommendation:** Mark M9 as substantially complete, proceed to M10 (Configuration and UI)

---

**Analysis Completed:** 2026-08-01  
**Baseline Status:** 30/30 tests passing (100% success rate)  
**Next Milestone:** M10 - Configuration and UI