<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

All notable changes to OpenPFC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-01

### Added
- **Domain Pattern**: Modern spatial abstraction replacing World pattern with Box3i + Domain architecture
- **Unified Field API**: Modern `Field<T, MemorySpace>` template-based field container system
- **Single-Source GPU Runtime**: Unified CUDA/HIP implementation with vendor-agnostic GPU API
- **GPU Hardware Support**: Verified on NVIDIA H100 80GB HBM3 with CUDA 13.0, 22/22 GPU tests passing
- **Halo Exchange Consolidation**: Unified communication layer with multi-field batching support
- **Honest FFT Interfaces**: Device-side k-space iteration and GPU spectral methods
- **Unified Stepper Protocol**: Consistent interface across 7 time integration methods
- **Configuration Catalog**: JSON/TOML-driven type registration system for field modifiers
- **Results Writer Catalog**: Unified output format system (Binary, VTK, etc.)
- **Application Modernization**: 4/6 production apps fully migrated to modern architecture patterns

### Changed
- **Breaking**: World pattern deprecated in favor of Domain + Box3i (compatibility shim maintained)
- **Breaking**: Legacy field containers (LocalField, PaddedBrick, DiscreteField, Array) removed
- **GPU Architecture**: Migrated from vendor-specific implementations to single-source runtime/gpu/ directory
- **Field Operations**: Migrated from World-based to Domain-based APIs throughout kernel
- **Simulator Architecture**: Separated concerns between orchestration, simulation, and physics
- **Build System**: Updated to support C++20 with GCC 15.2.0 + OpenMPI 5.0.10 baseline
- **Testing**: Comprehensive test suite with 30/30 tests passing (100% success rate) on CPU baseline

### Deprecated
- **World Pattern**: `get_world()` functions and `World()` constructor deprecated for removal in 0.3.0
- **Model World Constructor**: `Model(fft, World)` constructor deprecated, use `Model(fft, Domain)` instead
- **Legacy Field APIs**: Previous field container interfaces removed from public API

### Removed
- **Legacy Field Containers**: LocalField, PaddedBrick, DiscreteField, Array deleted
- **Vendor-Specific GPU Code**: Separate CUDA/HIP directories consolidated to runtime/gpu/
- **Kokkos Facsimile**: Experimental Kokkos compatibility layer removed
- **Legacy GPU Headers**: gpu_vector.hpp, kernels_simple files, static_assert tombstones deleted

### Fixed
- **Compiler Compatibility**: C++20 compatibility issues resolved across GCC 8.5.0 to 15.2.0
- **GPU Build Issues**: CUDA compilation problems resolved with proper HeFFTe integration
- **Memory Management**: GPU memory allocation and halo exchange optimizations
- **Performance**: Multi-field halo batching reduces communication overhead

### Performance
- **GPU Verification**: 8x NVIDIA H100 80GB verified with comprehensive GPU test suite
- **Communication**: Library-level multi-field halo batching reduces synchronization overhead
- **Memory**: Unified Field<T,MemorySpace> API optimizes memory access patterns
- **Spectral Methods**: Device-side k-space iteration improves GPU spectral performance

### Migration Guide

#### From World to Domain Pattern
The World pattern is deprecated but maintained for backward compatibility. New code should use Domain + Box3i:

```cpp
// Old (deprecated):
const auto &world = pfc::get_world(model);
auto size = pfc::get_size(world);

// New (recommended):
const auto &domain = pfc::get_domain(model);
auto size = domain.size;

// For spatial operations:
Box3i box = pfc::domain::index_box(domain);
```

#### From Legacy Field Containers
Previous container types have been unified into `Field<T, MemorySpace>`:

```cpp
// Old (deprecated):
PaddedBrick<double> field(size);
LocalField<double> local_field(...);

// New (recommended):
using Field = pfc::data::Field<double, pfc::HostSpace>;
Field field = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
```

#### Model Constructor
World-based Model constructors are deprecated:

```cpp
// Old (deprecated):
MyModel model(fft, World({0,0,0}, {N-1,N-1,N-1}, domain));

// New (recommended):
MyModel model(fft, domain);
```

### Documentation
- **Comprehensive Analysis Documents**: M0-M11 detailed analysis documents created
- **Architecture Documentation**: Domain pattern, Field API, GPU runtime thoroughly documented
- **Migration Guides**: Step-by-step guides for World→Domain and legacy field migration
- **GPU Setup**: HPC cluster installation guides for LUMI-G, Tohtori NVIDIA H100

### Test Coverage
- **CPU Tests**: 30/30 tests passing (100% success rate)
- **GPU Tests**: 22/22 tests passing on NVIDIA H100
- **Integration Tests**: Full application stack tested end-to-end
- **Performance Tests**: Scaling and benchmarking suites verified

---

## [0.1.x] - Previous Releases

### Features
- Initial OpenPFC implementation with World spatial abstraction
- Basic spectral and finite difference methods
- CPU-only implementations
- Example applications and tutorials

---

## Migration Timeline

### OpenPFC 0.2.0 (Current)
- ✅ Domain pattern established as canonical
- ✅ Modern Field<T,MemorySpace> API deployed
- ✅ Single-source GPU runtime unified
- ⚠️ **World pattern deprecated but maintained** - full removal planned for 0.3.0

### OpenPFC 0.3.0 (Planned)
- 🔄 Complete World pattern removal
- 🔄 Migrate core infrastructure to Domain-only APIs
- 🔄 Update production apps to eliminate all World usage
- 🔄 Test suite modernization to Domain patterns

---

## Upgrade Notes

### For OpenPFC 0.1.x Users
OpenPFC 0.2.0 represents a major architectural refactor. While we maintain compatibility through the World pattern shim, we recommend:

1. **Review_migration guides** above for API changes
2. **Update Model constructors** to use Domain instead of World
3. **Migrate field containers** to `Field<T, MemorySpace>` API
4. **Test with existing configurations** - most JSON/TOML configs remain compatible
5. **Plan for 0.3.0** - World pattern will be removed in next release

### For New Users
OpenPFC 0.2.0 provides a modern, well-documented architecture. Start with:

- Domain + Box3i pattern for spatial operations
- `Field<T, MemorySpace>` for field management  
- `pfc::ui::App<Model>` for JSON/TOML configuration
- Single-source GPU runtime for CUDA/HIP support

---

## Acknowledgments

### Major Architectural Achievements (M0-M11)
- **M1**: World→Domain consolidation with clean spatial abstraction
- **M2**: Unified Field API eliminating legacy container fragmentation  
- **M3**: Single-source GPU runtime enabling portable CUDA/HIP code
- **M4**: Communication layer consolidation with multi-field batching
- **M5**: Honest FFT interfaces with device-side iteration
- **M6**: Unified stepper protocol across 7 time integration methods
- **M7**: Physics interface consolidation (boundary/initial conditions)
- **M8**: Orchestration pattern verification and documentation
- **M9**: I/O and output management unification
- **M10**: Configuration and UI pattern modernization
- **M11**: Application migration analysis showing 4/6 fully modern apps

### Hardware Verification
- **NVIDIA H100 80GB HBM3**: 8x GPUs verified operational, CUDA 13.0
- **GPU Tests**: 22/22 GPU tests passing, comprehensive coverage
- **Performance**: Multi-field batching optimizations verified

### Community and Testing
- **Test Coverage**: 30/30 CPU tests passing (100% success rate)
- **Documentation**: Comprehensive analysis guides for each milestone
- **Examples**: Updated examples showcasing modern architecture patterns
- **Applications**: 4/6 production apps fully migrated to modern patterns

---

**OpenPFC 0.2.0 represents a major architectural milestone** with modern C++20 design, unified GPU support, and comprehensive refactoring across all subsystems. The project maintains backward compatibility where practical while establishing clear migration paths for future releases.
