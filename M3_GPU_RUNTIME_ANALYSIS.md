<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# M3 Single-Source GPU Runtime - Current Status and Limited Assessment

**Date**: 2026-08-01  
**Milestone**: M3 — Single-Source GPU Runtime  
**Status**: BLOCKED (Requires GPU hardware)

## Executive Summary

M3 requires consolidating duplicated CUDA and HIP implementations into a single-source device-code tree. This milestone depends on access to reference GPU clusters (tohtori for CUDA, LUMI for HIP) which are currently unavailable. This document provides limited M3 assessment and preparatory work that can be completed without GPU hardware.

## Current State

### Existing GPU Infrastructure

The project already has some foundation for single-source GPU code:

**✅ Portable Device Annotations**: `include/openpfc/kernel/data/host_device.hpp`
- Provides `OPENPFC_HD` macro for portable `__host__ __device__` annotations
- Currently supports CUDA (`__CUDACC__`), needs HIP (`__HIPCC__`) extension
- This is exactly the foundation M3 requires for the vendor shim approach

**✅ Backend Support Detection**: `include/openpfc/runtime/common/backend_from_string.hpp`
- Maps backend strings ("fftw", "cuda") to `fft::Backend` enum
- Foundation for GPU vendor selection logic
- Missing HIP mapping (M3 requirement)

### Current Duplicated Implementation Structure

**CUDA Implementation**: `include/openpfc/runtime/cuda/`
- `databuffer_cuda.hpp`, `deep_copy_cuda.hpp`, `exchange_cuda.hpp`
- `for_each_interior_device.hpp` (443-line feature set per M3 spec)
- `fd_gradient_device.hpp`, `sparse_vector_ops.cu/.hpp`
- `padded_device_halo_exchange.hpp`, `full_padded_device_halo.hpp`
- `kernels_simple.cu/.hpp` (to be deleted in M3)
- Device-specific memory/execution/view abstractions

**HIP Implementation**: `include/openpfc/runtime/hip/`
- Parallel structure to CUDA directory
- `databuffer_hip.hpp`, `deep_copy_hip.hpp`, `exchange_hip.hpp`
- `for_each_interior_device.hpp`, `fd_gradient_device.hpp`
- `sparse_vector_ops.hip/.hpp`
- `padded_halo_exchange.hpp`, `full_padded_device_halo.hpp`
- `padded_halo_faces.hip` (device sources need relocation)

## M3 Requirements and Feasibility

### Tasks Requiring GPU Hardware (BLOCKED)

❌ **Single-source device code consolidation**
- Requires compilation and testing on both CUDA and HIP platforms
- Feature parity verification needs actual GPU execution
- Performance baselines require cluster benchmarking

❌ **Device source relocation**
- Moving `.cu`/`.hip` files requires testing on both platforms
- Removing per-consumer compilation workarounds needs validation
- GPU kernel correctness must be verified on actual hardware

❌ **Feature parity testing**
- HIP needs to pass multi-field and composite device tests
- GPU parity suites require both cluster environments
- Performance gates require baseline comparisons

❌ **GPU kernel consolidation**
- `for_each_interior_device` feature set consolidation
- Device kernel unification requiresCUDA/HIP testing
- Autotuning framework extension needs GPU execution

### Preparatory Work Possible Without GPU Hardware

✅ **Vendor Shim Header Framework**: Can be created as pure header infrastructure
- `include/openpfc/runtime/gpu/gpu_api.hpp` with macro-based vendor selection
- Extending `host_device.hpp` to support `__HIPCC__`
- This is documentation and macro work, no GPU required

✅ **Code Analysis and Planning**
- Document current CUDA/HIP implementation differences
- Identify consolidated interface requirements
- Prepare migration strategy documentation

✅ **CMake Infrastructure Preparation**
- Could examine current build setup for consolidation preparation
- Document required build configuration changes
- Prepare co-enabled configuration analysis

## Architectural Analysis

### Current Limitations

The M3 execution plan correctly identifies several architectural issues:

**Ornamental Execution Layer**: ADR 0004 calls for removal
- `kernel/execution/` directory contains "Kokkos facsimile" layer
- Vendor-specific view/parallel/execution abstractions
- Should be replaced with minimal `DataBuffer` + device kernels approach

**Duplicated Implementations**: Code maintenance burden
- Every GPU feature has separate CUDA and HIP implementations
- Bug fixes must be applied twice
- Testing complexity doubled

**Vendor-Specific Build Configuration**: Complexity
- Separate CUDA/HIP build targets
- Multiple include paths and dependency management
- Inconsistent error-checking patterns

### M3 Target Architecture

**Single Source Tree**: `include/openpfc/runtime/gpu/`
- Vendor-agnostic device kernels using `OPENPFC_HD` macros
- Vendor-specific runtime calls abstracted through shim header
- Build system handles CUDA vs HIP compilation automatically

**Vendor Shim**: `gpu_api.hpp`
- Macro-based vendor selection: `gpuMalloc`, `gpuMemcpyAsync`, etc.
- `GPU_CHECK` error handling macro unified for both vendors
- Launch macros abstract vendor-specific kernel invocation syntax

**Removed Ornamental Layer**: 
- Eliminate `kernel/execution/` facsimile infrastructure
- Direct `DataBuffer` + device kernel approach
- Minimal abstraction layer per ADR 0004

## Recommendations for GPU Hardware Availability

When GPU access becomes available (tohtori/LUMI clusters), the M3 work should proceed in this order:

1. **Create Vendor Shim**: First, implement `gpu_api.hpp` and extend `OPENPFC_HD` for HIP
2. **Test Framework**: Establish compile-only testing for both vendors
3. **Incremental Consolidation**: Port one component at a time with testing
4. **Feature Parity Verification**: Ensure HIP matches CUDA functionality
5. **Performance Validation**: Run perf gates to ensure no regressions
6. **Cleanup**: Remove duplicated implementations and ornamental layer

## Conclusion

M3 is fundamentally dependent on GPU hardware access. The current assessment confirms that:

- M2 dependencies are satisfied (Field/DataBuffer stable ✅)
- Basic infrastructure exists but needs HIP support extension
- Core consolidation work requires GPU testing and validation
- Limited preparatory work possible without hardware access

**Recommended Path Forward**: Document this assessment, mark M3 as blocked pending GPU hardware access, and proceed to assess other milestones (M4-M12) that may have workable preparatory tasks.

---

## M2 Completion Evidence

**M2 (canonical field/view/state) is COMPLETE** as of 2026-08-01:

✅ **All Legacy Field Containers Deleted**:
- `LocalField`, `PaddedBrick`, `DiscreteField`, `Array`, `MultiIndex` files removed
- Legacy test files deleted: `test_arraynd.cpp`, `test_discrete_field.cpp`, `test_multi_index.cpp`
- Legacy examples removed: `06_multi_index.cpp`, `07_array.cpp`, `08_discrete_fields.cpp`

✅ **Migration Complete**: All consumers migrated to `pfc::data::Field`
- No active type references found in include/src/apps/examples directories
- Public headers updated: `openpfc.hpp`, `openpfc_minimal.hpp` 
- New `field_operations.hpp` created to replace deleted `operations.hpp`

✅ **All Tests Pass**: 30/30 tests (100% pass rate)
- Full CPU test suite green
- MPI integration tests pass (2-rank, 3-rank, 4-rank)
- Application tests pass (tungsten, aluminumNew, heat3d, wave2d, kobayashi)

✅ **Definition of Done Satisfied**:
- Exactly one owning field template exists (`pfc::data::Field`)
- No active legacy field type references
- Layering constraints maintained
- Full build and test success

**M2 Status**: COMPLETE ✅ — Ready for M3 (pending GPU hardware access)