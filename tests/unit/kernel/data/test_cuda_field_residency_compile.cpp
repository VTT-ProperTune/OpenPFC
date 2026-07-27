// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>

// Compile-only test: verify CUDA device residency code compiles.
// No runtime execution required – the test passes if this TU compiles cleanly.
namespace {

struct CompileCheck {
  CompileCheck() {
    using namespace pfc;

    // Small size to compile quickly; no device execution needed.
    Int3 extents = {10, 1, 1};

    // Instantiate a small CUDA-backed field.
    data::Field<double, pfc::CudaSpace> field(domain::create(extents),
                                              Box3i::from_bounds({0, 0, 0}, {9, 0, 0}),
                                              0);

    // Call device-specific methods to pull CUDA residency branch into compilation.
    field.sync_to_device();
    field.note_device_write();
    const auto &residency = field.residency();

    // Store to volatile to prevent dead-code elimination.
    volatile auto r = residency;
    (void)r;
  }
} check;

} // anonymous namespace

#endif // OpenPFC_ENABLE_CUDA
