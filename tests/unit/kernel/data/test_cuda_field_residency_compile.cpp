// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>

// Compile-only test: verify CUDA device residency code compiles.
// No runtime execution required – the test passes if this TU compiles cleanly.
namespace {

struct CompileCheck {
    CompileCheck() {
        // Small size to compile quickly; no device execution needed.
        using namespace pfc;

        // Create a 10x1x1 domain
        auto domain = domain::create(GridSize({10, 1, 1}),
                                     PhysicalOrigin({0.0, 0.0, 0.0}),
                                     GridSpacing({1.0, 1.0, 1.0}));

        // Create a 10x1x1 local box (0 to 9 on x-axis, single point on y and z)
        auto local_box = Box3i::from_bounds({0, 0, 0}, {9, 0, 0});

        // Instantiate a small CUDA-backed field with no halo
        data::Field<double, pfc::CUDASpace> field(domain, local_box, 0);

        // Call device-specific methods to pull CUDA residency branch into compilation.
        field.sync_to_device();
        field.note_device_write();
        const data::Residency &residency = field.residency();

        // Store to volatile to prevent dead-code elimination.
        volatile auto r = &residency;
        (void)r;
    }
} check;

} // anonymous namespace

#endif // OpenPFC_ENABLE_CUDA
