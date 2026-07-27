// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/field.hpp>

// Compile-only test: verify CUDA device residency code compiles.
// No runtime execution required – the test passes if this TU compiles cleanly.
namespace {

struct CompileCheck {
    CompileCheck() {
        // Small size to compile quickly; no device execution needed.
        std::array<std::size_t, 3> extents = {10, 1, 1};
        pfc::data::Field<double, pfc::CudaSpace> field(extents);

        // Call device-specific methods to pull CUDA residency branch into compilation.
        field.sync_to_device();
        field.note_device_write();
        pfc::MemoryResidency residency = field.residency();

        // Store to volatile to prevent dead-code elimination.
        volatile auto r = residency;
        (void)r;
    }
} check;

} // anonymous namespace

#endif // OpenPFC_ENABLE_CUDA
