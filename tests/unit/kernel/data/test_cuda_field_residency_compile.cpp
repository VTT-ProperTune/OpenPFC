// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Compile-only test for CUDA device-residency code paths in pfc::data::Field.
// Instantiates Field<double, CudaSpace> and calls device-residency methods
// to force the CUDA code path into the compilation unit. No execution required.

#if defined(OpenPFC_ENABLE_CUDA)

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

namespace {
// Force compilation of the device-residency branch in Field<T, CudaSpace>.
void compile_device_residency() {
  using namespace pfc;

  // Instantiate a small CUDA-backed field.
  data::Field<double, pfc::CudaSpace> field(domain::create({2, 2, 2}),
                                            Box3i::from_bounds({0, 0, 0}, {1, 1, 1}),
                                            0);

  // Call device-residency methods to force the corresponding code paths
  // into the compilation unit. No device execution is required.
  field.sync_to_device();
  field.note_device_write();

  // Access residency state to ensure the device-side implementation is pulled in.
  [[maybe_unused]] const auto &residency = field.residency();
}
} // namespace

// The test itself validates device-residency code compiles correctly by
// ensuring the translation unit builds without errors.
TEST_CASE("CUDA device-residency code compiles", "[cuda][compile-only]") {
  compile_device_residency();
}

#endif // OpenPFC_ENABLE_CUDA
