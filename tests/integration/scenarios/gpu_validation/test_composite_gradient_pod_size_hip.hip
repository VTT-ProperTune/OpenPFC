// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_composite_gradient_pod_size_hip.hip
 * @brief HIP twin of `test_composite_gradient_pod_size.cu`.
 *
 * Locks `sizeof(CompositeGradientDevicePOD)` to the documented layout via
 * the HIP re-export (`pfc::hip`). Compiling under HIP is an M3 requirement;
 * executing on a GPU is M-LUMI (this case is host-side sizeof only).
 */

#include <catch2/catch_test_macros.hpp>

#if defined(OpenPFC_ENABLE_HIP)

#include <openpfc/runtime/hip/fd_gradient_device.hpp>

TEST_CASE("test_composite_gradient_pod_size_hip",
          "[hip][fd_gradient_device][multi-field][integration]") {
  CHECK(sizeof(pfc::hip::CompositeGradientDevicePOD) == 2088);
  CHECK(pfc::hip::kMaxCompositeFields == 4);
  CHECK(pfc::hip::kFdDeviceMaxHw1 == 7);
  CHECK(pfc::hip::kFdDeviceMaxHw2 == 10);
}

#else // !OpenPFC_ENABLE_HIP

TEST_CASE("test_composite_gradient_pod_size_hip skipped (HIP disabled)",
          "[hip][fd_gradient_device][multi-field][integration]") {
  SUCCEED("Skipping: OpenPFC_ENABLE_HIP is off.");
}

#endif
