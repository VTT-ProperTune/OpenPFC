// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_composite_gradient_pod_size.cu
 * @brief Lock `sizeof(CompositeGradientDevicePOD)` to the documented layout.
 */

#include <catch2/catch_test_macros.hpp>

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/runtime/cuda/fd_gradient_device.hpp>

TEST_CASE("test_composite_gradient_pod_size",
          "[cuda][fd_gradient_device][multi-field][integration]") {
  CHECK(sizeof(pfc::cuda::CompositeGradientDevicePOD) == 2088);
  CHECK(pfc::cuda::kMaxCompositeFields == 4);
  CHECK(pfc::cuda::kFdDeviceMaxHw1 == 7);
  CHECK(pfc::cuda::kFdDeviceMaxHw2 == 10);
}

#else // !OpenPFC_ENABLE_CUDA

TEST_CASE("test_composite_gradient_pod_size skipped (CUDA disabled)",
          "[cuda][fd_gradient_device][multi-field][integration]") {
  SUCCEED("Skipping: OpenPFC_ENABLE_CUDA is off.");
}

#endif
