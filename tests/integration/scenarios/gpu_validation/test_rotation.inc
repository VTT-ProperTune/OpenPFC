// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/geometry/rotation.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

#include <cmath>
#include <numbers>
#include <type_traits>
#include <vector>

namespace {
using pfc::geometry::Rotation;
using pfc::geometry::Vec3;
struct RotationResult {
  Vec3 applied, restored, shared, identity;
};
static_assert(std::is_trivially_copyable_v<Rotation>);

__global__ void rotate_cells(const Rotation *rotations, Rotation shared,
                             RotationResult *out, int count) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= count) return;
  const Vec3 input{double(q + 1), -2.0, 0.5};
  const auto rotation = rotations[q];
  const auto applied = rotation.apply(input);
  out[q] = {applied, rotation.apply_transpose(applied), shared.apply(input),
            pfc::geometry::identity_rotation().apply(input)};
}
struct DeviceMemory {
  void *ptr = nullptr;
  ~DeviceMemory() {
    if (ptr) (void)pfc::gpuFree(ptr);
  }
};
} // namespace

TEST_CASE("host-validated rotations act on device cells", "[rotation][device]") {
  int count = 0;
#if defined(OPENPFC_GPU_API_HIP)
  const auto status = hipGetDeviceCount(&count);
#else
  const auto status = cudaGetDeviceCount(&count);
#endif
  if (status != pfc::gpuSuccess || count == 0) SKIP("No allocated device");
  constexpr int n = 33;
  const auto shared = pfc::geometry::rotation_about({0, 0, 1}, std::numbers::pi / 2);
  std::vector<Rotation> rotations(n);
  for (int q = 0; q < n; ++q) {
    rotations[q] =
        q % 2 == 0 ? shared : pfc::geometry::rotation_about({1, 2, 3}, 0.13 * q);
  }
  DeviceMemory device_rotations, device_output;
  REQUIRE(pfc::gpuMalloc(&device_rotations.ptr, n * sizeof(Rotation)) ==
          pfc::gpuSuccess);
  REQUIRE(pfc::gpuMalloc(&device_output.ptr, n * sizeof(RotationResult)) ==
          pfc::gpuSuccess);
  REQUIRE(pfc::gpuMemcpy(device_rotations.ptr, rotations.data(),
                         n * sizeof(Rotation),
                         pfc::gpuMemcpyHostToDevice) == pfc::gpuSuccess);
  rotate_cells<<<1, 64>>>(static_cast<Rotation *>(device_rotations.ptr), shared,
                          static_cast<RotationResult *>(device_output.ptr), n);
  REQUIRE(pfc::gpuGetLastError() == pfc::gpuSuccess);
  std::vector<RotationResult> got(n);
  REQUIRE(pfc::gpuMemcpy(got.data(), device_output.ptr, n * sizeof(RotationResult),
                         pfc::gpuMemcpyDeviceToHost) == pfc::gpuSuccess);
  for (int q = 0; q < n; ++q) {
    const Vec3 input{double(q + 1), -2, 0.5};
    const auto reference = rotations[q].apply(input);
    CHECK(pfc::geometry::norm(got[q].applied - reference) < 1e-12);
    CHECK(pfc::geometry::norm(got[q].restored - input) < 1e-12);
    CHECK(pfc::geometry::norm(got[q].shared - Vec3{2, double(q + 1), 0.5}) < 1e-12);
    CHECK(pfc::geometry::norm(got[q].identity - input) == 0);
  }
  const double reflection[9] = {-1, 0, 0, 0, 1, 0, 0, 0, 1};
  REQUIRE_THROWS_AS(pfc::geometry::rotation_from_matrix(reflection),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::geometry::rotation_about({0, 0, 0}, 1),
                    std::invalid_argument);
}
