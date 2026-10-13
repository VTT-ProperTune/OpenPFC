// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Device reflection against the host fill. One upload, then an in-place
 * kernel. The comparison pulls the device buffer back; the fill itself
 * does not.
 */

#include "test_helpers.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>

#if defined(OPENPFC_TEST_REFLECTION_CUDA) || defined(OPENPFC_TEST_REFLECTION_HIP)
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/reflection_ghosts_gpu.hpp>
#endif

using pfc::communication::apply_reflection_ghosts;
using pfc::communication::BoundaryCentering;
using pfc::communication::reflection_walls;
using pfc::communication::ReflectionParity;

namespace {

template <typename T> void fill_host(pfc::data::Field<T, pfc::HostSpace> &u) {
  const int hw = u.storage_halo();
  const auto n = u.local_size();
  for (int k = -hw; k < n[2] + hw; ++k) {
    for (int j = -hw; j < n[1] + hw; ++j) {
      for (int i = -hw; i < n[0] + hw; ++i) {
        const bool owned =
            i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
        if (owned) {
          const auto g = u.global(i, j, k);
          u(i, j, k) = static_cast<T>(0.5) * static_cast<T>(g[0]) -
                       static_cast<T>(1.25) * static_cast<T>(g[1]) +
                       static_cast<T>(0.125) * static_cast<T>(g[2]);
        } else {
          u(i, j, k) = static_cast<T>(-1234567.0);
        }
      }
    }
  }
}

template <typename DeviceField> void require_device_matches_host() {
  using T = typename DeviceField::value_type;
  auto domain = pfc::domain::create(
      pfc::GridSize({8, 6, 5}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, false, false});
  const auto box = pfc::domain::index_box(domain);
  constexpr int hw = 3;
  pfc::data::Field<T, pfc::HostSpace> host(domain, box, hw);
  fill_host(host);
  DeviceField device(domain, box, hw);
  device.with_host_view([&](T *dst, std::size_t n) {
    REQUIRE(n == host.size());
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = host.data()[i];
    }
  });

  const auto walls = reflection_walls<T>(
      ReflectionParity::Odd, BoundaryCentering::Node, static_cast<T>(1.5));
  apply_reflection_ghosts(host, walls);
  apply_reflection_ghosts(device, walls);
  REQUIRE(device.residency().device_valid());
  REQUIRE_FALSE(device.residency().host_valid());

  device.with_host_read([&](const T *src, std::size_t n) {
    REQUIRE(n == host.size());
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(src[i] == host.data()[i]);
    }
  });

  const auto cell =
      reflection_walls<T>(ReflectionParity::Even, BoundaryCentering::CellFace, T{});
  apply_reflection_ghosts(host, cell);
  apply_reflection_ghosts(device, cell);
  device.with_host_read([&](const T *src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(src[i] == host.data()[i]);
    }
  });

  auto linear = reflection_walls<T>(
      ReflectionParity::Odd, BoundaryCentering::CellFace, static_cast<T>(1.5));
  for (auto *side :
       {&linear.xm, &linear.xp, &linear.ym, &linear.yp, &linear.zm, &linear.zp})
    side->extension = pfc::communication::BoundaryExtension::Linear;
  apply_reflection_ghosts(host, linear);
  apply_reflection_ghosts(device, linear);
  REQUIRE(device.residency().device_valid());
  REQUIRE_FALSE(device.residency().host_valid());
  std::vector<T> accepted;
  device.with_host_read([&](const T *src, std::size_t n) {
    accepted.assign(src, src + n);
    for (std::size_t i = 0; i < n; ++i) REQUIRE(src[i] == host.data()[i]);
  });
  // Independent tensor-product Lagrange weights: (-3)^3*u + 6*(9-3+1).
  REQUIRE(host(-2, -2, -2) == static_cast<T>(42));
  REQUIRE(host(9, 7, 6) == static_cast<T>(102.75));
  linear.xm.centering = BoundaryCentering::Node;
  REQUIRE_THROWS_AS(apply_reflection_ghosts(device, linear), std::invalid_argument);
  device.with_host_read([&](const T *src, std::size_t n) {
    REQUIRE(std::vector<T>(src, src + n) == accepted);
  });
}

} // namespace

#if defined(OPENPFC_TEST_REFLECTION_CUDA)
TEST_CASE("CUDA reflection ghosts match the host fill", "[gpu][reflection][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  require_device_matches_host<pfc::data::Field<double, pfc::CUDASpace>>();
  require_device_matches_host<pfc::data::Field<float, pfc::CUDASpace>>();
}
#endif

#if defined(OPENPFC_TEST_REFLECTION_HIP)
TEST_CASE("HIP reflection ghosts match the host fill", "[gpu][reflection][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  require_device_matches_host<pfc::data::Field<double, pfc::HIPSpace>>();
  require_device_matches_host<pfc::data::Field<float, pfc::HIPSpace>>();
}
#endif

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
