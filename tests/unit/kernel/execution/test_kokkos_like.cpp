// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/execution/create_mirror.hpp>
#include <openpfc/kernel/execution/deep_copy.hpp>
#include <openpfc/kernel/execution/execution_space.hpp>
#include <openpfc/kernel/execution/layout.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/execution/parallel.hpp>
#include <openpfc/kernel/execution/policy.hpp>
#include <openpfc/kernel/execution/view.hpp>

using Catch::Approx;

TEST_CASE("Execution and memory space tags exist", "[kokkos_like][core]") {
  REQUIRE(sizeof(pfc::Serial) >= 0);
  REQUIRE(sizeof(pfc::OpenMP) >= 0);
  REQUIRE(sizeof(pfc::HostSpace) >= 0);
  REQUIRE(sizeof(pfc::DefaultExecutionSpace) >= 0);
  REQUIRE(sizeof(pfc::DefaultMemorySpace) >= 0);
}

TEST_CASE("Layout stride computation", "[kokkos_like][core]") {
  std::array<std::size_t, 3> extents{{4, 5, 6}};
  auto strides_r = pfc::strides_from_extents<pfc::LayoutRight>(extents);
  REQUIRE(strides_r[2] == 1);
  REQUIRE(strides_r[1] == 6);
  REQUIRE(strides_r[0] == 30);
  auto strides_l = pfc::strides_from_extents<pfc::LayoutLeft>(extents);
  REQUIRE(strides_l[0] == 1);
  REQUIRE(strides_l[1] == 4);
  REQUIRE(strides_l[2] == 20);
}

TEST_CASE("View 3D construction and access", "[kokkos_like][core]") {
  pfc::View<double, 3, pfc::LayoutRight, pfc::HostSpace> v("v", 2, 3, 4);
  REQUIRE(v.rank() == 3);
  REQUIRE(v.extent(0) == 2);
  REQUIRE(v.extent(1) == 3);
  REQUIRE(v.extent(2) == 4);
  REQUIRE(v.size() == 24);
  REQUIRE(v.span() == 24);
  REQUIRE(v.stride(2) == 1);
  REQUIRE(v.data() != nullptr);
  REQUIRE(v.is_managed());

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 4; ++k) v(i, j, k) = 100.0 * i + 10.0 * j + k;

  REQUIRE(v(0, 0, 0) == Approx(0.0));
  REQUIRE(v(1, 2, 3) == Approx(123.0));
}

TEST_CASE("View 1D and 2D", "[kokkos_like][core]") {
  pfc::View<double, 1, pfc::LayoutRight, pfc::HostSpace> v1("v1", 10);
  REQUIRE(v1.size() == 10);
  v1(5) = 42.0;
  REQUIRE(v1(5) == Approx(42.0));

  pfc::View<double, 2, pfc::LayoutRight, pfc::HostSpace> v2("v2", 3, 4);
  REQUIRE(v2.size() == 12);
  v2(1, 2) = 7.0;
  REQUIRE(v2(1, 2) == Approx(7.0));
}

TEST_CASE("View unmanaged constructor", "[kokkos_like][core]") {
  std::vector<double> storage(6, 0.0);
  pfc::View<double, 2, pfc::LayoutRight, pfc::HostSpace> v(storage.data(), 2, 3);
  REQUIRE(!v.is_managed());
  REQUIRE(v.size() == 6);
  v(1, 1) = 99.0;
  REQUIRE(storage[4] == Approx(99.0)); // LayoutRight: (1,1) -> 1*3+1
}

TEST_CASE("RangePolicy and parallel_for", "[kokkos_like][core]") {
  pfc::View<double, 1, pfc::LayoutRight, pfc::HostSpace> v("v", 100);
  pfc::deep_copy(v, 0.0);
  auto policy = pfc::RangePolicy<pfc::Serial>(0, 100);
  pfc::parallel_for(policy, [&v](std::size_t i) { v(i) = static_cast<double>(i); });
  for (std::size_t i = 0; i < 100; ++i)
    REQUIRE(v(i) == Approx(static_cast<double>(i)));
}

TEST_CASE("MDRangePolicy 3D and parallel_for", "[kokkos_like][core]") {
  pfc::View<double, 3, pfc::LayoutRight, pfc::HostSpace> v("v", 4, 5, 6);
  auto policy = pfc::MDRangePolicy<pfc::Serial, pfc::Rank<3>>(0, 4, 0, 5, 0, 6);
  pfc::parallel_for(policy, [&v](std::size_t i, std::size_t j, std::size_t k) {
    v(i, j, k) = static_cast<double>(i * 100 + j * 10 + k);
  });
  REQUIRE(v(1, 2, 3) == Approx(123.0));
}

TEST_CASE("fence compiles and runs", "[kokkos_like][core]") {
  pfc::fence();
  pfc::fence(pfc::Serial{});
}

TEST_CASE("deep_copy View to View host", "[kokkos_like][core]") {
  pfc::View<double, 3, pfc::LayoutRight, pfc::HostSpace> a("a", 2, 3, 4);
  pfc::View<double, 3, pfc::LayoutRight, pfc::HostSpace> b("b", 2, 3, 4);
  for (std::size_t i = 0; i < a.size(); ++i) a.data()[i] = static_cast<double>(i);
  pfc::deep_copy(b, a);
  for (std::size_t i = 0; i < a.size(); ++i)
    REQUIRE(b.data()[i] == Approx(static_cast<double>(i)));
}

TEST_CASE("deep_copy scalar fill", "[kokkos_like][core]") {
  pfc::View<double, 2, pfc::LayoutRight, pfc::HostSpace> v("v", 3, 4);
  pfc::deep_copy(v, 3.14);
  for (std::size_t i = 0; i < v.size(); ++i) REQUIRE(v.data()[i] == Approx(3.14));
}

TEST_CASE("create_mirror and create_mirror_view", "[kokkos_like][core]") {
  pfc::View<double, 3, pfc::LayoutRight, pfc::HostSpace> src("src", 2, 3, 4);
  src(0, 0, 0) = 1.0;
  src(1, 2, 3) = 2.0;

  auto mirror = pfc::create_mirror(src);
  REQUIRE(mirror.extent(0) == 2);
  REQUIRE(mirror.extent(1) == 3);
  REQUIRE(mirror.extent(2) == 4);
  pfc::deep_copy(mirror, src);
  REQUIRE(mirror(0, 0, 0) == Approx(1.0));
  REQUIRE(mirror(1, 2, 3) == Approx(2.0));

  auto mirror_view = pfc::create_mirror_view(src);
  REQUIRE(mirror_view.data() == src.data());
  REQUIRE(mirror_view(0, 0, 0) == Approx(1.0));
}
