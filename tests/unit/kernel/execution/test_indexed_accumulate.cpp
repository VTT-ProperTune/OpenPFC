// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/execution/indexed_accumulate.hpp>

using Catch::Approx;
using pfc::backend::CPUTag;
using pfc::core::BinMap;
using pfc::core::DataBuffer;

namespace {

template <typename T> DataBuffer<CPUTag, T> buffer(const std::vector<T> &v) {
  DataBuffer<CPUTag, T> b(v.size());
  b.copy_from_host(v);
  return b;
}

} // namespace

TEST_CASE("BinMap records extents and rejects unequal lists",
          "[core][indexed_accumulate][cpu]") {
  const BinMap<CPUTag> map(std::vector<std::size_t>{4, 0, 2},
                           std::vector<std::size_t>{1, 1, 6});
  REQUIRE(map.size() == 3);
  REQUIRE(map.source_end() == 5);
  REQUIRE(map.bin_end() == 7);

  const BinMap<CPUTag> none(std::vector<std::size_t>{}, std::vector<std::size_t>{});
  REQUIRE(none.empty());
  REQUIRE(none.source_end() == 0);
  REQUIRE(none.bin_end() == 0);

  REQUIRE_THROWS_AS(
      BinMap<CPUTag>(std::vector<std::size_t>{0, 1}, std::vector<std::size_t>{0}),
      std::invalid_argument);
}

TEST_CASE("scatter_add sums duplicate bins and keeps prior contents",
          "[core][indexed_accumulate][cpu]") {
  // Cells 0..4 into bins {0, 1}; bin 1 receives three entries, cell 2 twice.
  const BinMap<CPUTag> map(std::vector<std::size_t>{0, 2, 3, 2, 4},
                           std::vector<std::size_t>{0, 1, 1, 0, 1});
  const auto in = buffer<double>({1.0, 10.0, 2.0, 3.0, 4.0});
  const auto w = buffer<double>({1.0, 0.5, 2.0, 1.0, 0.25});
  auto out = buffer<double>({100.0, 200.0, 7.0});
  pfc::core::scatter_add(out, in, map, w);
  REQUIRE(out[0] == Approx(100.0 + 1.0 * 1.0 + 1.0 * 2.0));
  REQUIRE(out[1] == Approx(200.0 + 0.5 * 2.0 + 2.0 * 3.0 + 0.25 * 4.0));
  REQUIRE(out[2] == 7.0);
}

TEST_CASE("scatter_add_abs2 bins a power spectrum",
          "[core][indexed_accumulate][cpu]") {
  using C = std::complex<double>;
  const BinMap<CPUTag> map(std::vector<std::size_t>{0, 1, 2, 1},
                           std::vector<std::size_t>{0, 0, 1, 1});
  const auto in = buffer<C>({C{1.0, 2.0}, C{0.0, -3.0}, C{4.0, 0.0}});
  const auto w = buffer<double>({1.0, 2.0, 0.5, 1.0});
  auto out = buffer<double>({0.0, 0.0});
  pfc::core::scatter_add_abs2(out, in, map, w);
  REQUIRE(out[0] == Approx(5.0 + 2.0 * 9.0));
  REQUIRE(out[1] == Approx(0.5 * 16.0 + 9.0));
}

TEST_CASE("gather_add is the adjoint of scatter_add",
          "[core][indexed_accumulate][cpu]") {
  const BinMap<CPUTag> map(std::vector<std::size_t>{0, 2, 3, 2, 4},
                           std::vector<std::size_t>{0, 1, 1, 0, 1});
  const auto w = buffer<double>({1.0, 0.5, 2.0, 1.0, 0.25});
  const auto x = buffer<double>({0.3, -1.0, 2.0, 0.7, -0.4});
  const auto g = buffer<double>({1.5, -2.5});

  auto sx = buffer<double>({0.0, 0.0});
  pfc::core::scatter_add(sx, x, map, w);
  auto gty = buffer<double>({0.0, 0.0, 0.0, 0.0, 0.0});
  pfc::core::gather_add(gty, g, map, w);

  // <S x, g> == <x, S^T g>
  const double lhs = sx[0] * g[0] + sx[1] * g[1];
  double rhs = 0.0;
  for (std::size_t i = 0; i < 5; ++i) rhs += x[i] * gty[i];
  REQUIRE(lhs == Approx(rhs));
  REQUIRE(gty[1] == 0.0); // cell 1 is in no bin
  REQUIRE(gty[2] == Approx(0.5 * -2.5 + 1.0 * 1.5));
}

TEST_CASE("empty bin map leaves output untouched",
          "[core][indexed_accumulate][cpu]") {
  const BinMap<CPUTag> map;
  const DataBuffer<CPUTag, float> in(0);
  const DataBuffer<CPUTag, float> w(0);
  auto out = buffer<float>({3.0F});
  pfc::core::scatter_add(out, in, map, w);
  pfc::core::gather_add(out, in, map, w);
  REQUIRE(out[0] == 3.0F);
}

TEST_CASE("indexed accumulation rejects out-of-range maps before writing",
          "[core][indexed_accumulate][cpu]") {
  const BinMap<CPUTag> map(std::vector<std::size_t>{0, 3},
                           std::vector<std::size_t>{0, 2});
  const auto w = buffer<double>({1.0, 1.0});
  auto out = buffer<double>({0.0, 0.0, 0.0});
  const auto small_in = buffer<double>({1.0, 1.0, 1.0});
  REQUIRE_THROWS_AS(pfc::core::scatter_add(out, small_in, map, w),
                    std::out_of_range);
  auto small_out = buffer<double>({0.0, 0.0});
  const auto in = buffer<double>({1.0, 1.0, 1.0, 1.0});
  REQUIRE_THROWS_AS(pfc::core::scatter_add(small_out, in, map, w),
                    std::out_of_range);
  REQUIRE(small_out[0] == 0.0);
  const auto short_w = buffer<double>({1.0});
  REQUIRE_THROWS_AS(pfc::core::scatter_add(out, in, map, short_w),
                    std::invalid_argument);
  auto small_grad = buffer<double>({0.0, 0.0, 0.0});
  REQUIRE_THROWS_AS(pfc::core::gather_add(small_grad, out, map, w),
                    std::out_of_range);
}
