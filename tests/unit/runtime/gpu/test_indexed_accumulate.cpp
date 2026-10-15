// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/execution/indexed_accumulate.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/indexed_accumulate_gpu.hpp>

using Catch::Approx;
using pfc::backend::CPUTag;
using pfc::core::BinMap;
using pfc::core::DataBuffer;

namespace {

#if defined(OPENPFC_TEST_INDEXED_CUDA)
using DeviceTag = pfc::backend::CUDATag;
bool device_available() { return pfc::gpu::test::is_cuda_available(); }
#elif defined(OPENPFC_TEST_INDEXED_HIP)
using DeviceTag = pfc::backend::HIPTag;
bool device_available() { return pfc::gpu::test::is_hip_available(); }
#endif

template <typename Tag, typename T>
DataBuffer<Tag, T> upload(const std::vector<T> &v) {
  DataBuffer<Tag, T> b(v.size());
  b.copy_from_host(v);
  return b;
}

/// Many entries into few bins: every bin is hit by hundreds of threads, and
/// most source cells appear in several entries (both atomic paths collide).
struct Case {
  std::vector<std::size_t> source;
  std::vector<std::size_t> bin;
  std::size_t cells = 0;
  std::size_t bins = 0;
};

Case colliding_case(std::size_t entries, std::size_t cells, std::size_t bins) {
  Case c;
  c.cells = cells;
  c.bins = bins;
  for (std::size_t i = 0; i < entries; ++i) {
    c.source.push_back((i * 7919U) % cells);
    c.bin.push_back((i * 31U + i / 5U) % bins);
  }
  return c;
}

template <typename T> std::vector<T> values(std::size_t n, T scale) {
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = scale * static_cast<T>(static_cast<int>(i % 13) - 6);
  }
  return v;
}

} // namespace

TEMPLATE_TEST_CASE("device indexed accumulation matches the host path",
                   "[gpu][indexed_accumulate]", float, double) {
  if (!device_available()) {
    SKIP("no device available");
  }
  using T = TestType;
  const Case c = colliding_case(20000, 1031, 7);
  const BinMap<CPUTag> host_map(c.source, c.bin);
  const BinMap<DeviceTag> dev_map(c.source, c.bin);
  REQUIRE(dev_map.source_end() == host_map.source_end());
  REQUIRE(dev_map.bin_end() == host_map.bin_end());

  // Small integers keep every partial sum exact, so atomic order cannot
  // change the result: device and host must agree bit for bit.
  const std::vector<T> w = values<T>(c.source.size(), T(1));
  const std::vector<T> in = values<T>(c.cells, T(2));
  const std::vector<T> g = values<T>(c.bins, T(3));
  std::vector<std::complex<T>> z(c.cells);
  for (std::size_t k = 0; k < c.cells; ++k) {
    z[k] = {static_cast<T>(static_cast<int>(k % 5) - 2),
            static_cast<T>(static_cast<int>(k % 3) - 1)};
  }
  const std::vector<T> out0(c.bins, T(10));
  const std::vector<T> grad0(c.cells, T(-1));

  auto h_w = upload<CPUTag>(w);
  auto d_w = upload<DeviceTag>(w);

  SECTION("scatter_add") {
    auto h_out = upload<CPUTag>(out0);
    auto d_out = upload<DeviceTag>(out0);
    pfc::core::scatter_add(h_out, upload<CPUTag>(in), host_map, h_w);
    pfc::core::scatter_add(d_out, upload<DeviceTag>(in), dev_map, d_w);
    REQUIRE(d_out.to_host() == h_out.as_vector());
  }
  SECTION("scatter_add_abs2") {
    auto h_out = upload<CPUTag>(out0);
    auto d_out = upload<DeviceTag>(out0);
    pfc::core::scatter_add_abs2(h_out, upload<CPUTag>(z), host_map, h_w);
    pfc::core::scatter_add_abs2(d_out, upload<DeviceTag>(z), dev_map, d_w);
    REQUIRE(d_out.to_host() == h_out.as_vector());
  }
  SECTION("gather_add") {
    auto h_grad = upload<CPUTag>(grad0);
    auto d_grad = upload<DeviceTag>(grad0);
    pfc::core::gather_add(h_grad, upload<CPUTag>(g), host_map, h_w);
    pfc::core::gather_add(d_grad, upload<DeviceTag>(g), dev_map, d_w);
    REQUIRE(d_grad.to_host() == h_grad.as_vector());
  }
}

TEST_CASE("device scatter_add with non-integer weights agrees to rounding",
          "[gpu][indexed_accumulate]") {
  if (!device_available()) {
    SKIP("no device available");
  }
  const Case c = colliding_case(5000, 257, 3);
  std::vector<double> w(c.source.size());
  std::vector<double> in(c.cells);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.1 + 1e-3 * (i % 97);
  for (std::size_t k = 0; k < in.size(); ++k) in[k] = 1.0 / (1.0 + k);
  auto h_out = upload<CPUTag>(std::vector<double>(c.bins, 0.0));
  auto d_out = upload<DeviceTag>(std::vector<double>(c.bins, 0.0));
  pfc::core::scatter_add(h_out, upload<CPUTag>(in), BinMap<CPUTag>(c.source, c.bin),
                         upload<CPUTag>(w));
  pfc::core::scatter_add(d_out, upload<DeviceTag>(in),
                         BinMap<DeviceTag>(c.source, c.bin), upload<DeviceTag>(w));
  const std::vector<double> dev = d_out.to_host();
  for (std::size_t b = 0; b < c.bins; ++b) {
    REQUIRE(dev[b] == Approx(h_out[b]).epsilon(1e-12));
  }
}

TEST_CASE("device indexed accumulation: empty map and range checks",
          "[gpu][indexed_accumulate]") {
  if (!device_available()) {
    SKIP("no device available");
  }
  const BinMap<DeviceTag> empty(std::vector<std::size_t>{},
                                std::vector<std::size_t>{});
  const DataBuffer<DeviceTag, double> none(0);
  auto out = upload<DeviceTag>(std::vector<double>{4.0, 5.0});
  pfc::core::scatter_add(out, none, empty, none);
  pfc::core::gather_add(out, none, empty, none);
  REQUIRE(out.to_host() == std::vector<double>{4.0, 5.0});

  const BinMap<DeviceTag> map(std::vector<std::size_t>{0, 3},
                              std::vector<std::size_t>{0, 2});
  const auto w = upload<DeviceTag>(std::vector<double>{1.0, 1.0});
  const auto in = upload<DeviceTag>(std::vector<double>{1.0, 1.0, 1.0, 1.0});
  REQUIRE_THROWS_AS(pfc::core::scatter_add(out, in, map, w), std::out_of_range);
  REQUIRE(out.to_host() == std::vector<double>{4.0, 5.0});
  const auto short_w = upload<DeviceTag>(std::vector<double>{1.0});
  auto big = upload<DeviceTag>(std::vector<double>{0.0, 0.0, 0.0});
  REQUIRE_THROWS_AS(pfc::core::scatter_add(big, in, map, short_w),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::core::gather_add(big, big, map, w), std::out_of_range);
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
