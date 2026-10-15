// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"
#include "transform_functors.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/execution/transform.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/transform_gpu.hpp>

using pfc::backend::CPUTag;
using pfc::core::DataBuffer;
using ComplexD = std::complex<double>;

namespace {

#if defined(OPENPFC_TEST_TRANSFORM_CUDA)
using DeviceTag = pfc::backend::CUDATag;
bool device_available() { return pfc::gpu::test::is_cuda_available(); }
#elif defined(OPENPFC_TEST_TRANSFORM_HIP)
using DeviceTag = pfc::backend::HIPTag;
bool device_available() { return pfc::gpu::test::is_hip_available(); }
#endif

template <typename Tag, typename T>
DataBuffer<Tag, T> upload(const std::vector<T> &v) {
  DataBuffer<Tag, T> b(v.size());
  b.copy_from_host(v);
  return b;
}

std::vector<double> ramp(std::size_t n, double a, double b) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = a + b * static_cast<double>(i % 37);
  return v;
}

/// Run the functor on host and device buffers built from the same inputs.
template <typename Out, typename F, typename... In>
void require_parity(std::size_t n, F f, const std::vector<In> &...in) {
  DataBuffer<CPUTag, Out> h_out(n);
  DataBuffer<DeviceTag, Out> d_out(n);
  pfc::core::transform(h_out, f, upload<CPUTag>(in)...);
  pfc::core::transform(d_out, f, upload<DeviceTag>(in)...);
  // Same expression, same IEEE operations; FMA contraction is the only
  // difference allowed, so compare to a tight relative tolerance.
  const std::vector<Out> dev = d_out.to_host();
  for (std::size_t i = 0; i < n; ++i) {
    const auto diff = std::abs(dev[i] - h_out[i]);
    REQUIRE(diff <= 1e-12 * (1.0 + std::abs(h_out[i])));
  }
}

} // namespace

TEST_CASE("device transform matches the host path for 1 to 4 inputs",
          "[gpu][transform]") {
  if (!device_available()) {
    SKIP("no device available");
  }
  const std::size_t n = 4099; // not a multiple of the block size
  const auto a = ramp(n, -1.0, 0.05);
  const auto b = ramp(n, 0.3, -0.02);
  const auto c = ramp(n, 2.0, 0.01);
  const auto d = ramp(n, -0.7, 0.03);
  std::vector<ComplexD> z(n);
  for (std::size_t i = 0; i < n; ++i) z[i] = {a[i], b[i]};

  require_parity<double>(n, pfc::test::Square{}, a);
  require_parity<double>(n, pfc::test::Reaction{0.1, 0.5, -0.2}, a, b, c);
  require_parity<double>(n, pfc::test::Weighted4{}, a, b, c, d);
  require_parity<ComplexD>(n, pfc::test::ScaleComplex{}, z, c);
  require_parity<double>(n, pfc::test::Abs2{}, z);

  std::vector<float> f(n);
  for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(a[i]);
  DataBuffer<DeviceTag, float> d_f(n);
  pfc::core::transform(d_f, pfc::test::Halve{}, upload<DeviceTag>(f));
  const std::vector<float> got = d_f.to_host();
  for (std::size_t i = 0; i < n; ++i) REQUIRE(got[i] == 0.5F * f[i]);
}

TEST_CASE("device transform writes in place and checks sizes", "[gpu][transform]") {
  if (!device_available()) {
    SKIP("no device available");
  }
  auto x = upload<DeviceTag>(std::vector<double>{1.0, -2.0, 3.0});
  pfc::core::transform(x, pfc::test::Square{}, x);
  REQUIRE(x.to_host() == std::vector<double>{1.0, 4.0, 9.0});

  DataBuffer<DeviceTag, double> empty(0);
  pfc::core::transform(empty, pfc::test::Square{}, empty);

  DataBuffer<DeviceTag, double> out(3);
  const auto short_in = upload<DeviceTag>(std::vector<double>{1.0, 2.0});
  REQUIRE_THROWS_AS(
      pfc::core::transform(out, pfc::test::Reaction{1.0, 1.0, 1.0}, x, short_in, x),
      std::invalid_argument);
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
