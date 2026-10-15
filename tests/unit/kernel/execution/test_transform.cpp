// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/execution/transform.hpp>

using Catch::Approx;
using pfc::backend::CPUTag;
using pfc::core::DataBuffer;

namespace {

template <typename T> DataBuffer<CPUTag, T> buffer(const std::vector<T> &v) {
  DataBuffer<CPUTag, T> b(v.size());
  b.copy_from_host(v);
  return b;
}

struct Reaction {
  double l1;
  double l2;
  double c;
  OPENPFC_HD double operator()(double grad, double phi, double phi0) const {
    return grad + l1 * (phi - phi0) + l2 * phi * (phi * phi - 1.0) + c;
  }
};

struct Abs2 {
  OPENPFC_HD double operator()(const std::complex<double> &z) const {
    return std::norm(z);
  }
};

} // namespace

TEST_CASE("transform applies a functor to three inputs", "[core][transform][cpu]") {
  const auto grad = buffer<double>({1.0, 0.0, -1.0});
  const auto phi = buffer<double>({0.5, 1.0, 2.0});
  const auto phi0 = buffer<double>({0.0, 1.0, 1.0});
  DataBuffer<CPUTag, double> out(3);
  const Reaction r{0.1, 0.5, -0.2};
  pfc::core::transform(out, r, grad, phi, phi0);
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(out[i] == Approx(r(grad[i], phi[i], phi0[i])));
  }
  REQUIRE(out[1] == Approx(-0.2));
}

TEST_CASE("transform maps complex input to real output and runs in place",
          "[core][transform][cpu]") {
  using C = std::complex<double>;
  const auto z = buffer<C>({C{3.0, 4.0}, C{0.0, -1.0}});
  DataBuffer<CPUTag, double> out(2);
  pfc::core::transform(out, Abs2{}, z);
  REQUIRE(out[0] == 25.0);
  REQUIRE(out[1] == 1.0);

  auto x = buffer<double>({2.0, -3.0});
  pfc::core::transform(x, [](double v) { return v * v; }, x);
  REQUIRE(x[0] == 4.0);
  REQUIRE(x[1] == 9.0);
}

TEST_CASE("transform rejects inputs whose size differs from out",
          "[core][transform][cpu]") {
  DataBuffer<CPUTag, double> out(3);
  const auto a = buffer<double>({1.0, 2.0, 3.0});
  const auto b = buffer<double>({1.0, 2.0});
  REQUIRE_THROWS_AS(pfc::core::transform(
                        out, [](double x, double y) { return x + y; }, a, b),
                    std::invalid_argument);
  DataBuffer<CPUTag, double> empty(0);
  pfc::core::transform(empty, [](double x) { return x; }, empty);
  REQUIRE(empty.empty());
}
