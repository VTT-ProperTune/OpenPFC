// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <heffte.h>
#include <mpi.h>

namespace {

using Point = std::array<int, 3>;

std::size_t offset(const Point &p, const heffte::box3d<> &box) {
  const auto &o = box.order;
  return (p[o[2]] * box.size[o[1]] + p[o[1]]) * box.size[o[0]] + p[o[0]];
}

template <typename F> void each_point(const Point &size, F &&f) {
  for (int z = 0; z < size[2]; ++z)
    for (int y = 0; y < size[1]; ++y)
      for (int x = 0; x < size[0]; ++x) f(Point{x, y, z});
}

template <typename T> void check_transform(const Point &size, const Point &order) {
  INFO("scalar bytes " << sizeof(T));
  const heffte::box3d<> inbox({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1},
                              order);
  const auto outbox = inbox.r2c(0);
  auto options = heffte::default_options<heffte::backend::fftw>();
  options.use_reorder = false; // Exercise strided, batched plans as well.
  heffte::fft3d_r2c<heffte::backend::fftw> fft(inbox, outbox, 0, MPI_COMM_SELF,
                                               options);
  std::vector<T> input(inbox.count()), restored(inbox.count());
  std::vector<std::complex<T>> spectrum(outbox.count());
  each_point(size, [&](const Point &p) {
    const double t = p[0] + 13.0 * p[1] + 29.0 * p[2];
    input[offset(p, inbox)] =
        T(0.5 + 0.2 * std::cos(0.1 * t) + 0.05 * std::sin(0.37 * t));
  });
  const auto original = input;
  fft.forward(input.data(), spectrum.data(), heffte::scale::none);

  // A round trip alone can hide matching errors in the two transforms. Sum
  // the DFT independently from the original, precision-rounded real samples.
  const double tolerance = sizeof(T) == sizeof(float) ? 2.0e-5 : 1.0e-12;
  each_point(outbox.size, [&](const Point &k) {
    std::complex<double> expected{};
    each_point(size, [&](const Point &p) {
      double phase = 0.0;
      for (int axis = 0; axis < 3; ++axis)
        phase += double(k[axis]) * p[axis] / size[axis];
      expected += double(original[offset(p, inbox)]) *
                  std::polar(1.0, -2.0 * std::numbers::pi * phase);
    });
    const std::complex<double> actual = spectrum[offset(k, outbox)];
    REQUIRE(std::isfinite(actual.real()));
    REQUIRE(std::isfinite(actual.imag()));
    REQUIRE(std::abs(actual - expected) < tolerance * inbox.count());
  });

  fft.backward(spectrum.data(), restored.data(), heffte::scale::full);
  for (std::size_t i = 0; i < original.size(); ++i) {
    REQUIRE(std::isfinite(restored[i]));
    REQUIRE(std::abs(double(restored[i]) - original[i]) < tolerance);
  }
}

} // namespace

TEST_CASE("real transform plans agree with the DFT", "[fft][unit][planner]") {
  // MPI_COMM_SELF keeps these bounded local-plan checks valid in any runner.
  // Include the failing 512-point line, a reduced singleton axis, odd lengths,
  // and non-unit strides in both real-transform precision specializations.
  for (const Point size : {Point{256, 1, 1}, Point{512, 1, 1}, Point{1, 24, 20},
                           Point{8, 1, 6}, Point{6, 4, 1}, Point{7, 3, 5}}) {
    for (const Point order : {Point{0, 1, 2}, Point{1, 2, 0}}) {
      INFO("shape " << size[0] << 'x' << size[1] << 'x' << size[2]);
      INFO("order " << order[0] << order[1] << order[2]);
      check_transform<float>(size, order);
      check_transform<double>(size, order);
    }
  }
}
