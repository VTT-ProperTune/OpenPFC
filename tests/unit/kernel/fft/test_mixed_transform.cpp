// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <numbers>
#include <openpfc/kernel/fft/mixed_transform.hpp>
#include <vector>

using namespace pfc::fft;
using Catch::Matchers::WithinAbs;

TEST_CASE("mixed periodic periodic bounded solves on MPI slabs",
          "[mixed_transform][MPI]") {
  constexpr double pi = std::numbers::pi;
  int ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  for (auto t : {RealTransformType::DCT1, RealTransformType::DCT2,
                 RealTransformType::DST1, RealTransformType::DST2}) {
    for (int n : {5, 6}) {
      BoundedAxis z{t,
                    t == RealTransformType::DCT1   ? UniformGrid::Endpoints
                    : t == RealTransformType::DST1 ? UniformGrid::Interior
                                                   : UniformGrid::CellCentered,
                    n, 2.3, 0.2};
      int nx = 7, ny = std::max(6, ranks);
      MixedTransform plan(nx, ny, 3.1, 4.2, z);
      auto box = plan.inbox();
      std::vector<double> u(plan.size_inbox()), f(u.size()), back(u.size()),
          lap(u.size());
      std::vector<std::complex<double>> c(plan.size_outbox());
      double kz = wavenumber(z, 1), kx = 2 * pi / 3.1, ky = 2 * pi / 4.2;
      pfc::for_each_index(box, [&](auto p) {
        double phase = kz * (coordinate(z, p[2]) - z.origin);
        u[box.to_linear(p)] = std::cos(2 * pi * p[0] / nx) *
                              std::sin(2 * pi * p[1] / ny) *
                              (is_cosine(t) ? std::cos(phase) : std::sin(phase));
      });
      auto saved = u;
      plan.forward(u, c);
      auto saved_c = c;
      plan.backward(c, back);
      REQUIRE(u == saved);
      REQUIRE(c == saved_c);
      for (std::size_t i = 0; i < u.size(); ++i)
        REQUIRE_THAT(back[i], WithinAbs(u[i], 2e-12));
      for (auto s :
           {LaplacianSymbol::Spectral, LaplacianSymbol::CenteredDifference}) {
        double lambda = plan.eigenvalue({1, 1, 1}, s);
        if (s == LaplacianSymbol::Spectral)
          REQUIRE_THAT(lambda, WithinAbs(kx * kx + ky * ky + kz * kz, 1e-12));
        for (std::size_t i = 0; i < u.size(); ++i) f[i] = lambda * u[i];
        plan.poisson(f, back, s);
        plan.laplacian(u, lap, s);
        for (std::size_t i = 0; i < u.size(); ++i) {
          REQUIRE_THAT(back[i], WithinAbs(u[i], 2e-12));
          REQUIRE_THAT(lap[i], WithinAbs(-f[i], 2e-10));
        }
      }
      if (is_cosine(t)) {
        std::fill(f.begin(), f.end(), 1);
        REQUIRE_THROWS_AS(plan.poisson(f, back), std::invalid_argument);
        std::fill(f.begin(), f.end(), 0);
        plan.poisson(f, back);
        for (double v : back) REQUIRE(v == 0);
      }
    }
  }
}

TEST_CASE("mixed transforms validate geometry collectively",
          "[mixed_transform][MPI]") {
  int ranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  BoundedAxis z{RealTransformType::DST2, UniformGrid::CellCentered, 3, 1};
  REQUIRE_THROWS_AS(MixedTransform(3, ranks - 1, 1, 1, z), std::invalid_argument);
  if (ranks > 1)
    REQUIRE_THROWS_AS(MixedTransform(3, ranks, 1 + rank, 1, z),
                      std::invalid_argument);
  // The same implementation also covers a single bounded line.
  MixedTransform line(1, 1, 1, 1, z, MPI_COMM_SELF);
  std::vector<double> in = {0.2, 0.7, -0.1}, back(3);
  std::vector<std::complex<double>> c(line.size_outbox());
  line.forward(in, c);
  line.backward(c, back);
  for (int j = 0; j < 3; ++j) REQUIRE_THAT(back[j], WithinAbs(in[j], 1e-12));
}
