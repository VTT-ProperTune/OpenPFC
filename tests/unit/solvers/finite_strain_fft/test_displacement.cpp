// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/solvers/finite_strain_fft/displacement.hpp>

TEST_CASE("periodic displacement recovers mixed analytic modes and gauge",
          "[finite_strain][displacement]") {
  using namespace pfc;
  using namespace pfc::finite_strain;
  auto domain = domain::create(GridSize({7, 9, 11}), PhysicalOrigin({0, 0, 0}),
                               GridSpacing({2. / 7, 3. / 9, 4. / 11}));
  auto d = decomposition::create(domain, 1);
  auto fft = fft::create(d, 0, MPI_COMM_SELF);
  auto macro = identity2();
  macro(0, 0) = 1.1;
  macro(1, 2) = .07;
  std::vector<Tensor2> f(7 * 9 * 11);
  std::vector<std::array<double, 3>> exact(f.size());
  const double q[3] = {2 * M_PI / 2, -4 * M_PI / 3, 2 * M_PI / 4};
  for (int z = 0; z < 11; ++z)
    for (int y = 0; y < 9; ++y)
      for (int x = 0; x < 7; ++x) {
        int p = x + 7 * (y + 9 * z);
        double phase = q[0] * x * 2 / 7 + q[1] * y * 3 / 9 + q[2] * z * 4 / 11;
        f[p] = macro;
        for (int i = 0; i < 3; ++i) {
          double amplitude = .003 * (i + 1);
          exact[p][i] = amplitude * std::sin(phase) + .002 * std::cos(2 * phase);
          for (int j = 0; j < 3; ++j)
            f[p](i, j) += amplitude * q[j] * std::cos(phase) -
                          .004 * q[j] * std::sin(2 * phase);
        }
      }
  auto result = reconstruct_displacement(fft, domain, f, macro);
  REQUIRE(std::sqrt(result.gradient_error_squared / result.gradient_squared) <
          1e-13);
  std::array<double, 3> mean{};
  for (std::size_t p = 0; p < f.size(); ++p)
    for (int i = 0; i < 3; ++i) {
      REQUIRE(std::abs(result.fluctuation[p][i] - exact[p][i]) < 1e-14);
      mean[i] += result.fluctuation[p][i] / f.size();
    }
  for (double value : mean) REQUIRE(std::abs(value) < 1e-15);
  // A varying F00 depending only on y cannot be a compatible row gradient.
  for (int z = 0; z < 11; ++z)
    for (int y = 0; y < 9; ++y)
      for (int x = 0; x < 7; ++x)
        f[x + 7 * (y + 9 * z)](0, 0) += .02 * std::sin(2 * M_PI * y / 9);
  result = reconstruct_displacement(fft, domain, f, macro);
  REQUIRE(result.gradient_error_squared > 1e-3);
  f[0](0, 0) = NAN;
  REQUIRE_THROWS_AS(reconstruct_displacement(fft, domain, f, macro),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(reconstruct_displacement(fft, domain, {}, macro),
                    std::invalid_argument);
}

TEST_CASE("displacement reconstruction rejects an even grid",
          "[finite_strain][displacement]") {
  using namespace pfc;
  using namespace pfc::finite_strain;
  auto domain = domain::create(Int3{4, 4, 4});
  auto decomposition = decomposition::create(domain, 1);
  auto fft = fft::create(decomposition, 0, MPI_COMM_SELF);
  std::vector<Tensor2> field(fft.size_inbox(), identity2());
  REQUIRE_THROWS_AS(reconstruct_displacement(fft, domain, field, identity2()),
                    std::invalid_argument);
}
