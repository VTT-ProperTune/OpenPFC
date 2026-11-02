// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <vector>

#include <openpfc/kernel/fft/bounded_axis_cost.hpp>
#include <openpfc/kernel/fft/chebyshev.hpp>
#include <openpfc/kernel/fft/complex_line_fft.hpp>
#include <openpfc/kernel/fft/distributed_complex_line_fft.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Distributed complex line matches the one-rank DFT",
          "[fft][line_dft][distributed_line]") {
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  constexpr int n = 17;
  std::vector<std::complex<double>> global(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    global[static_cast<std::size_t>(i)] = {std::sin(0.3 * i), std::cos(0.7 * i)};
  }
  pfc::fft::ComplexLineFFT serial(n);
  std::vector<std::complex<double>> expected(global.size());
  serial.forward(global, expected);

  const auto block = pfc::fft::line_block(n, rank, ranks);
  pfc::fft::DistributedComplexLineFFT distributed(n, MPI_COMM_WORLD);
  REQUIRE(distributed.global_length() == n);
  REQUIRE(distributed.local_length() == block.count);
  REQUIRE(distributed.local_offset() == block.offset);
  std::vector<std::complex<double>> local(
      global.begin() + block.offset, global.begin() + block.offset + block.count);
  const auto saved = local;
  std::vector<std::complex<double>> spectrum(local.size());
  std::vector<std::complex<double>> back(local.size());
  distributed.forward(local, spectrum);
  distributed.backward(spectrum, back);
  REQUIRE(local == saved);

  double peak = 0;
  double roundtrip = 0;
  for (int i = 0; i < block.count; ++i) {
    const auto got = spectrum[static_cast<std::size_t>(i)];
    const auto want = expected[static_cast<std::size_t>(block.offset + i)];
    peak = std::max(peak, std::abs(got - want));
    roundtrip = std::max(roundtrip, std::abs(back[static_cast<std::size_t>(i)] /
                                                 static_cast<double>(n) -
                                             local[static_cast<std::size_t>(i)]));
  }
  double reduced_peak = 0;
  double reduced_round = 0;
  MPI_Allreduce(&peak, &reduced_peak, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&roundtrip, &reduced_round, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(reduced_peak < 1e-10);
  REQUIRE(reduced_round < 1e-10);
  if (ranks > 1) {
    REQUIRE(distributed.comm_bytes() > 0);
  }
}

TEST_CASE("Distributed Dirichlet Poisson returns the owned Chebyshev slice",
          "[fft][chebyshev][distributed_line]") {
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  constexpr int degree = 16;
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> forcing(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) forcing[i] = std::exp(nodes[i]);
  const double at_plus = std::exp(1.0);
  const double at_minus = std::exp(-1.0);
  const auto full =
      pfc::fft::chebyshev_dirichlet_poisson(forcing, at_plus, at_minus);
  const auto block = pfc::fft::line_block(degree + 1, rank, ranks);
  const std::span<const double> local(forcing.data() + block.offset, block.count);
  const auto owned = pfc::fft::distributed_chebyshev_dirichlet_poisson(
      local, degree, at_plus, at_minus, MPI_COMM_WORLD);
  REQUIRE(owned.size() == static_cast<std::size_t>(block.count));
  double peak = 0;
  for (int i = 0; i < block.count; ++i) {
    peak =
        std::max(peak, std::abs(owned[static_cast<std::size_t>(i)] -
                                full[static_cast<std::size_t>(block.offset + i)]));
  }
  double reduced = 0;
  MPI_Allreduce(&peak, &reduced, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(reduced < 1e-12);
}

TEST_CASE("Distributed Neumann Poisson returns the owned Chebyshev slice",
          "[fft][chebyshev][distributed_line]") {
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  constexpr int degree = 16;
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto full = pfc::fft::chebyshev_neumann_poisson(forcing, -2.0, 2.0);
  const auto block = pfc::fft::line_block(degree + 1, rank, ranks);
  const std::span<const double> local(forcing.data() + block.offset, block.count);
  const auto owned = pfc::fft::distributed_chebyshev_neumann_poisson(
      local, degree, -2.0, 2.0, MPI_COMM_WORLD);
  REQUIRE(owned.size() == static_cast<std::size_t>(block.count));
  double peak = 0;
  for (int i = 0; i < block.count; ++i) {
    peak =
        std::max(peak, std::abs(owned[static_cast<std::size_t>(i)] -
                                full[static_cast<std::size_t>(block.offset + i)]));
  }
  double reduced = 0;
  MPI_Allreduce(&peak, &reduced, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(reduced < 1e-12);
}

TEST_CASE("Distributed Robin Poisson returns the owned Chebyshev slice",
          "[fft][chebyshev][distributed_line]") {
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  constexpr int degree = 16;
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto full =
      pfc::fft::chebyshev_robin_poisson(forcing, 1.0, 1.0, -2.0, 1.0, 1.0, 2.0);
  const auto block = pfc::fft::line_block(degree + 1, rank, ranks);
  const std::span<const double> local(forcing.data() + block.offset, block.count);
  const auto owned = pfc::fft::distributed_chebyshev_robin_poisson(
      local, degree, 1.0, 1.0, -2.0, 1.0, 1.0, 2.0, MPI_COMM_WORLD);
  REQUIRE(owned.size() == static_cast<std::size_t>(block.count));
  double peak = 0;
  for (int i = 0; i < block.count; ++i) {
    peak =
        std::max(peak, std::abs(owned[static_cast<std::size_t>(i)] -
                                full[static_cast<std::size_t>(block.offset + i)]));
  }
  double reduced = 0;
  MPI_Allreduce(&peak, &reduced, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(reduced < 1e-12);
}

TEST_CASE("Bounded-axis cost prefers Chebyshev and records the line",
          "[fft][chebyshev][distributed_line]") {
  const auto cost = pfc::fft::compare_bounded_axis(32, MPI_COMM_WORLD);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  REQUIRE(cost.ranks == ranks);
  REQUIRE(cost.points == 33);
  REQUIRE(cost.chebyshev_error < 1e-8);
  REQUIRE(cost.fd4_error < 1e-2);
  REQUIRE(cost.fd8_error < cost.fd4_error);
  REQUIRE(cost.prefer_chebyshev);
  REQUIRE(cost.distributed_roundtrip_peak < 1e-10);
  REQUIRE(cost.chebyshev_seconds >= 0);
  REQUIRE(cost.distributed_poisson_seconds >= 0);
  REQUIRE(cost.local_line_seconds >= 0);
  REQUIRE(cost.distributed_line_seconds >= 0);
  REQUIRE(cost.fd4_seconds >= 0);
  REQUIRE(cost.fd8_seconds >= 0);
  REQUIRE(cost.fd_matrix_bytes ==
          static_cast<std::size_t>(31 * 31) * sizeof(double));
}

TEST_CASE("Distributed line rejects a length shorter than the communicator",
          "[fft][line_dft][distributed_line]") {
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks == 1) {
    REQUIRE_THROWS_AS(pfc::fft::DistributedComplexLineFFT(1, MPI_COMM_WORLD),
                      std::invalid_argument);
  } else {
    REQUIRE_THROWS_AS(pfc::fft::DistributedComplexLineFFT(ranks - 1, MPI_COMM_WORLD),
                      std::invalid_argument);
  }
  REQUIRE_THROWS_AS(pfc::fft::compare_bounded_axis(8, MPI_COMM_WORLD),
                    std::invalid_argument);
}
