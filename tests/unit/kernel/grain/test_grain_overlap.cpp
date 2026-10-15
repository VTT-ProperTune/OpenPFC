// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/grain/overlap.hpp>

namespace ov = pfc::grain::overlap;
using Catch::Matchers::WithinRel;
using pfc::grain::Id;

namespace {

using FFT = decltype(pfc::fft::create(
    pfc::decomposition::create(std::declval<const pfc::Domain &>(), 1)));

struct Setup {
  pfc::Domain domain;
  FFT fft;
  pfc::Box3i inbox;
  Setup(pfc::Int3 n, double h, int nproc)
      : domain(pfc::domain::create(pfc::GridSize(n),
                                   pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                   pfc::GridSpacing({h, h, h}))),
        fft(pfc::fft::create(pfc::decomposition::create(domain, nproc))),
        inbox(fft.get_inbox_bounds()) {}
};

} // namespace

TEST_CASE("overlap force is the derivative of the overlap energy",
          "[grain][overlap]") {
  int nproc = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Setup s({12, 10, 8}, 0.7, nproc);
  const auto n = s.fft.size_inbox();
  std::mt19937_64 rng(11 + rank);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> phi(n), v(n), force(n), scratch(n);
  std::vector<Id> labels(n);
  for (std::size_t i = 0; i < n; ++i) {
    phi[i] = u(rng);
    v[i] = u(rng) - 0.5;
    labels[i] = static_cast<Id>(u(rng) * 4.0); // 0 (background) .. 3
  }
  const std::vector<Id> ids{1, 2, 3};
  const auto kernel = ov::kernel(s.fft.get_outbox_bounds(), s.domain, 1.1);
  const double omega = ov::weight(0.3, 1.1);
  ov::evaluate(s.fft, kernel, omega, phi, labels, ids, force, MPI_COMM_WORLD);
  double slope = 0.0;
  for (std::size_t i = 0; i < n; ++i) slope += force[i] * v[i];
  MPI_Allreduce(MPI_IN_PLACE, &slope, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  slope /= 12.0 * 10.0 * 8.0; // F/V changes by (1/N) sum f v
  const double e = 1.0e-5;
  auto at = [&](double t) {
    std::vector<double> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = phi[i] + t * v[i];
    return ov::evaluate(s.fft, kernel, omega, p, labels, ids, scratch,
                        MPI_COMM_WORLD);
  };
  const double numeric = (at(e) - at(-e)) / (2.0 * e);
  REQUIRE_THAT(numeric, WithinRel(slope, 1.0e-7));
}

TEST_CASE("two labelled half-spaces carry the declared boundary energy",
          "[grain][overlap]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  // 128 cells of h = 0.5 along x, sigma = 2: interfaces 32 apart, kernel
  // tails negligible; the periodic box has two interfaces of area Ly Lz.
  // Sharp indicators on the grid leave a 0.3 % quadrature error.
  Setup s({128, 4, 4}, 0.5, nproc);
  const auto n = s.fft.size_inbox();
  std::vector<double> phi(n, 1.0), force(n);
  std::vector<Id> labels(n);
  std::size_t m = 0;
  for (int k = s.inbox.low[2]; k <= s.inbox.high[2]; ++k)
    for (int j = s.inbox.low[1]; j <= s.inbox.high[1]; ++j)
      for (int i = s.inbox.low[0]; i <= s.inbox.high[0]; ++i)
        labels[m++] = i < 64 ? 1 : 2;
  const double sigma = 2.0, gamma = 0.25;
  const auto kernel = ov::kernel(s.fft.get_outbox_bounds(), s.domain, sigma);
  const double fv =
      ov::evaluate(s.fft, kernel, ov::weight(gamma, sigma), phi, labels,
                   std::vector<Id>{1, 2}, force, MPI_COMM_WORLD);
  const double volume = 64.0 * 2.0 * 2.0, area = 2.0 * 2.0;
  REQUIRE_THAT(fv * volume / (2.0 * area), WithinRel(gamma, 5.0e-3));
}

TEST_CASE("one identity has no overlap energy", "[grain][overlap]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  Setup s({8, 8, 8}, 1.0, nproc);
  const auto n = s.fft.size_inbox();
  std::vector<double> phi(n, 0.7), force(n, 1.0);
  std::vector<Id> labels(n, 5);
  const auto kernel = ov::kernel(s.fft.get_outbox_bounds(), s.domain, 1.0);
  const double fv = ov::evaluate(s.fft, kernel, ov::weight(1.0, 1.0), phi, labels,
                                 std::vector<Id>{5}, force, MPI_COMM_WORLD);
  REQUIRE(std::abs(fv) < 1.0e-14);
  for (double f : force) REQUIRE(std::abs(f) < 1.0e-13);
}
