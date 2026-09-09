// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_order_parameter.cpp
 * @brief Catch2 tests for the real-space bond-orientational order metric
 *        and the free-energy/structure-factor sampler added for `#118`.
 *
 * @details
 * The key test the issue asks for: `bond_orientational_order` on
 * *analytically constructed* ideal square and triangular lattices (not on
 * anything peak-detected from a simulated field) returns the exact values
 * derived in `order_parameter.hpp`'s file comment,
 * \f$(\psi_4,\psi_6)=(1,0)\f$ for square and \f$(0,1)\f$ for triangular. That
 * is what makes \f$(\psi_4,\psi_6)\f$ usable as a real structural classifier.
 * A looser end-to-end test then checks the same thing through the actual
 * peak detector on a `lattice_seed`-initialised field, and
 * `FreeEnergySampler` is checked against a constant field, where the free
 * energy density is elementary algebra.
 *
 * This file has no `main()`; it links into `test_higher_order_pfc`, which
 * owns the Catch2 session and `MPI_Init`/`MPI_Finalize`.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <numbers>

#include <nlohmann/json.hpp>

#include <higher_order_pfc/free_energy.hpp>
#include <higher_order_pfc/higher_order_pfc_physics.hpp>
#include <higher_order_pfc/lattice_seed.hpp>
#include <higher_order_pfc/order_parameter.hpp>
#include <higher_order_pfc/seeded_noise.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

using Catch::Matchers::WithinAbs;
using nlohmann::json;

namespace hop = higher_order_pfc;

namespace {
int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
} // namespace

// ---------------------------------------------------------------------------
// The key test: analytically constructed lattices (acceptance criterion 6)
// ---------------------------------------------------------------------------

TEST_CASE("Bond-orientational order is exactly (1,0) for an ideal square lattice",
          "[higher_order_pfc][order][analytical]") {
  constexpr double a = 1.3; // lattice constant is arbitrary; a ratio is measured
  constexpr int n_cells = 12;
  const auto pts = hop::ideal_square_lattice(a, n_cells);
  REQUIRE(pts.size() == std::size_t(n_cells) * std::size_t(n_cells));

  const auto bo = hop::bond_orientational_order(pts, a * n_cells, a * n_cells, 1.3);
  REQUIRE(bo.n_points == pts.size());
  REQUIRE_THAT(bo.mean_neighbours, WithinAbs(4.0, 1e-9)); // 4 nearest neighbours
  REQUIRE_THAT(bo.psi4.global, WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(bo.psi4.local, WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(bo.psi6.global, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(bo.psi6.local, WithinAbs(0.0, 1e-9));
}

TEST_CASE("Bond-orientational order is exactly (0,1) for an ideal triangular lattice",
          "[higher_order_pfc][order][analytical]") {
  constexpr double a = 0.9;
  constexpr int n_x = 12, n_y = 12;
  const auto pts = hop::ideal_triangular_lattice(a, n_x, n_y);
  REQUIRE(pts.size() == std::size_t(n_x) * std::size_t(2 * n_y));

  const double Lx = a * n_x;
  const double Ly = a * std::numbers::sqrt3 * n_y;
  const auto bo = hop::bond_orientational_order(pts, Lx, Ly, 1.3);
  REQUIRE(bo.n_points == pts.size());
  REQUIRE_THAT(bo.mean_neighbours, WithinAbs(6.0, 1e-9)); // 6 nearest neighbours
  REQUIRE_THAT(bo.psi6.global, WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(bo.psi6.local, WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(bo.psi4.global, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(bo.psi4.local, WithinAbs(0.0, 1e-9));
}

TEST_CASE("A rotated square lattice still gives psi4 = 1: the metric is "
          "orientation-blind, only the box need not be",
          "[higher_order_pfc][order][analytical]") {
  // Rotate every point by a fixed angle about the origin; bond angles shift by
  // the same amount, which leaves |psi_4| invariant. Skip periodicity (huge
  // box) so the rotation cannot wrap oddly at the edges.
  constexpr double a = 1.0;
  constexpr int n_cells = 10;
  constexpr double theta = 0.37;
  auto pts = hop::ideal_square_lattice(a, n_cells);
  const double c = std::cos(theta), s = std::sin(theta);
  for (auto &p : pts) {
    const double x = p.x, y = p.y;
    p.x = c * x - s * y;
    p.y = s * x + c * y;
  }
  const auto bo = hop::bond_orientational_order(pts, 1.0e6, 1.0e6, 1.3);
  REQUIRE_THAT(bo.psi4.global, WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(bo.psi6.global, WithinAbs(0.0, 1e-9));
}

TEST_CASE("A polycrystalline mix of two square grains suppresses the global "
          "order parameter but not the local one",
          "[higher_order_pfc][order][analytical]") {
  // Two square grains at very different orientations, side by side (each
  // large enough that its own periodic image dominates its neighbour search).
  // Global psi4 averages the two grains' *complex* order parameters, which
  // point in different directions and partially cancel; local psi4 does not
  // care about relative orientation and stays near 1 in both grains.
  constexpr double a = 1.0;
  constexpr int n_cells = 10;
  auto grain_a = hop::ideal_square_lattice(a, n_cells);
  auto grain_b = hop::ideal_square_lattice(a, n_cells);
  constexpr double theta = std::numbers::pi / 4.0; // 45 degrees: maximally different
  const double c = std::cos(theta), s = std::sin(theta);
  const double shift = a * n_cells * 3.0; // grains far enough apart, own images
  for (auto &p : grain_b) {
    const double x = p.x, y = p.y;
    p.x = c * x - s * y + shift;
    p.y = s * x + c * y;
  }
  std::vector<hop::Peak> both = grain_a;
  both.insert(both.end(), grain_b.begin(), grain_b.end());
  const double L = 1.0e7; // effectively non-periodic across the two grains
  const auto bo = hop::bond_orientational_order(both, L, L, 1.3);
  REQUIRE(bo.psi4.local > 0.9);   // each grain is still locally square
  REQUIRE(bo.psi4.global < 0.5);  // but the two grains disagree on orientation
}

// ---------------------------------------------------------------------------
// End-to-end: peak detection on a lattice_seed field (looser tolerance)
// ---------------------------------------------------------------------------

TEST_CASE("Peak detection + bond order recover psi4 ~ 1 on a lattice_seed square field",
          "[higher_order_pfc][order][pipeline]") {
  if (world_size() != 1) {
    SKIP("single-rank whole-grid peak detection");
  }
  constexpr int N = 64;
  constexpr double dx = 2.0 * std::numbers::pi / 8.0; // |k|=1 on mode N/8
  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({dx, dx, dx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &psi = stack.u();

  hop::LatticeSeed seed;
  seed.set_psi0(0.0);
  seed.set_amplitude(0.6);
  seed.set_modes({{N / 8, 0, 0}, {0, N / 8, 0}}); // |k|=1 square: (10) family
  pfc::apply_field_modifier(seed, psi, 0.0);

  const auto peaks = hop::detect_peaks(psi, N, N, dx, dx);
  REQUIRE(peaks.size() > 4); // more than a handful of density maxima detected
  const auto bo = hop::bond_orientational_order(peaks, dx * N, dx * N, 1.3);
  REQUIRE(bo.psi4.global > 0.8);
  REQUIRE(bo.psi6.global < 0.3);
}

// ---------------------------------------------------------------------------
// power_near tie-breaking (regression: an exact-boundary target must not
// silently read the empty neighbouring shell)
// ---------------------------------------------------------------------------

TEST_CASE("power_near breaks an exact-distance tie towards the shell that "
          "carries power",
          "[higher_order_pfc][energy][analytical]") {
  pfc::apps::StructureFactor sf;
  // Two shells equidistant from k=1.0 (0.5 and 1.5), the lower-index one
  // empty: exactly the situation a lattice_seed run produces on the shipped
  // grid, where k=1 sits precisely on a shell boundary.
  sf.k = {0.5, 1.5};
  sf.S = {0.0, 42.0};
  REQUIRE_THAT(hop::power_near(sf, 1.0), WithinAbs(42.0, 1e-15));

  // And the reverse: empty neighbour on the high side.
  sf.S = {42.0, 0.0};
  REQUIRE_THAT(hop::power_near(sf, 1.0), WithinAbs(42.0, 1e-15));
}

// ---------------------------------------------------------------------------
// FreeEnergySampler against an elementary case
// ---------------------------------------------------------------------------

TEST_CASE("Free-energy density on a constant field matches the elementary "
          "k=0 formula",
          "[higher_order_pfc][energy][analytical]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral sampler");
  }
  constexpr int N = 16;
  constexpr double dx = 2.0 * std::numbers::pi / 8.0;
  constexpr double psi0 = -0.12;
  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({dx, dx, dx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &psi = stack.u();
  psi.apply([&](double, double, double) { return psi0; });

  const auto p = hop::HigherOrderPFCParams{}; // defaults: eps=0.25, two-mode, g=0
  hop::FreeEnergySampler<pfc::HostSpace> sampler(domain, stack.fft(), MPI_COMM_WORLD);
  const auto sample = sampler.sample(psi, p, 32);

  REQUIRE_THAT(sample.mean_psi, WithinAbs(psi0, 1e-12));
  // Constant field: all spectral weight at k=0, so the quadratic term is
  // Lambda(0) * psi0^2 / 2 exactly.
  const double lambda0 = p.kernel(0.0);
  const double expected =
      0.5 * lambda0 * psi0 * psi0 - (p.g / 3.0) * psi0 * psi0 * psi0 +
      0.25 * psi0 * psi0 * psi0 * psi0;
  REQUIRE_THAT(sample.free_energy_density, WithinAbs(expected, 1e-9));
}

TEST_CASE("Diagnostics mean_psi tracks exact mass conservation through ETD steps",
          "[higher_order_pfc][energy][mass]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral sampler");
  }
  constexpr int N = 32;
  constexpr double dx = 2.0 * std::numbers::pi / 8.0;
  constexpr double dt = 0.05;
  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({dx, dx, dx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = hop::HigherOrderPFCPhysics<>::from_json(json{{"g", 0.5}}, domain,
                                                       stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &psi = state.get_field<double>("psi");
  hop::SeededNoise noise; // reuse the app's own decomposition-independent seed
  noise.psi0 = -0.05;
  noise.amplitude = 1.0e-2;
  noise.seed = 5;
  pfc::apply_field_modifier(noise, psi, 0.0);

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<hop::HigherOrderPFCPhysics<>> sys(phys, stack.fft(), state,
                                                                dt, opt);
  hop::FreeEnergySampler<pfc::HostSpace> sampler(domain, stack.fft(), MPI_COMM_WORLD);
  const double mean0 = sampler.sample(psi, phys.params, 16).mean_psi;
  REQUIRE_THAT(mean0, WithinAbs(-0.05, 1e-12));

  double t = 0.0;
  for (int step = 0; step < 20; ++step) t = sys.step(t);

  const double mean1 = sampler.sample(psi, phys.params, 16).mean_psi;
  REQUIRE_THAT(mean1, WithinAbs(mean0, 1e-12));
}
