// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_higher_order_pfc.cpp
 * @brief Catch2 tests for the higher-order (two-mode) PFC spectral app.
 *
 * @details
 * The interesting claims are all about the reciprocal-space kernel, so most of
 * these tests never touch a grid: they check that the implemented polynomial
 * really is the analytical one through \f$k^8\f$ (kernel) and \f$k^{10}\f$
 * (conserved evolution), that the band structure is what the two-mode
 * construction promises, and that mass conservation is exact rather than
 * approximate. The grid tests then confirm an ETD run reproduces the predicted
 * growth rate at a timestep far beyond any explicit stability limit.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <numbers>
#include <vector>

#include <nlohmann/json.hpp>

#include <higher_order_pfc/correlation_kernel.hpp>
#include <higher_order_pfc/higher_order_pfc_physics.hpp>
#include <higher_order_pfc/higher_order_pfc_session.hpp>
#include <higher_order_pfc/seeded_noise.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using nlohmann::json;

namespace hop = higher_order_pfc;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

/// Eight cells per \f$2\pi\f$ puts \f$|k|=1\f$ on mode 4 of a 32-cell box and
/// \f$|k|=\sqrt2\f$ on the (4,4) diagonal, so both correlation peaks are
/// exactly representable.
constexpr double kDx = 2.0 * std::numbers::pi / 8.0;

hop::HigherOrderPFCParams params_from(const json &j) {
  hop::HigherOrderPFCParams p;
  hop::apply_higher_order_pfc_json(j, p);
  return p;
}

/// The kernel written out in factored form, independently of the implementation.
double reference_two_mode(double u, double eps, double q1, double r1) {
  const double a = q1 * q1;
  const double one_plus_u = 1.0 + u;
  const double a_plus_u = a + u;
  return -eps + one_plus_u * one_plus_u * (r1 + a_plus_u * a_plus_u);
}

double reference_single_mode(double u, double eps) {
  const double one_plus_u = 1.0 + u;
  return -eps + one_plus_u * one_plus_u;
}

double mean_psi(const pfc::data::Field<double> &psi) {
  const auto n = psi.local_size();
  double sum = 0.0;
  std::size_t count = 0;
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        sum += psi(i, j, k);
        ++count;
      }
  return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

double variance_psi(const pfc::data::Field<double> &psi) {
  const double mu = mean_psi(psi);
  const auto n = psi.local_size();
  double acc = 0.0;
  std::size_t count = 0;
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const double d = psi(i, j, k) - mu;
        acc += d * d;
        ++count;
      }
  return (count > 0) ? acc / static_cast<double>(count) : 0.0;
}

/// Projection of `psi` onto cos(2 pi (nx x / Lx + ny y / Ly)).
double cosine_amplitude(const pfc::data::Field<double> &psi, double psi0, int nx,
                        int ny) {
  const auto n = psi.local_size();
  const auto sp = psi.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double Ly = static_cast<double>(n[1]) * sp[1];
  const double twopi = 2.0 * std::numbers::pi;
  double num = 0.0, den = 0.0;
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto x = psi.coords(i, j, k);
        const double w = std::cos(twopi * (static_cast<double>(nx) * x[0] / Lx +
                                           static_cast<double>(ny) * x[1] / Ly));
        num += (psi(i, j, k) - psi0) * w;
        den += w * w;
      }
  return (den > 0.0) ? num / den : 0.0;
}

json mini_session_json() {
  return {{"model", {{"name", "higher_order_pfc"}, {"params", {{"eps", 0.25}}}}},
          {"domain",
           {{"Lx", 16},
            {"Ly", 16},
            {"Lz", 1},
            {"dx", kDx},
            {"dy", kDx},
            {"dz", kDx},
            {"origin", "corner"}}},
          {"timestepping", {{"t0", 0.0}, {"t1", 0.1}, {"dt", 0.05}, {"saveat", -1.0}}},
          {"initial_conditions",
           {{{"target", "psi"},
             {"type", "cosine_mode"},
             {"psi0", 0.0},
             {"amplitude", 0.01},
             {"nx", 2},
             {"ny", 0},
             {"nz", 0}}}}};
}

} // namespace

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

TEST_CASE("HigherOrderPFC schema defaults describe a 2D square two-mode kernel",
          "[higher_order_pfc][schema]") {
  const hop::HigherOrderPFCParams p;
  REQUIRE(p.n_modes == 2);
  REQUIRE_THAT(p.eps, WithinAbs(0.25, 1e-15));
  REQUIRE_THAT(p.q1, WithinRel(std::sqrt(2.0), 1e-15));
  REQUIRE_THAT(p.r1, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(p.M, WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(p.g, WithinAbs(0.0, 1e-15));
}

TEST_CASE("HigherOrderPFC schema round-trips JSON and rejects a bad mode count",
          "[higher_order_pfc][schema]") {
  const auto p = params_from(json{{"eps", 0.4}, {"q1", 1.25}, {"r1", 0.1}, {"M", 2.0},
                                  {"g", 0.5}, {"n_modes", 1}});
  REQUIRE_THAT(p.eps, WithinAbs(0.4, 1e-15));
  REQUIRE_THAT(p.q1, WithinAbs(1.25, 1e-15));
  REQUIRE_THAT(p.r1, WithinAbs(0.1, 1e-15));
  REQUIRE_THAT(p.M, WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(p.g, WithinAbs(0.5, 1e-15));
  REQUIRE(p.n_modes == 1);

  REQUIRE_THROWS(params_from(json{{"n_modes", 3}}));
  REQUIRE_THROWS(params_from(json{{"n_modes", 0}}));
}

// ---------------------------------------------------------------------------
// The kernel really is the analytical polynomial (acceptance criterion 2)
// ---------------------------------------------------------------------------

TEST_CASE("Two-mode kernel matches the factored analytical form through k^8",
          "[higher_order_pfc][kernel][analytical]") {
  constexpr double eps = 0.3, q1 = 1.35, r1 = 0.07;
  const auto p = params_from(json{{"eps", eps}, {"q1", q1}, {"r1", r1}, {"n_modes", 2}});

  // |k| from 0 to well past both peaks, i.e. u = -k^2 down to -16.
  for (int i = 0; i <= 400; ++i) {
    const double k = 4.0 * static_cast<double>(i) / 400.0;
    const double u = -(k * k);
    REQUIRE_THAT(p.kernel(u), WithinRel(reference_two_mode(u, eps, q1, r1), 1e-12));
  }
}

TEST_CASE("Two-mode kernel coefficients are the derived expansion",
          "[higher_order_pfc][kernel][analytical]") {
  constexpr double eps = 0.3, q1 = 1.35, r1 = 0.07;
  const auto p = params_from(json{{"eps", eps}, {"q1", q1}, {"r1", r1}, {"n_modes", 2}});

  // Lambda(u) = -eps + (1+u)^2 [r1 + (a+u)^2], a = q1^2.
  // With A = r1 + a^2, B = 2a, C = 1 this expands to
  //   (A-eps) + (B+2A) u + (C+2B+A) u^2 + (2C+B) u^3 + C u^4.
  const double a = q1 * q1;
  const double A = r1 + a * a;
  const double B = 2.0 * a;
  const double C = 1.0;
  REQUIRE_THAT(p.kernel.coeff[0], WithinRel(A - eps, 1e-14));
  REQUIRE_THAT(p.kernel.coeff[1], WithinRel(B + 2.0 * A, 1e-14));
  REQUIRE_THAT(p.kernel.coeff[2], WithinRel(C + 2.0 * B + A, 1e-14));
  REQUIRE_THAT(p.kernel.coeff[3], WithinRel(2.0 * C + B, 1e-14));
  REQUIRE_THAT(p.kernel.coeff[4], WithinRel(C, 1e-14));

  // Quartic in u is eighth order in k, and the top coefficient is not zero.
  REQUIRE(hop::QuadraticKernel::degree == 4);
  REQUIRE(p.kernel.coeff[4] != 0.0);
}

TEST_CASE("Single-mode kernel is the classical fourth-order operator",
          "[higher_order_pfc][kernel][analytical]") {
  constexpr double eps = 0.3;
  const auto p = params_from(json{{"eps", eps}, {"n_modes", 1}});
  REQUIRE_THAT(p.kernel.coeff[0], WithinRel(1.0 - eps, 1e-14));
  REQUIRE_THAT(p.kernel.coeff[1], WithinAbs(2.0, 1e-14));
  REQUIRE_THAT(p.kernel.coeff[2], WithinAbs(1.0, 1e-14));
  // No k^6 or k^8 content at all.
  REQUIRE(p.kernel.coeff[3] == 0.0);
  REQUIRE(p.kernel.coeff[4] == 0.0);

  for (int i = 0; i <= 200; ++i) {
    const double k = 3.0 * static_cast<double>(i) / 200.0;
    const double u = -(k * k);
    REQUIRE_THAT(p.kernel(u), WithinRel(reference_single_mode(u, eps), 1e-12));
  }
}

TEST_CASE("Conserved evolution symbol is tenth order in k and matches M u Lambda",
          "[higher_order_pfc][kernel][analytical]") {
  constexpr double eps = 0.3, q1 = 1.35, r1 = 0.07, M = 1.7;
  const auto p = params_from(
      json{{"eps", eps}, {"q1", q1}, {"r1", r1}, {"M", M}, {"n_modes", 2}});

  REQUIRE(hop::EvolutionSymbol::degree == 5);
  REQUIRE(p.symbol.coeff[0] == 0.0);
  for (std::size_t i = 0; i <= hop::QuadraticKernel::degree; ++i)
    REQUIRE_THAT(p.symbol.coeff[i + 1], WithinRel(M * p.kernel.coeff[i], 1e-14));
  // Quintic in u is tenth order in k, with a nonzero leading term.
  REQUIRE(p.symbol.coeff[5] != 0.0);

  for (int i = 0; i <= 400; ++i) {
    const double k = 4.0 * static_cast<double>(i) / 400.0;
    const double u = -(k * k);
    const double expected = M * u * reference_two_mode(u, eps, q1, r1);
    REQUIRE_THAT(p.symbol(u), WithinAbs(expected, 1e-9 * (1.0 + std::abs(expected))));
  }
}

// ---------------------------------------------------------------------------
// Mass conservation (acceptance criterion 6)
// ---------------------------------------------------------------------------

TEST_CASE("Conserved dynamics kills the k=0 mode exactly",
          "[higher_order_pfc][mass]") {
  for (const int n_modes : {1, 2}) {
    const auto p = params_from(json{{"eps", 0.3}, {"M", 2.5}, {"n_modes", n_modes}});
    // Not "small": bit-exactly zero, because coeff[0] is a literal 0.0.
    REQUIRE(p.symbol.coeff[0] == 0.0);
    REQUIRE(p.symbol(0.0) == 0.0);
  }
}

// ---------------------------------------------------------------------------
// Band structure (acceptance criteria 3 and 5)
// ---------------------------------------------------------------------------

TEST_CASE("With r1 = 0 both correlation peaks are degenerate at -eps",
          "[higher_order_pfc][kernel][band]") {
  constexpr double eps = 0.25, q1 = 1.4142135623730951;
  const auto p = params_from(json{{"eps", eps}, {"q1", q1}, {"r1", 0.0}, {"n_modes", 2}});

  REQUIRE_THAT(p.kernel(-1.0), WithinAbs(-eps, 1e-14));          // |k| = 1
  REQUIRE_THAT(p.kernel(-(q1 * q1)), WithinAbs(-eps, 1e-14));    // |k| = q1

  // Both are genuine minima: the kernel rises on either side of each.
  for (const double kc : {1.0, q1}) {
    const double u = -(kc * kc);
    REQUIRE(p.kernel(u - 0.05) > p.kernel(u));
    REQUIRE(p.kernel(u + 0.05) > p.kernel(u));
  }
}

TEST_CASE("r1 > 0 lifts the second peak by exactly (1-q1^2)^2 r1",
          "[higher_order_pfc][kernel][band]") {
  constexpr double eps = 0.25, q1 = 1.4142135623730951, r1 = 0.05;
  const auto p = params_from(json{{"eps", eps}, {"q1", q1}, {"r1", r1}, {"n_modes", 2}});
  const double first = p.kernel(-1.0);
  const double second = p.kernel(-(q1 * q1));
  REQUIRE_THAT(second - first, WithinRel(p.second_mode_offset(), 1e-12));
  REQUIRE(second > first); // the k=1 peak stays the deeper one
}

TEST_CASE("The unstable band is exactly where the kernel is negative",
          "[higher_order_pfc][kernel][band]") {
  const auto p = params_from(json{{"eps", 0.25}, {"n_modes", 2}});
  for (int i = 1; i <= 400; ++i) {
    const double k = 4.0 * static_cast<double>(i) / 400.0;
    const double u = -(k * k);
    // L = M u Lambda with u < 0, so growth means a negative kernel.
    REQUIRE((p.symbol(u) > 0.0) == (p.kernel(u) < 0.0));
  }
}

TEST_CASE("Two modes destabilise |k| = sqrt(2) where one mode cannot",
          "[higher_order_pfc][kernel][comparison]") {
  constexpr double eps = 0.25;
  const double q1 = std::sqrt(2.0);
  const auto two = params_from(json{{"eps", eps}, {"q1", q1}, {"r1", 0.0}, {"n_modes", 2}});
  const auto one = params_from(json{{"eps", eps}, {"n_modes", 1}});

  const double u1 = -1.0;            // |k| = 1
  const double u2 = -(q1 * q1);      // |k| = sqrt(2)

  // Both kernels destabilise the first peak identically.
  REQUIRE(two.symbol(u1) > 0.0);
  REQUIRE(one.symbol(u1) > 0.0);
  REQUIRE_THAT(two.symbol(u1), WithinRel(one.symbol(u1), 1e-12));

  // Only the eighth-order kernel destabilises the second one. This is the
  // physical content of the extra terms: a second unstable band, which is what
  // selects square (2D) / FCC (3D) ordering over triangular / BCC.
  REQUIRE(two.symbol(u2) > 0.0);
  REQUIRE(one.symbol(u2) < 0.0);
}

// ---------------------------------------------------------------------------
// ETD on a grid (acceptance criteria 3, 6 and 7)
// ---------------------------------------------------------------------------

TEST_CASE("ETD reproduces L(k) growth and conserves the mean density",
          "[higher_order_pfc][spectral][mass]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 4; // |k| = 1 with eight cells per 2 pi
  constexpr double amp0 = 1.0e-6;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;

  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}),
                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({kDx, kDx, kDx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = hop::HigherOrderPFCPhysics<>::from_json(json::object(), domain,
                                                      stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &psi = state.get_field<double>("psi");

  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N) * kDx;
  psi.apply([&](double x, double, double) {
    return amp0 * std::cos(twopi * static_cast<double>(nx) * x / Lx);
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<hop::HigherOrderPFCPhysics<>> sys(phys, stack.fft(),
                                                                state, dt, opt);

  const double k = twopi * static_cast<double>(nx) / Lx;
  REQUIRE_THAT(k, WithinRel(1.0, 1e-12)); // the box really does resolve |k| = 1
  const double lambda = phys.linear_symbol(-(k * k));
  REQUIRE_THAT(lambda, WithinRel(0.25, 1e-12)); // M * 1 * eps
  REQUIRE(lambda > 0.0);

  const double mean0 = mean_psi(psi);
  REQUIRE_THAT(mean0, WithinAbs(0.0, 1e-14));

  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) t = sys.step(t);

  REQUIRE_THAT(mean_psi(psi), WithinAbs(mean0, 1e-13));
  const double amp = cosine_amplitude(psi, 0.0, nx, 0);
  REQUIRE_THAT(amp, WithinRel(amp0 * std::exp(lambda * t), 1e-6));
}

TEST_CASE("ETD is stable far beyond the explicit limit set by the k^10 term",
          "[higher_order_pfc][spectral][stiffness]") {
  if (world_size() != 1) {
    SKIP("single-rank stiffness check");
  }
  constexpr int N = 32;
  constexpr double dt = 0.05;
  constexpr int n_steps = 40;

  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}),
                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({kDx, kDx, kDx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = hop::HigherOrderPFCPhysics<>::from_json(json::object(), domain,
                                                      stack.fft().get_inbox_bounds());

  // Most negative eigenvalue on this grid sits at the Nyquist wave number.
  const double k_max = std::numbers::pi / kDx;
  const double lambda_max = std::abs(phys.linear_symbol(-(k_max * k_max)));
  const double dt_explicit = 2.0 / lambda_max;
  // The k^10 term makes this brutal: forward Euler would need a timestep
  // thousands of times smaller than the one ETD runs happily.
  REQUIRE(dt / dt_explicit > 1000.0);

  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &psi = state.get_field<double>("psi");
  hop::SeededNoise noise;
  noise.psi0 = 0.0;
  noise.amplitude = 1.0e-3;
  noise.seed = 7;
  pfc::apply_field_modifier(noise, psi, 0.0);

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<hop::HigherOrderPFCPhysics<>> sys(phys, stack.fft(),
                                                                state, dt, opt);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) t = sys.step(t);

  const auto n = psi.local_size();
  for (int j = 0; j < n[1]; ++j)
    for (int i = 0; i < n[0]; ++i) REQUIRE(std::isfinite(psi(i, j, 0)));
  REQUIRE(variance_psi(psi) < 1.0); // bounded, not blown up
}

TEST_CASE("A seeded perturbation orders while the mean density is held fixed",
          "[higher_order_pfc][spectral][crystal]") {
  if (world_size() != 1) {
    SKIP("single-rank ordering check");
  }
  constexpr int N = 32;
  constexpr double dt = 0.05;
  constexpr int n_steps = 400;
  // The local term shifts the effective rate about a nonzero mean to
  // sigma(k) = M u (Lambda(u) + n'(psibar)), n'(psi) = 3 psi^2 - 2 g psi.
  // At psi0 = -0.05, g = 0.5 that leaves sigma(|k|=1) = 0.1925 — decisive over
  // this run — while psi0 = -0.15 would leave only 0.0325 and the band would
  // barely move. The mean density is a control parameter, not a detail.
  constexpr double psi0 = -0.05;

  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}),
                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({kDx, kDx, kDx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = hop::HigherOrderPFCPhysics<>::from_json(
      json{{"eps", 0.25}, {"g", 0.5}}, domain, stack.fft().get_inbox_bounds());

  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &psi = state.get_field<double>("psi");
  hop::SeededNoise noise;
  noise.psi0 = psi0;
  noise.amplitude = 1.0e-2;
  noise.seed = 11;
  pfc::apply_field_modifier(noise, psi, 0.0);

  const double mean0 = mean_psi(psi);
  const double var0 = variance_psi(psi);
  // Mode 4 is |k| = 1, inside the unstable band. Mode 10 is |k| = 2.5, where
  // the k^10 term gives sigma ~ -3100 and nothing can survive.
  const double band0 = std::abs(cosine_amplitude(psi, psi0, 4, 0));
  REQUIRE_THAT(mean0, WithinAbs(psi0, 1e-12));

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<hop::HigherOrderPFCPhysics<>> sys(phys, stack.fft(),
                                                                state, dt, opt);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) t = sys.step(t);

  // Wavelength selection: the band the kernel prefers grows out of the noise,
  // everything well outside it is annihilated. That contrast, not the raw
  // variance, is what the correlation kernel is for.
  const double band = std::abs(cosine_amplitude(psi, psi0, 4, 0));
  const double out_of_band = std::abs(cosine_amplitude(psi, psi0, 10, 0));
  REQUIRE(band > 5.0 * band0);
  // Not identically zero: once the selected band saturates, the cubic term
  // feeds a little power back into every mode. It stays orders of magnitude
  // below the selected band, which is the claim that matters.
  REQUIRE(out_of_band < 1.0e-3 * band);
  // Structure grows out of the noise ...
  REQUIRE(variance_psi(psi) > 10.0 * var0);
  // ... and conserved dynamics never moves the mean, which is the
  // thermodynamic control parameter for which phase you land in.
  REQUIRE_THAT(mean_psi(psi), WithinAbs(mean0, 1e-12));
}

// ---------------------------------------------------------------------------
// Initial conditions and session
// ---------------------------------------------------------------------------

TEST_CASE("Seeded noise gives the same field on every decomposition",
          "[higher_order_pfc][ic]") {
  constexpr int N = 16;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto domain =
      pfc::domain::create(pfc::GridSize({N, N, 1}),
                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                          pfc::GridSpacing({kDx, kDx, kDx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, world_size(),
                                           MPI_COMM_WORLD);

  hop::SeededNoise noise;
  noise.psi0 = -0.15;
  noise.amplitude = 1.0e-2;
  noise.seed = 3;

  // Distributed apply on this rank's slab ...
  const pfc::SimulationContext context(MPI_COMM_WORLD);
  pfc::apply_field_modifier(noise, stack.u(), 0.0, &context);
  // ... versus a whole-grid apply every rank can do on its own.
  auto full = pfc::data::field_from_inbox<double>(
      domain, pfc::Box3i::from_bounds({0, 0, 0}, {N - 1, N - 1, 0}));
  pfc::apply_field_modifier(noise, full, 0.0);

  auto &psi = stack.u();
  psi.for_each_owned([&](int i, int j, int k) {
    REQUIRE(psi(i, j, k) == full(i + psi.box().low[0], j + psi.box().low[1],
                                 k + psi.box().low[2]));
  });
  REQUIRE_THAT(mean_psi(full), WithinAbs(-0.15, 1e-13));
  REQUIRE(variance_psi(full) > 0.0);

  REQUIRE_THROWS(hop::from_json(
      json{{"type", "seeded_noise"}, {"psi0", 0.0}, {"amplitude", 0.01}, {"seed", -1}},
      noise));
}

TEST_CASE("HigherOrderPFCSession runs a short JSON case",
          "[higher_order_pfc][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  hop::register_catalog();
  hop::HigherOrderPFCSession session(mini_session_json(), 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(pfc::time::current(session.time()) == Catch::Approx(0.1).margin(1e-12));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
