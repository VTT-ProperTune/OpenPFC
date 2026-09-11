// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_alloy_dendrite.cpp
 * @brief Tests for the thermo-solutal alloy core, equations (1)-(4).
 *
 * @details
 * Three layers, deliberately:
 *
 *  1. **Closed-form checks of the relations** in `parameters.hpp`. Cheap, and
 *     they catch a typo in a constant that would otherwise be absorbed
 *     silently by a loose band on a physics test.
 *  2. **The measurement code against analytic input.** `measure_planar_front`
 *     is handed a profile that is exactly the sharp-interface solution, and
 *     `measure_tip` is handed an exact parabola. If a Stage-1 number ever
 *     disagrees with theory, these say whether the model or the ruler is
 *     wrong -- which is the first question and normally the expensive one.
 *  3. **A real Stage-1 planar run** at a small size, asserting on
 *     conservation, `k_eff`, the boundary layer and the velocity. This runs
 *     the shipped `run_planar`, not a copy of it.
 *
 * Plus a check that the elastic hook is actually wired, since the whole
 * point of leaving equations (5)-(7) out was to leave a working attachment
 * point behind, and an attachment point nothing tests is a comment.
 *
 * Bands are set from the measured convergence study in the app README, at
 * roughly three times the observed error, and every one of them is
 * annotated with what was measured. A band with no measurement behind it is
 * a wish.
 */

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <alloy_dendrite/cases.hpp>
#include <alloy_dendrite/diagnostics.hpp>
#include <alloy_dendrite/parameters.hpp>
#include <alloy_dendrite/step.hpp>

using Catch::Approx;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Closed-form relations
// ---------------------------------------------------------------------------

TEST_CASE("thin-interface relations are self-consistent", "[unit][params]") {
  alloy_dendrite::ModelParams p;
  p.D_l = 2.0;
  p.lambda = 1.0;
  p.k = 0.15;

  // beta vanishes exactly at the classical "vanishing kinetics" coupling.
  auto q = p;
  q.lambda = p.D_l * p.tau0 / (alloy_dendrite::kA2 * p.W0 * p.W0);
  REQUIRE(alloy_dendrite::kinetic_coefficient(q) == Approx(0.0).margin(1e-15));

  // d0 = a1 W0 / lambda.
  REQUIRE(alloy_dendrite::capillary_length(p) ==
          Approx(alloy_dendrite::kA1 / p.lambda));

  // Omega(V) and V(Omega) are inverses.
  for (double v : {0.01, 0.05, 0.2}) {
    const double om = alloy_dendrite::planar_supersaturation(p, v);
    REQUIRE(alloy_dendrite::planar_steady_velocity(p, om) == Approx(v));
    REQUIRE(om > 1.0);
  }

  // The anti-trapping coefficient is the one that cancels the solid-side
  // solute gradient: U'(phi=+1) is proportional to 1 - 2 sqrt(2) a_t.
  REQUIRE(1.0 - 2.0 * std::sqrt(2.0) * alloy_dendrite::kAntiTrapCoeff ==
          Approx(0.0).margin(1e-15));

  // P(phi) is 1 in the liquid, k in the solid, and never zero in between.
  REQUIRE(alloy_dendrite::solute_prefactor(p.k, -1.0) == Approx(1.0));
  REQUIRE(alloy_dendrite::solute_prefactor(p.k, +1.0) == Approx(p.k));
  REQUIRE(alloy_dendrite::solute_mobility(+1.0) == Approx(0.0));
}

TEST_CASE("cubic anisotropy matches equation (1)", "[unit][aniso]") {
  alloy_dendrite::ModelParams p;
  p.eps4 = 0.2;
  p.W0 = 1.0;

  SECTION("isotropic when eps4 is zero") {
    auto q = p;
    q.eps4 = 0.0;
    const auto a = alloy_dendrite::evaluate_anisotropy<2>(q, 0.3, -0.7, 0.0);
    REQUIRE(a.a_s == Approx(1.0));
    REQUIRE(a.W == Approx(q.W0));
    REQUIRE(a.tau == Approx(q.tau0));
    REQUIRE(a.flux[0] == Approx(0.0));
    REQUIRE(a.flux[1] == Approx(0.0));
  }

  SECTION("axis-aligned normal is a stationary point of a_s") {
    // n = (1, 0): sum n_i^4 = 1, so a_s = 1 + eps4 and the flux, which is
    // proportional to n_i^3 - (sum n_j^4) n_i, vanishes identically.
    const auto a = alloy_dendrite::evaluate_anisotropy<2>(p, -2.5, 0.0, 0.0);
    REQUIRE(a.a_s == Approx(1.0 + p.eps4));
    REQUIRE(a.W == Approx(p.W0 * (1.0 + p.eps4)));
    REQUIRE(a.tau == Approx(p.tau0 * (1.0 + p.eps4) * (1.0 + p.eps4)));
    REQUIRE(a.flux[0] == Approx(0.0).margin(1e-14));
    REQUIRE(a.flux[1] == Approx(0.0).margin(1e-14));
  }

  SECTION("diagonal normal is the other stationary point") {
    // n = (1,1)/sqrt(2): sum n_i^4 = 1/2, the minimum of a_s in 2-D.
    const auto a = alloy_dendrite::evaluate_anisotropy<2>(p, 1.0, 1.0, 0.0);
    REQUIRE(a.a_s == Approx(1.0 + 0.5 * p.eps4));
    REQUIRE(a.flux[0] == Approx(0.0).margin(1e-14));
    REQUIRE(a.flux[1] == Approx(0.0).margin(1e-14));
  }

  SECTION("a_s depends only on direction; the flux scales linearly") {
    const auto a1 = alloy_dendrite::evaluate_anisotropy<2>(p, 0.3, 0.8, 0.0);
    const auto a2 = alloy_dendrite::evaluate_anisotropy<2>(p, 3.0, 8.0, 0.0);
    REQUIRE(a2.a_s == Approx(a1.a_s));
    // A_i = |grad phi|^2 W dW/d(d_i phi) carries exactly one spare power of
    // the gradient, which is why it needs no regularisation.
    REQUIRE(a2.flux[0] == Approx(10.0 * a1.flux[0]));
    REQUIRE(a2.flux[1] == Approx(10.0 * a1.flux[1]));
  }

  SECTION("vanishing gradient is handled without a 0/0") {
    const auto a = alloy_dendrite::evaluate_anisotropy<2>(p, 0.0, 0.0, 0.0);
    REQUIRE(std::isfinite(a.a_s));
    REQUIRE(std::isfinite(a.flux[0]));
    REQUIRE(a.flux[0] == Approx(0.0));
  }

  SECTION("3-D <100> and <111>") {
    const auto ax = alloy_dendrite::evaluate_anisotropy<3>(p, 1.0, 0.0, 0.0);
    REQUIRE(ax.a_s == Approx(1.0 + p.eps4));
    const auto ad = alloy_dendrite::evaluate_anisotropy<3>(p, 1.0, 1.0, 1.0);
    REQUIRE(ad.a_s == Approx(1.0 + p.eps4 / 3.0));
  }
}

// ---------------------------------------------------------------------------
// 2. The measurements, against analytic input
// ---------------------------------------------------------------------------

TEST_CASE("measure_planar_front recovers the sharp-interface solution",
          "[unit][diagnostics]") {
  alloy_dendrite::ModelParams p;
  p.k = 0.15;
  p.W0 = 1.0;
  p.D_l = 2.0;

  const int nx = 800;
  const double dx = 0.5;
  const double xc = 0.5 * nx * dx;
  const double half = 20.0;
  const double v = 0.1;
  const double ell = p.D_l / v;
  const double u_i = -alloy_dendrite::kinetic_coefficient(p) * v;
  const double u_inf = -alloy_dendrite::planar_supersaturation(p, v);

  std::vector<double> phi(static_cast<std::size_t>(nx));
  std::vector<double> u(static_cast<std::size_t>(nx));
  for (int i = 0; i < nx; ++i) {
    const double x = i * dx;
    const double d = std::fabs(x - xc) - half;
    phi[static_cast<std::size_t>(i)] = std::tanh(-d / std::sqrt(2.0));
    u[static_cast<std::size_t>(i)] =
        (d <= 0.0) ? u_i : (u_inf + (u_i - u_inf) * std::exp(-d / ell));
  }

  const auto f = alloy_dendrite::measure_planar_front(phi, u, dx, p);
  REQUIRE(f.valid);
  // The zero crossing of the analytic profile sits exactly at xc + half.
  REQUIRE(f.x_if == Approx(xc + half).margin(1e-3));
  // The log-linear fit must reproduce the seeded decay length and the
  // interface value it was extrapolated from.
  REQUIRE(f.ell == Approx(ell).epsilon(2e-3));
  REQUIRE(f.u_interface == Approx(u_i).margin(2e-3));
  REQUIRE(f.u_solid == Approx(u_i).margin(1e-9));
  // U_s == U_i is exactly the statement k_eff == k.
  REQUIRE(f.k_eff == Approx(p.k).epsilon(5e-3));
  REQUIRE(f.fit_r2 > 0.9999);
}

TEST_CASE("measure_planar_front reports failure rather than a number",
          "[unit][diagnostics]") {
  alloy_dendrite::ModelParams p;
  // All-liquid profile: no front, so there is nothing to measure.
  std::vector<double> phi(400, -1.0);
  std::vector<double> u(400, -0.5);
  const auto f = alloy_dendrite::measure_planar_front(phi, u, 0.5, p);
  REQUIRE_FALSE(f.valid);
  REQUIRE(std::isnan(f.k_eff));
}

TEST_CASE("measure_tip recovers an exact parabola", "[unit][diagnostics]") {
  const int nx = 200;
  const int ny = 160;
  const double dx = 0.5;
  const int i_seed = 40;
  const int j_seed = ny / 2;
  const double rho = 12.0;
  const double x_tip = 70.0;
  const double y_tip = j_seed * dx;

  // A piecewise-linear profile across the front, so the linear interpolation
  // of the zero crossing is exact and this test measures the parabola fit
  // alone rather than the interpolation on top of it.
  std::vector<double> phi(static_cast<std::size_t>(nx * ny));
  for (int j = 0; j < ny; ++j) {
    const double dy = j * dx - y_tip;
    const double xf = x_tip - dy * dy / (2.0 * rho);
    for (int i = 0; i < nx; ++i) {
      const double s = (xf - i * dx) / dx;
      phi[static_cast<std::size_t>(i + j * nx)] = std::fmax(-1.0, std::fmin(1.0, s));
    }
  }

  const auto t = alloy_dendrite::measure_tip(phi, nx, ny, dx, dx, i_seed, j_seed, 5);
  REQUIRE(t.valid);
  REQUIRE(t.x_tip == Approx(x_tip).margin(1e-9));
  REQUIRE(t.y_tip == Approx(y_tip).margin(1e-12));
  REQUIRE(t.rho == Approx(rho).epsilon(1e-9));
  REQUIRE(t.fit_rms < 1e-9);
  REQUIRE(t.fit_rows == 11);
}

TEST_CASE("trailing_slope is a fit, not a difference", "[unit][diagnostics]") {
  std::vector<double> t, x;
  for (int i = 0; i < 100; ++i) {
    t.push_back(0.1 * i);
    // A clean ramp plus a sawtooth the size of one cell: this is what grid
    // pinning does to a level-set crossing, and a two-point difference of it
    // is wrong by 100%.
    x.push_back(0.25 * t.back() + ((i % 3) - 1) * 0.05);
  }
  REQUIRE(alloy_dendrite::trailing_slope(t, x, 0.5) == Approx(0.25).epsilon(0.02));
}

// ---------------------------------------------------------------------------
// 3. The real thing
// ---------------------------------------------------------------------------

namespace {

/// Small Stage-1 case: a `384 W0` box at `dx = 0.6 W0`, about ten seconds per
/// run. Same physics as the shipped verification at a quarter of the end
/// time; the half-box is still 9.6 solute boundary layers, which is above the
/// eight-layer floor where the two fronts' tails start biasing `U_far`.
alloy_dendrite::PlanarConfig small_planar_case() {
  alloy_dendrite::PlanarConfig cfg;
  cfg.nx = 640;
  cfg.ny = 4;
  cfg.dx = 0.6;
  cfg.fd_order = 4;
  cfg.t_end = 150.0;
  cfg.n_sample = 100;
  cfg.velocity_target = 0.1;
  cfg.quiet = true;
  cfg.run_id = "ctest";
  return cfg;
}

} // namespace

TEST_CASE("planar front: solute and latent heat are conserved to round-off",
          "[planar]") {
  const auto cfg = small_planar_case();
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const auto r = alloy_dendrite::run_planar(cfg, rank, nproc, MPI_COMM_WORLD);
  REQUIRE(r.valid);

  // Both invariants are exact identities of the discretisation (see
  // step.hpp), so the only thing that accumulates is round-off in a sum over
  // ~2600 cells and 13000 steps. Measured 6e-15; 1e-11 leaves three decades
  // of headroom for a different summation order on another machine while
  // still failing hard if a term stops telescoping.
  CHECK(r.solute_drift_rel < 1e-11);
  CHECK(r.heat_drift_rel < 1e-11);
  // phi must stay in [-1, 1] to well within the overshoot a fourth-order
  // stencil produces at a tanh front (measured 1.4e-4).
  CHECK(r.phi_min > -1.01);
  CHECK(r.phi_max < 1.01);
}

TEST_CASE("planar front: k_eff equals k, which is the anti-trapping test",
          "[planar]") {
  const auto cfg = small_planar_case();
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const auto r = alloy_dendrite::run_planar(cfg, rank, nproc, MPI_COMM_WORLD);
  REQUIRE(r.valid);

  // Measured at this resolution and end time: k_eff / k - 1 = -0.37%, and
  // -0.49% to +0.16% across the whole dx/W0 = 0.8 .. 0.4 study. The band is 2%.
  // Switching the anti-trapping current off (at_scale = 0) moves this to
  // +7.8% at V = 0.05 and +47% at V = 0.2, and flipping its sign moves it to
  // +16% and beyond, so 2% separates the right model from both wrong ones by
  // a wide margin at every velocity tested.
  CHECK(r.k_eff == Approx(cfg.model.k).epsilon(0.02));
  // The boundary layer must be D_l / V. Measured -0.75%.
  CHECK(r.ell_measured == Approx(cfg.model.D_l / r.v_measured).epsilon(0.02));
  // Velocity against the thin-interface prediction. Measured -0.33% here and
  // -0.19% in the full-length dx = 0.6 W0 run; the band is 5%.
  CHECK(r.v_measured == Approx(r.v_predicted).epsilon(0.05));
}

TEST_CASE("planar front: switching the anti-trapping current off is visible",
          "[planar]") {
  // The point of a verification app is that the failure mode is detectable,
  // not merely that the good case passes. If this ever stops failing, the
  // anti-trapping current has stopped doing anything.
  auto cfg = small_planar_case();
  cfg.model.at_scale = 0.0;
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const auto r = alloy_dendrite::run_planar(cfg, rank, nproc, MPI_COMM_WORLD);
  REQUIRE(r.valid);
  // Measured +12.8% at this velocity and end time with the current off, and
  // +17% in the full-length run; it grows to +65% at V = 0.39.
  CHECK(r.k_eff > 1.05 * cfg.model.k);
  // Conservation is a property of the discretisation, not of the physics,
  // so it must hold even for a model that is wrong.
  CHECK(r.solute_drift_rel < 1e-11);
}

TEST_CASE("planar front is deterministic", "[planar]") {
  auto cfg = small_planar_case();
  cfg.nx = 200;
  cfg.t_end = 20.0;
  cfg.n_sample = 10;
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const auto a = alloy_dendrite::run_planar(cfg, rank, nproc, MPI_COMM_WORLD);
  const auto b = alloy_dendrite::run_planar(cfg, rank, nproc, MPI_COMM_WORLD);
  // Bitwise, not approximately: nothing in the step is order-dependent on a
  // fixed rank count, and a later agent's science runs depend on that.
  REQUIRE(a.v_measured == b.v_measured);
  REQUIRE(a.k_eff == b.k_eff);
  REQUIRE(a.solute_drift_rel == b.solute_drift_rel);
}

// ---------------------------------------------------------------------------
// 3b. The third dimension
// ---------------------------------------------------------------------------

TEST_CASE("a z-invariant 3-D run reproduces the 2-D run", "[dim3]") {
  // The strongest statement available about the 3-D path without a reference
  // solution: for a configuration with no z-dependence, every z term in
  // equations (1)-(4) is identically zero, so Stepper<3> must reproduce
  // Stepper<2> to round-off. If it does not, either a z derivative is being
  // taken of an unexchanged halo or the anisotropy is not reducing correctly
  // -- both silent failures that a 3-D dendrite picture would not reveal.
  if (world_size() != 1) {
    SKIP("single-rank dimensional-consistency check");
  }
  alloy_dendrite::ModelParams p;
  p.eps4 = 0.2;
  p.D_th = 1.0;
  p.M_c = 0.5;
  p.evolve_theta = true;

  auto run = [&](int nz) {
    const int nx = 48;
    const int ny = 32;
    const double dx = 0.8;
    auto domain = pfc::domain::create(pfc::GridSize({nx, ny, nz}),
                                      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                      pfc::GridSpacing({dx, dx, dx}));
    // A cylinder along z, i.e. a 2-D disc extruded: no z-dependence anywhere.
    auto init = [&](auto &st) {
      const double xc = 0.5 * nx * dx;
      const double yc = 0.5 * ny * dx;
      st.phi().for_each_owned([&](int i, int j, int kk) {
        const auto c = st.phi().coords(i, j, kk);
        const double r = std::hypot(c[0] - xc, c[1] - yc);
        st.phi()(i, j, kk) = std::tanh((6.0 - r) / std::sqrt(2.0));
        st.solute()(i, j, kk) = -0.6;
        st.temperature()(i, j, kk) = 0.0;
      });
      st.seed_conserved_solute();
    };
    double sums[3] = {0.0, 0.0, 0.0};
    if (nz == 1) {
      pfc::comm::HaloExchangeOptions opt;
      opt.directions = alloy_dendrite::Stepper<2>::directions();
      pfc::sim::stacks::FDPaddedCPUStack stack(domain, 2, 0, 1, MPI_COMM_WORLD, opt);
      alloy_dendrite::Stepper<2> st(stack, p, 4);
      init(st);
      for (int n = 0; n < 40; ++n) {
        st.step(0.01);
      }
      st.phi().for_each_owned([&](int i, int j, int kk) {
        sums[0] += st.phi()(i, j, kk);
        sums[1] += st.solute()(i, j, kk);
        sums[2] += st.temperature()(i, j, kk);
      });
    } else {
      pfc::comm::HaloExchangeOptions opt;
      opt.directions = alloy_dendrite::Stepper<3>::directions();
      pfc::sim::stacks::FDPaddedCPUStack stack(domain, 2, 0, 1, MPI_COMM_WORLD, opt);
      alloy_dendrite::Stepper<3> st(stack, p, 4);
      init(st);
      for (int n = 0; n < 40; ++n) {
        st.step(0.01);
      }
      st.phi().for_each_owned([&](int i, int j, int kk) {
        sums[0] += st.phi()(i, j, kk);
        sums[1] += st.solute()(i, j, kk);
        sums[2] += st.temperature()(i, j, kk);
      });
      for (double &v : sums) {
        v /= static_cast<double>(nz);
      }
    }
    return std::array<double, 3>{sums[0], sums[1], sums[2]};
  };

  const auto a = run(1);
  const auto b = run(8);
  for (int q = 0; q < 3; ++q) {
    // Round-off only: the z stencils act on a constant, whose exact
    // derivative is zero and whose floating-point derivative is 1e-16.
    REQUIRE(b[q] == Approx(a[q]).epsilon(1e-11).margin(1e-11));
  }
}

TEST_CASE("3-D dendrite smoke", "[dim3]") {
  // Small and short: this asserts that the 3-D driver path runs, conserves,
  // and finds a tip, not that the morphology is right. Stage 3 of the
  // verification ladder needs a machine, not a ctest.
  alloy_dendrite::DendriteConfig cfg;
  cfg.model.D_l = 2.0;
  cfg.model.k = 0.15;
  cfg.model.lambda = cfg.model.D_l / alloy_dendrite::kA2;
  cfg.model.D_th = 2.0;
  cfg.model.M_c = 0.5;
  cfg.model.eps4 = 0.2;
  cfg.nx = 48;
  cfg.ny = 48;
  cfg.nz = 48;
  cfg.dx = 1.0;
  cfg.t_end = 8.0;
  cfg.n_sample = 8;
  cfg.seed_radius = 5.0;
  cfg.tip_fit_halfwidth = 3;
  cfg.quiet = true;
  cfg.run_id = "ctest3d";
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const auto r = alloy_dendrite::run_dendrite_case(cfg, rank, nproc, MPI_COMM_WORLD);
  REQUIRE(r.valid);
  CHECK(r.solute_drift_rel < 1e-11);
  CHECK(r.heat_drift_rel < 1e-11);
  CHECK(r.v_tip > 0.0);
  CHECK(r.rho_tip > 0.0);
  CHECK(r.phi_min > -1.01);
  CHECK(r.phi_max < 1.01);
}

// ---------------------------------------------------------------------------
// 4. The elastic hook
// ---------------------------------------------------------------------------

TEST_CASE("the elastic driving-force hook is wired", "[unit][elastic-hook]") {
  // Equations (5)-(7) are not implemented here; what must survive until they
  // are is the ability to inject dF_el/dphi into equation (2) without
  // touching the stepper. This drives the hook with a constant field, which
  // is not physics -- it is a wiring check.
  if (world_size() != 1) {
    SKIP("single-rank hook check");
  }
  alloy_dendrite::ModelParams p;
  p.eps4 = 0.0;
  p.evolve_theta = false;

  auto run = [&](double lambda_el, const bool install) {
    auto q = p;
    q.lambda_el = lambda_el;
    auto domain = pfc::domain::create(pfc::GridSize({64, 4, 1}),
                                      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                      pfc::GridSpacing({0.8, 0.8, 0.8}));
    pfc::comm::HaloExchangeOptions opt;
    opt.directions = alloy_dendrite::Stepper<2>::directions();
    pfc::sim::stacks::FDPaddedCPUStack stack(domain, 2, 0, 1, MPI_COMM_WORLD, opt);
    alloy_dendrite::Stepper<2> st(stack, q, 4);
    auto dfel = stack.make_field();
    st.phi().for_each_owned([&](int i, int j, int kk) {
      const double x = st.phi().coords(i, j, kk)[0];
      st.phi()(i, j, kk) = std::tanh((12.0 - std::fabs(x - 25.6)) / std::sqrt(2.0));
      st.solute()(i, j, kk) = -1.0;
      st.temperature()(i, j, kk) = 0.0;
      dfel(i, j, kk) = 1.0;
    });
    st.seed_conserved_solute();
    if (install) {
      st.set_elastic_driving_force(&dfel);
    }
    for (int n = 0; n < 50; ++n) {
      st.step(0.01);
    }
    double s = 0.0;
    st.phi().for_each_owned([&](int i, int j, int kk) { s += st.phi()(i, j, kk); });
    return s;
  };

  const double base = run(0.0, false);
  // Installed but with lambda_el = 0: the term must be exactly absent, not
  // merely small, or a default-configured run would silently pay for it.
  REQUIRE(run(0.0, true) == base);
  // Not installed but with lambda_el != 0: no field, no term.
  REQUIRE(run(0.5, false) == base);
  // Both: the term fires. A positive dF_el/dphi opposes solidification, so
  // the total phi must come out lower.
  const double driven = run(0.5, true);
  REQUIRE(driven != base);
  REQUIRE(driven < base);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
