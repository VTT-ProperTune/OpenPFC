// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_homogenization.cpp
 * @brief Forward C_H oracles and a discrete-sensitivity finite-difference check.
 *
 * @details
 * Stages 1 and 4 of issue #161. Every expected value is derived in the
 * comment above the test that uses it. Decreasing an inverse-design
 * objective is not an oracle for the gradient.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/homogenization.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::engineering_unit_strain;
using pfc::apps::exact_binary_laminate;
using pfc::apps::exact_laminate_from_field;
using pfc::apps::frobenius_inner;
using pfc::apps::HomogenizationResult;
using pfc::apps::is_spd;
using pfc::apps::kVoigtDim;
using pfc::apps::MicroelasticityParams;
using pfc::apps::PeriodicHomogenizer;
using pfc::apps::reuss_bound;
using pfc::apps::Stiffness;
using pfc::apps::tensor_mismatch;
using pfc::apps::voigt_bound;
using pfc::apps::voigt_from_stiffness;
using pfc::apps::Voigt6;

using RealField = pfc::data::Field<double>;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

template <class... Ts> std::string precise(Ts &&...parts) {
  std::ostringstream oss;
  oss << std::setprecision(12);
  (oss << ... << parts);
  return oss.str();
}

pfc::Domain cube(int n, double dx) {
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, dx}));
}

template <class Fn> void fill(RealField &f, Fn &&fn) {
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = f.coords(i, j, k);
        f(i, j, k) = fn(x[0], x[1], x[2]);
      }
    }
  }
  f.note_host_write();
}

void fill_value(RealField &f, double v) {
  std::fill(f.vec().begin(), f.vec().end(), v);
  f.note_host_write();
}

double max_abs_diff(const Voigt6 &a, const Voigt6 &b) {
  double m = 0.0;
  for (int i = 0; i < kVoigtDim; ++i)
    for (int j = 0; j < kVoigtDim; ++j)
      m = std::max(m, std::abs(a(i, j) - b(i, j)));
  return m;
}

double rel_frobenius(const Voigt6 &got, const Voigt6 &want) {
  const double n = want.frobenius_norm();
  return (n > 0.0) ? (got - want).frobenius_norm() / n : got.frobenius_norm();
}

struct Case {
  pfc::Domain domain;
  pfc::sim::stacks::SpectralCPUStack stack;
  RealField h, dJdh, hdir;

  explicit Case(int n, double dx = 1.0)
      : domain(cube(n, dx)),
        stack(domain, world_rank(), world_size(), MPI_COMM_WORLD),
        h(pfc::data::field_from_inbox<double>(domain,
                                              stack.fft().get_inbox_bounds())),
        dJdh(pfc::data::field_from_inbox<double>(domain,
                                                 stack.fft().get_inbox_bounds())),
        hdir(pfc::data::field_from_inbox<double>(domain,
                                                 stack.fft().get_inbox_bounds())) {}
};

MicroelasticityParams two_phase(const Stiffness &cs, const Stiffness &cl) {
  MicroelasticityParams p;
  p.c_solid = cs;
  p.c_liquid = cl;
  p.tol_el = 1.0e-10;
  p.n_el_iter = 80;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;
  return p;
}

} // namespace

// ---------------------------------------------------------------------------
// Algebra of the Voigt convention and the laminate closed form (no FFT)
// ---------------------------------------------------------------------------
//
// Engineering C_44 = c_44 = mu, not 2 mu. A round-trip through the inverse
// must recover the input. The laminate numbers for isotropic (lambda, mu) =
// (1,1) and (2,2) at f = 1/2, normal z, are derived in the issue
// implementation notes:
//   C33 = 4, C13 = 4/3, C11 = 40/9, C12 = 13/9, C44 = C55 = 4/3, C66 = 3/2.
TEST_CASE("Engineering Voigt and the binary-laminate closed form",
          "[homogenization][algebra]") {
  const Stiffness iso = Stiffness::isotropic(1.0, 0.3);
  const Voigt6 C = voigt_from_stiffness(iso);
  REQUIRE_THAT(C(0, 0), WithinRel(iso.c11, 1e-15));
  REQUIRE_THAT(C(0, 1), WithinRel(iso.c12, 1e-15));
  REQUIRE_THAT(C(3, 3), WithinRel(iso.c44, 1e-15));
  REQUIRE_THAT(C(4, 4), WithinRel(iso.c44, 1e-15));
  REQUIRE_THAT(C(5, 5), WithinRel(iso.c44, 1e-15));
  REQUIRE(C(0, 3) == 0.0);
  REQUIRE(is_spd(C));

  Voigt6 back = C;
  REQUIRE(pfc::apps::invert_voigt(back));
  Voigt6 round = back;
  REQUIRE(pfc::apps::invert_voigt(round));
  REQUIRE(max_abs_diff(round, C) < 1.0e-12);

  const Stiffness a = Stiffness::from_lame(1.0, 1.0); // c11=3, c12=1, c44=1
  const Stiffness b = Stiffness::from_lame(2.0, 2.0); // c11=6, c12=2, c44=2
  const Voigt6 L = exact_binary_laminate(a, b, 0.5, /*axis=*/2);
  REQUIRE_THAT(L(2, 2), WithinRel(4.0, 1e-14));
  REQUIRE_THAT(L(0, 2), WithinRel(4.0 / 3.0, 1e-14));
  REQUIRE_THAT(L(0, 0), WithinRel(40.0 / 9.0, 1e-14));
  REQUIRE_THAT(L(1, 1), WithinRel(40.0 / 9.0, 1e-14));
  REQUIRE_THAT(L(0, 1), WithinRel(13.0 / 9.0, 1e-14));
  REQUIRE_THAT(L(3, 3), WithinRel(4.0 / 3.0, 1e-14));
  REQUIRE_THAT(L(4, 4), WithinRel(4.0 / 3.0, 1e-14));
  REQUIRE_THAT(L(5, 5), WithinRel(1.5, 1e-14));
  REQUIRE(is_spd(L));

  const Voigt6 V = voigt_bound(a, b, 0.5);
  const Voigt6 R = reuss_bound(a, b, 0.5);
  // Iso-strain in-plane shear of a z-laminate is the Voigt average of mu.
  REQUIRE_THAT(L(5, 5), WithinRel(V(5, 5), 1e-14));
  // Iso-stress out-of-plane shear is the Reuss average of mu.
  REQUIRE_THAT(L(3, 3), WithinRel(R(3, 3), 1e-14));
  REQUIRE_THAT(L(2, 2), WithinRel(R(2, 2), 1e-14));
  REQUIRE(L(0, 0) < V(0, 0) + 1e-12);
  REQUIRE(L(0, 0) > R(0, 0) - 1e-12);

  const auto e_shear = engineering_unit_strain(5);
  REQUIRE_THAT(e_shear[5], WithinRel(0.5, 1e-15));
  REQUIRE_THAT(engineering_unit_strain(0)[0], WithinRel(1.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Homogeneous cell: C_H is the input stiffness, any volume fraction
// ---------------------------------------------------------------------------
TEST_CASE("Homogeneous unit cell recovers the interpolated stiffness",
          "[homogenization][homogeneous][mpi]") {
  constexpr int N = 16;
  Case cs(N);
  const Stiffness cs_ = Stiffness::isotropic(2.0, 0.25);
  const Stiffness cl_ = Stiffness::isotropic(0.5, 0.25);
  PeriodicHomogenizer hom(cs.domain, cs.stack.fft(), two_phase(cs_, cl_));

  const double hs[2] = {1.0, 0.4};
  for (double hv : hs) {
    fill_value(cs.h, hv);
    const auto r = hom.compute(cs.h);
    REQUIRE(r.all_converged());
    // 0.4 is not a binary fraction; the cell-average of N copies need not
    // round-trip through an MPI sum to 14 relative digits.
    REQUIRE_THAT(r.volume_fraction, WithinAbs(hv, 1.0e-12));
    const Stiffness c = Stiffness::blend(cs_, hv, cl_, 1.0 - hv);
    const Voigt6 want = voigt_from_stiffness(c);
    INFO(precise("h = ", hv, " rel F = ", rel_frobenius(r.stiffness, want)));
    REQUIRE(rel_frobenius(r.stiffness, want) < 1.0e-10);
    REQUIRE(is_spd(r.stiffness));
    for (int a = 0; a < kVoigtDim; ++a) {
      const auto applied = engineering_unit_strain(a);
      for (int cidx = 0; cidx < kVoigtDim; ++cidx) {
        REQUIRE_THAT(r.mean_strain[static_cast<std::size_t>(a)][cidx],
                     WithinAbs(applied[cidx], 1.0e-12));
      }
      REQUIRE(r.reports[static_cast<std::size_t>(a)].iterations == 1);
    }
  }
}

// ---------------------------------------------------------------------------
// Smooth z-laminate vs the 1-D closed form on the same h(z)
// ---------------------------------------------------------------------------
//
// A tanh interface is a graded laminate. The exact C_H is the Backus average
// of C(h(z)), not the two-phase formula at <h>. FFT homogenization of a
// z-only field must reproduce that average: there is no x- or y-fluctuation
// to resolve.
TEST_CASE("Smooth z-laminate matches the 1-D Backus average of C(h(z))",
          "[homogenization][laminate]") {
  constexpr int N = 16;
  Case cs(N);
  const Stiffness cs_ = Stiffness::from_lame(1.0, 1.0);
  const Stiffness cl_ = Stiffness::from_lame(2.0, 2.0);
  const double z0 = 0.5 * N;
  const double w = 1.5;
  fill(cs.h, [&](double, double, double z) {
    return 0.5 * (1.0 - std::tanh((z - z0) / w));
  });

  PeriodicHomogenizer hom(cs.domain, cs.stack.fft(), two_phase(cs_, cl_));
  const auto r = hom.compute(cs.h);
  REQUIRE(r.all_converged());
  REQUIRE(is_spd(r.stiffness));

  const Voigt6 want =
      exact_laminate_from_field(cs.h, cs_, cl_, /*axis=*/2, MPI_COMM_WORLD);
  const double rel = rel_frobenius(r.stiffness, want);
  INFO(precise("rel F = ", rel, " max abs = ", max_abs_diff(r.stiffness, want)));
  REQUIRE(rel < 1.0e-6);

  const Voigt6 V = voigt_bound(cs_, cl_, r.volume_fraction);
  const Voigt6 R = reuss_bound(cs_, cl_, r.volume_fraction);
  REQUIRE(r.stiffness(5, 5) <= V(5, 5) + 1.0e-8);
  REQUIRE(r.stiffness(3, 3) >= R(3, 3) - 1.0e-8);
}

// ---------------------------------------------------------------------------
// Sharp laminate: two-phase Postma formula, moderate contrast
// ---------------------------------------------------------------------------
TEST_CASE("Sharp z-laminate matches the two-phase Postma stiffness",
          "[homogenization][laminate]") {
  constexpr int N = 16;
  Case cs(N);
  const Stiffness cs_ = Stiffness::isotropic(1.0, 0.3);
  const Stiffness cl_ = Stiffness::isotropic(0.25, 0.3);
  fill(cs.h, [&](double, double, double z) { return (z < 0.5 * N) ? 1.0 : 0.0; });

  PeriodicHomogenizer hom(cs.domain, cs.stack.fft(), two_phase(cs_, cl_));
  const auto r = hom.compute(cs.h);
  REQUIRE(r.all_converged());
  REQUIRE_THAT(r.volume_fraction, WithinAbs(0.5, 1e-14));

  const Voigt6 want = exact_binary_laminate(cs_, cl_, 0.5, /*axis=*/2);
  const double rel = rel_frobenius(r.stiffness, want);
  INFO(precise("rel F = ", rel));
  // Sharp interfaces ring in the trigonometric basis; the *averages* that
  // make C_H still converge. Contrast 4 on 16^3 is well inside 1e-3.
  REQUIRE(rel < 5.0e-3);
}

// ---------------------------------------------------------------------------
// Cubic symmetry of a centred inclusion
// ---------------------------------------------------------------------------
TEST_CASE("A centred spherical inclusion produces a cubic C_H",
          "[homogenization][symmetry]") {
  constexpr int N = 16;
  Case cs(N);
  const Stiffness cs_ = Stiffness::isotropic(2.0, 0.3);
  const Stiffness cl_ = Stiffness::isotropic(0.5, 0.3);
  const double cx = 0.5 * N, R = 4.0, w = 1.2;
  fill(cs.h, [&](double x, double y, double z) {
    const double r =
        std::sqrt((x - cx) * (x - cx) + (y - cx) * (y - cx) + (z - cx) * (z - cx));
    return 0.5 * (1.0 - std::tanh((r - R) / w));
  });

  PeriodicHomogenizer hom(cs.domain, cs.stack.fft(), two_phase(cs_, cl_));
  const auto r = hom.compute(cs.h);
  REQUIRE(r.all_converged());
  REQUIRE(is_spd(r.stiffness));
  const Voigt6 &C = r.stiffness;
  const double scale = C.max_abs();
  REQUIRE_THAT(C(0, 0), WithinRel(C(1, 1), 2e-3));
  REQUIRE_THAT(C(0, 0), WithinRel(C(2, 2), 2e-3));
  REQUIRE_THAT(C(3, 3), WithinRel(C(4, 4), 2e-3));
  REQUIRE_THAT(C(3, 3), WithinRel(C(5, 5), 2e-3));
  REQUIRE_THAT(C(0, 1), WithinRel(C(0, 2), 5e-3));
  REQUIRE_THAT(C(0, 1), WithinRel(C(1, 2), 5e-3));
  REQUIRE_THAT(C(0, 3), WithinAbs(0.0, 1e-3 * scale));
  REQUIRE_THAT(C(0, 4), WithinAbs(0.0, 1e-3 * scale));
  REQUIRE_THAT(C(0, 5), WithinAbs(0.0, 1e-3 * scale));
}

// ---------------------------------------------------------------------------
// Sensitivity, homogeneous: closed form
// ---------------------------------------------------------------------------
//
// Uniform h, C_H = C(h), ∂C_H/∂h_e = ΔC / N. For J = 1/2 ||C_H||_F^2
// (target 0, W = 1) the directional derivative along a uniform shift is
//   dJ/dα |_{h+α} = <C, ΔC>_F.
TEST_CASE("Homogeneous sensitivity matches the closed-form dC/dh",
          "[homogenization][sensitivity]") {
  constexpr int N = 8;
  Case cs(N);
  const Stiffness cs_ = Stiffness::isotropic(2.0, 0.2);
  const Stiffness cl_ = Stiffness::isotropic(0.5, 0.2);
  const double hv = 0.35;
  fill_value(cs.h, hv);
  fill_value(cs.hdir, 1.0);

  PeriodicHomogenizer hom(cs.domain, cs.stack.fft(), two_phase(cs_, cl_));
  const auto r = hom.compute(cs.h);
  REQUIRE(r.all_converged());
  const Voigt6 W = Voigt6::ones();
  const Voigt6 Cstar{};
  hom.objective_sensitivity(cs.h, Cstar, W, cs.dJdh);
  const double DJ = hom.directional_derivative(cs.dJdh, cs.hdir);

  const Voigt6 dC = voigt_from_stiffness(
      Stiffness::blend(cs_, 1.0, cl_, -1.0));
  const double want = frobenius_inner(r.stiffness, dC);
  INFO(precise("DJ = ", DJ, " want ", want));
  REQUIRE_THAT(DJ, WithinRel(want, 1.0e-8));
  REQUIRE_THAT(hom.objective(r, Cstar, W),
               WithinRel(0.5 * frobenius_inner(r.stiffness, r.stiffness), 1e-14));
}

// ---------------------------------------------------------------------------
// Sensitivity, random h: finite-difference directional derivatives
// ---------------------------------------------------------------------------
//
// Central differences of J along three random mean-zero directions, swept
// in step size. The error vs eta is U-shaped (truncation then round-off).
// Acceptance: at the best eta, relative error < 1e-4 on every direction.
TEST_CASE("Adjoint sensitivity matches central finite differences",
          "[homogenization][sensitivity][fd]") {
  constexpr int N = 8;
  Case cs(N);
  const Stiffness cs_ = Stiffness::isotropic(1.5, 0.25);
  const Stiffness cl_ = Stiffness::isotropic(0.4, 0.25);
  PeriodicHomogenizer hom(cs.domain, cs.stack.fft(), two_phase(cs_, cl_));

  std::mt19937 rng(20260911u + static_cast<unsigned>(world_rank()));
  std::uniform_real_distribution<double> dist(0.25, 0.75);
  fill(cs.h, [&](double, double, double) { return dist(rng); });

  const Voigt6 W = Voigt6::ones();
  Voigt6 Cstar = voigt_from_stiffness(Stiffness::isotropic(0.8, 0.25));
  const auto r0 = hom.compute(cs.h);
  REQUIRE(r0.all_converged());
  hom.objective_sensitivity(cs.h, Cstar, W, cs.dJdh);
  const double J0 = hom.objective(r0, Cstar, W);

  const double etas[] = {1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4};
  for (int dir = 0; dir < 3; ++dir) {
    std::uniform_real_distribution<double> ddir(-1.0, 1.0);
    fill(cs.hdir, [&](double, double, double) { return ddir(rng); });
    // Remove the mean so the perturbation is volume-preserving; the
    // formula does not need that, but it keeps h in (0,1) more easily.
    double local_mean = 0.0;
    for (std::size_t i = 0; i < cs.hdir.size(); ++i) local_mean += cs.hdir.data()[i];
    double gmean = 0.0;
    MPI_Allreduce(&local_mean, &gmean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    gmean /= static_cast<double>(N) * N * N;
    for (std::size_t i = 0; i < cs.hdir.size(); ++i)
      cs.hdir.data()[i] -= gmean;
    cs.hdir.note_host_write();

    const double analytic = hom.directional_derivative(cs.dJdh, cs.hdir);
    REQUIRE(std::abs(analytic) > 1.0e-16);

    double best = 1.0;
    double best_eta = 0.0;
    RealField hp(pfc::data::field_from_inbox<double>(
        cs.domain, cs.stack.fft().get_inbox_bounds()));
    RealField hm(pfc::data::field_from_inbox<double>(
        cs.domain, cs.stack.fft().get_inbox_bounds()));
    for (double eta : etas) {
      for (std::size_t i = 0; i < cs.h.size(); ++i) {
        hp.data()[i] = cs.h.data()[i] + eta * cs.hdir.data()[i];
        hm.data()[i] = cs.h.data()[i] - eta * cs.hdir.data()[i];
      }
      hp.note_host_write();
      hm.note_host_write();
      const double Jp = hom.objective(hom.compute(hp), Cstar, W);
      const double Jm = hom.objective(hom.compute(hm), Cstar, W);
      const double fd = (Jp - Jm) / (2.0 * eta);
      const double rel = std::abs(fd - analytic) / std::abs(analytic);
      if (rel < best) {
        best = rel;
        best_eta = eta;
      }
      INFO(precise("dir ", dir, " eta ", eta, " analytic ", analytic, " fd ", fd,
                   " rel ", rel, " J0 ", J0));
    }
    INFO(precise("dir ", dir, " best rel ", best, " at eta ", best_eta));
    REQUIRE(best < 1.0e-4);
    // Restore the base fields used by the gradient (compute(hp) overwrote
    // the stored strains).
    hom.compute(cs.h);
    hom.objective_sensitivity(cs.h, Cstar, W, cs.dJdh);
  }
}
