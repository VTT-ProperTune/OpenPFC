// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_load_case_compliance.cpp
 * @brief SIMP interpolation and the stored energy of one load case:
 *        homogeneous closed form, z-continuity, and the self-adjoint
 *        gradient against a finite difference of re-converged solves.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/solvers/microelasticity/homogenization.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::solvers::LoadCaseCompliance;
using pfc::solvers::MicroelasticityParams;
using pfc::solvers::SimpInterpolation;
using pfc::solvers::Stiffness;
using pfc::solvers::Sym3;
using pfc::solvers::SYM_ZZ;
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
double gsum(double v) {
  double out = 0.0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

struct Cell {
  pfc::Domain domain;
  pfc::sim::stacks::SpectralCPUStack stack;
  RealField phi;
  explicit Cell(int nx, int ny, int nz)
      : domain(pfc::domain::create(pfc::GridSize({nx, ny, nz}),
                                   pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                   pfc::GridSpacing({1.0, 1.0, 1.0}))),
        stack(domain, world_rank(), world_size(), MPI_COMM_WORLD),
        phi(pfc::data::field_from_inbox<double>(domain,
                                                stack.fft().get_inbox_bounds())) {}
  template <class Fn> void fill(Fn &&fn) {
    const auto n = phi.local_size();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const auto x = phi.coords(i, j, k);
          phi(i, j, k) = fn(x[0], x[1], x[2]);
        }
    phi.note_host_write();
  }
};

MicroelasticityParams axial_params(double min_modulus) {
  MicroelasticityParams p;
  p.stiffness_at_one = Stiffness::isotropic(1.0, 0.3);
  p.stiffness_at_zero = Stiffness::isotropic(min_modulus, 0.3);
  p.applied_strain = Sym3{};
  p.applied_strain[SYM_ZZ] = 1.0;
  p.relative_tolerance = 1.0e-11;
  p.max_iterations = 400;
  p.comm = MPI_COMM_WORLD;
  return p;
}

} // namespace

TEST_CASE("SIMP weight and slope", "[simp]") {
  const SimpInterpolation linear{1.0};
  REQUIRE(linear.weight(0.25) == 0.25);
  REQUIRE(linear.slope(0.0) == 1.0);
  const SimpInterpolation cubic{3.0};
  REQUIRE_THAT(cubic.weight(0.5), WithinAbs(0.125, 1e-15));
  REQUIRE_THAT(cubic.slope(0.5), WithinAbs(0.75, 1e-15));
  REQUIRE(cubic.weight(-0.2) == 0.0);
  REQUIRE(cubic.weight(1.3) == 1.0);
  REQUIRE(cubic.slope(0.0) == 0.0);
  REQUIRE_THROWS(SimpInterpolation{0.5}.validate());
}

TEST_CASE("a uniform density stores the energy of its SIMP modulus",
          "[compliance]") {
  Cell cell(8, 8, 8);
  const double c = 0.6;
  cell.fill([&](double, double, double) { return c; });
  const double emin = 1.0e-2;
  LoadCaseCompliance<> load(cell.domain, cell.stack.fft(), axial_params(emin),
                            SimpInterpolation{3.0});
  auto grad = pfc::data::field_from_inbox<double>(
      cell.domain, cell.stack.fft().get_inbox_bounds());
  const auto r = load.evaluate(cell.phi, &grad);
  REQUIRE(r.report.converged);
  const double w = c * c * c;
  const Stiffness one = Stiffness::isotropic(1.0, 0.3);
  const Stiffness zero = Stiffness::isotropic(emin, 0.3);
  const double c33 = w * one.c11 + (1.0 - w) * zero.c11;
  REQUIRE_THAT(r.effective_modulus, WithinRel(c33, 1e-12));
  REQUIRE_THAT(r.energy_density, WithinRel(0.5 * c33, 1e-12));
  // dU/dphi = 1/2 * 3 c^2 * (C1 - C0)_3333 at every cell.
  const double expect = 0.5 * 3.0 * c * c * (one.c11 - zero.c11);
  for (double g : grad.vec()) REQUIRE_THAT(g, WithinRel(expect, 1e-10));
}

TEST_CASE("an axial load rewards a column along z over the same column across z",
          "[compliance]") {
  Cell cell(12, 12, 12);
  const double emin = 1.0e-2;
  auto column = [](double x, double y, double) {
    return ((x - 6.0) * (x - 6.0) + (y - 6.0) * (y - 6.0) < 9.0) ? 1.0 : 0.0;
  };
  cell.fill(column);
  LoadCaseCompliance<> load(cell.domain, cell.stack.fft(), axial_params(emin),
                            SimpInterpolation{3.0});
  const double u_column = load.evaluate(cell.phi).energy_density;
  // The same column along y: same volume, no load path along z.
  cell.fill([&](double x, double, double z) { return column(x, z, 0.0); });
  load.solver().reset();
  const double u_across = load.evaluate(cell.phi).energy_density;
  REQUIRE(u_column > 2.0 * u_across);
}

TEST_CASE("the load-case gradient matches a finite difference", "[compliance][fd]") {
  Cell cell(12, 10, 14);
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  cell.fill([&](double x, double y, double z) {
    return 0.5 + 0.3 * std::sin(kTwoPi * x / 12.0) * std::cos(kTwoPi * y / 10.0) +
           0.15 * std::cos(kTwoPi * z / 14.0 + 0.4);
  });
  LoadCaseCompliance<> load(cell.domain, cell.stack.fft(), axial_params(1.0e-2),
                            SimpInterpolation{3.0});
  auto grad = pfc::data::field_from_inbox<double>(
      cell.domain, cell.stack.fft().get_inbox_bounds());
  load.evaluate(cell.phi, &grad);

  // Direction: smooth and small, so the clamp of phi to [0, 1] never acts;
  // the mean part keeps it from being orthogonal to the gradient.
  auto dir = pfc::data::field_from_inbox<double>(
      cell.domain, cell.stack.fft().get_inbox_bounds());
  {
    const auto n = dir.local_size();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const auto x = dir.coords(i, j, k);
          dir(i, j, k) =
              0.2 + std::cos(kTwoPi * x[0] / 12.0) * std::cos(kTwoPi * x[1] / 10.0) +
              0.3 * std::sin(kTwoPi * (x[0] + 2.0 * x[2]) / 12.0);
        }
  }
  const double n_global = 12.0 * 10.0 * 14.0;
  double local = 0.0;
  for (std::size_t i = 0; i < grad.vec().size(); ++i)
    local += grad.vec()[i] * dir.vec()[i];
  const double predicted = gsum(local) / n_global;

  const double h = 1.0e-4;
  auto shifted = cell.phi;
  auto energy_at = [&](double s) {
    for (std::size_t i = 0; i < shifted.vec().size(); ++i)
      shifted.vec()[i] = cell.phi.vec()[i] + s * dir.vec()[i];
    shifted.note_host_write();
    return load.evaluate(shifted).energy_density;
  };
  const double fd = (energy_at(h) - energy_at(-h)) / (2.0 * h);
  REQUIRE(std::abs(predicted) > 1.0e-6);
  REQUIRE_THAT(fd, WithinRel(predicted, 1.0e-6));
}
