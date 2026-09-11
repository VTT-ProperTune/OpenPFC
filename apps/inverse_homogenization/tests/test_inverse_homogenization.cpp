// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_inverse_homogenization.cpp
 * @brief Allen–Cahn inverse step: Laplacian oracle, J descent, homogeneous recover.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>
#include <openpfc_apps/homogenization.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::MicroelasticityParams;
using pfc::apps::PeriodicHomogenizer;
using pfc::apps::Stiffness;
using pfc::apps::voigt_from_stiffness;
using pfc::apps::Voigt6;
using pfc::apps::inverse::double_well_prime;
using pfc::apps::inverse::InverseSpec;
using pfc::apps::inverse::PhaseFieldInverse;
using pfc::apps::inverse::spectral_laplacian;
using RealField = pfc::data::Field<double>;
using ComplexField = pfc::data::Field<std::complex<double>>;

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

pfc::Domain cube(int n) {
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

void fill_value(RealField &f, double v) {
  std::fill(f.vec().begin(), f.vec().end(), v);
  f.note_host_write();
}

struct Case {
  pfc::Domain domain;
  pfc::sim::stacks::SpectralCPUStack stack;
  RealField h;

  explicit Case(int n)
      : domain(cube(n)),
        stack(domain, world_rank(), world_size(), MPI_COMM_WORLD),
        h(pfc::data::field_from_inbox<double>(domain,
                                              stack.fft().get_inbox_bounds())) {}
};

MicroelasticityParams phases() {
  MicroelasticityParams p;
  p.c_solid = Stiffness::isotropic(2.0, 0.25);
  p.c_liquid = Stiffness::isotropic(0.5, 0.25);
  p.tol_el = 1.0e-8;
  p.n_el_iter = 40;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;
  return p;
}

} // namespace

TEST_CASE("Spectral Laplacian of a cosine is -k^2 times the cosine",
          "[inverse][laplacian]") {
  constexpr int N = 16;
  Case cs(N);
  const double kx = 2.0 * std::numbers::pi / static_cast<double>(N);
  const auto n = cs.h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto x = cs.h.coords(i, j, k);
        cs.h(i, j, k) = std::cos(kx * x[0]);
      }
  cs.h.note_host_write();

  ComplexField hat(cs.domain, cs.stack.fft().get_outbox_bounds(), 0);
  RealField lap = pfc::data::field_from_inbox<double>(
      cs.domain, cs.stack.fft().get_inbox_bounds());
  spectral_laplacian(cs.domain, cs.stack.fft(), cs.h, hat, lap);

  double worst = 0.0;
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto x = cs.h.coords(i, j, k);
        const double want = -kx * kx * std::cos(kx * x[0]);
        worst = std::max(worst, std::abs(lap(i, j, k) - want));
      }
  double gworst = 0.0;
  MPI_Allreduce(&worst, &gworst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(gworst < 1.0e-10);
}

TEST_CASE("Uniform inverse step drives volume toward the homogeneous target",
          "[inverse][homogeneous]") {
  constexpr int N = 8;
  Case cs(N);
  fill_value(cs.h, 0.80);

  const Stiffness cs_ = Stiffness::isotropic(2.0, 0.25);
  const Stiffness cl_ = Stiffness::isotropic(0.5, 0.25);
  InverseSpec spec;
  spec.C_target = voigt_from_stiffness(Stiffness::blend(cs_, 0.5, cl_, 0.5));
  spec.volume_target = 0.5;
  spec.lambda_volume = 1.0;
  spec.lambda_reg = 0.0; // double well would fight a grey homogeneous target
  spec.dt = 0.05;
  spec.mobility = 1.0;
  spec.clip = true;
  spec.normalize_grad = true;
  spec.max_abs_delta = 0.05;

  PhaseFieldInverse inv(cs.domain, cs.stack.fft(), phases());
  const auto first = inv.step(cs.h, spec);
  REQUIRE(first.elasticity_converged);
  // RMS-normalised step: volume must not jump by more than dt.
  REQUIRE(std::abs(first.volume_fraction - 0.80) <= spec.dt + 1.0e-6);
  REQUIRE(first.step_rms <= spec.max_abs_delta + 1.0e-12);

  double J0 = first.J;
  double vf0 = first.volume_fraction;
  pfc::apps::inverse::InverseStepReport last = first;
  for (int s = 0; s < 4; ++s) last = inv.step(cs.h, spec);
  REQUIRE(last.J < J0);
  REQUIRE(last.volume_fraction < vf0);
  REQUIRE(last.volume_fraction > 0.5);
}

TEST_CASE("Noisy initialization: a few AC steps decrease J",
          "[inverse][descent]") {
  constexpr int N = 8;
  Case cs(N);
  const auto n = cs.h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto g = cs.h.global(i, j, k);
        const double noise =
            0.05 * std::sin(2.0 * std::numbers::pi * (g[0] + 2.0 * g[1] + 3.0 * g[2]) /
                            N);
        cs.h(i, j, k) = 0.55 + noise;
      }
  cs.h.note_host_write();

  InverseSpec spec;
  spec.C_target = voigt_from_stiffness(Stiffness::isotropic(1.0, 0.25));
  spec.volume_target = 0.5;
  spec.lambda_volume = 0.5;
  spec.lambda_reg = 0.02;
  spec.epsilon = 2.0;
  spec.dt = 0.05;

  PhaseFieldInverse inv(cs.domain, cs.stack.fft(), phases());
  const auto r0 = inv.step(cs.h, spec);
  REQUIRE(r0.elasticity_converged);
  auto r = r0;
  for (int s = 0; s < 4; ++s) r = inv.step(cs.h, spec);
  REQUIRE(r.J < r0.J);
}

TEST_CASE("Double-well derivative vanishes at the wells and at 1/2",
          "[inverse][algebra]") {
  REQUIRE_THAT(double_well_prime(0.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(double_well_prime(1.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(double_well_prime(0.5), WithinAbs(0.0, 1e-15));
  REQUIRE(double_well_prime(0.25) > 0.0);
  REQUIRE(double_well_prime(0.75) < 0.0);
}
