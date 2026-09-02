// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <complex>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <fixtures/simulation_factories.hpp>
#include <fixtures/swift_hohenberg.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

using Catch::Approx;
using Complex = std::complex<double>;
using pfc::Box3i;
using pfc::Domain;
using pfc::SimulationState;
using pfc::data::Field;
using pfc::test::SHGrads;
using pfc::test::SwiftHohenberg;

namespace {

struct LapGrads {
  double value{};
  double xx{};
  double yy{};
  double zz{};
};

void fill_sh_ic(Field<double> &psi) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return 0.05 + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

} // namespace

TEST_CASE("Swift-Hohenberg: point-wise Euler vs spectral ETD",
          "[swift_hohenberg][unit]") {
  constexpr int N = 8;
  constexpr double dt = 1e-5;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  const auto inbox = fft.get_inbox_bounds();

  SwiftHohenberg phys{};
  phys.domain = domain;
  phys.box = inbox;
  phys.params.epsilon = 0.25;

  SimulationState etd_state;
  phys.declare_fields(etd_state);
  fill_sh_ic(etd_state.get_field<double>("psi"));
  pfc::sim::SpectralETDSystem<SwiftHohenberg> etd(phys, fft, etd_state, dt);
  etd.step(0.0);
  const auto etd_psi = etd_state.get_field<double>("psi").vec();

  // --- Point-wise rhs + SpectralGradient (lap then ∇⁴) + Euler ---
  auto fft_pw = pfc::fft::create(decomp);
  Field<double> u(domain, inbox, 0);
  Field<double> lap(domain, inbox, 0);
  fill_sh_ic(u);
  auto grad_u = pfc::field::create<LapGrads>(u, fft_pw);
  auto grad_lap = pfc::field::create<LapGrads>(lap, fft_pw);
  auto pw_rhs = [&](double t, std::vector<double> & /*u_buf*/,
                    std::vector<double> &du) {
    grad_u.prepare();
    const int nx = grad_u.imax();
    const int ny = grad_u.jmax();
    const int nz = grad_u.kmax();
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const auto g = grad_u(i, j, k);
          lap.vec()[grad_u.idx(i, j, k)] = g.xx + g.yy + g.zz;
        }
      }
    }
    grad_lap.prepare();
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const auto g = grad_u(i, j, k);
          const auto gl = grad_lap(i, j, k);
          const SHGrads sh{.value = g.value,
                           .lap = g.xx + g.yy + g.zz,
                           .biharm = gl.xx + gl.yy + gl.zz};
          du[grad_u.idx(i, j, k)] = phys.rhs(t, sh);
        }
      }
    }
  };
  pfc::sim::steppers::EulerStepper<decltype(pw_rhs)> euler(dt, u.size(),
                                                           pw_rhs);
  euler.step(0.0, u.vec());
  REQUIRE(max_abs_diff(u.vec(), etd_psi) < 1e-5);
}


