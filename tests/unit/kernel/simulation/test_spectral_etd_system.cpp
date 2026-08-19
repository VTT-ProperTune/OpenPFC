// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <complex>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>

using pfc::Box3i;
using pfc::Domain;
using pfc::SimulationState;
using pfc::sim::SpectralEtdPhysics;
using pfc::sim::SpectralEtdSystem;
using pfc::sim::add_declared_field;

namespace {

struct SHParams {
  double epsilon{0.25};
};

struct SwiftHohenberg {
  using parameters_type = SHParams;
  Domain domain{};
  Box3i box{};
  SHParams params{};

  void declare_fields(SimulationState &state) const {
    add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const double one_plus_lap = 1.0 + k_laplacian;
    return params.epsilon - one_plus_lap * one_plus_lap;
  }

  [[nodiscard]] double nonlinearity(double psi) const {
    return -psi * psi * psi;
  }
};

} // namespace

static_assert(SpectralEtdPhysics<SwiftHohenberg>);

TEST_CASE("SpectralEtdSystem zero field stays zero",
          "[spectral_etd][unit]") {
  constexpr int N = 8;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  SwiftHohenberg physics{};
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();

  SimulationState state;
  physics.declare_fields(state);
  SpectralEtdSystem<SwiftHohenberg> sys(physics, fft, state, 0.01);

  REQUIRE(sys.linear_symbol().size() == fft.size_outbox());
  REQUIRE(sys.linear_symbol()[0] ==
          Catch::Approx(physics.linear_symbol(0.0)));

  auto &psi = state.get_field<double>("psi").vec();
  for (double &v : psi) {
    v = 0.0;
  }
  const double t1 = sys.step(0.0);
  REQUIRE(t1 == Catch::Approx(0.01));
  for (double v : psi) {
    REQUIRE(v == Catch::Approx(0.0).margin(1e-14));
  }
}

TEST_CASE("SpectralEtdSystem uniform field matches zero-mode ETD1",
          "[spectral_etd][unit]") {
  constexpr int N = 8;
  constexpr double dt = 0.02;
  constexpr double c = 0.5;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  SwiftHohenberg physics{};
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();
  physics.params.epsilon = 0.25;

  SimulationState state;
  physics.declare_fields(state);
  SpectralEtdSystem<SwiftHohenberg> sys(physics, fft, state, dt);

  auto &psi = state.get_field<double>("psi").vec();
  for (double &v : psi) {
    v = c;
  }
  sys.step(0.0);

  const double L0 = physics.linear_symbol(0.0);
  const double N0 = physics.nonlinearity(c);
  const auto coeff = pfc::integrator::spectral_exp_coeffs(L0, dt);
  const double expected = coeff.exp_Ldt * c + coeff.phi1_L * N0;
  for (double v : psi) {
    REQUIRE(v == Catch::Approx(expected).margin(1e-10));
  }
}

TEST_CASE("SpectralEtdSystem optional dealias mask does not throw",
          "[spectral_etd][unit][dealias]") {
  constexpr int N = 8;
  auto domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  SwiftHohenberg physics{};
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();

  SimulationState state;
  physics.declare_fields(state);
  pfc::sim::SpectralEtdOptions opt{};
  opt.dealias = true;
  SpectralEtdSystem<SwiftHohenberg> sys(std::move(physics), fft, state, 0.01,
                                        opt);
  auto &psi = state.get_field<double>("psi").vec();
  psi[0] = 0.1;
  REQUIRE_NOTHROW(sys.step(0.0));
}
