// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/moving_frame_mean_field_etd.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

using pfc::Box3i;
using pfc::Domain;
using pfc::SimulationState;
using pfc::sim::add_declared_field;
using pfc::sim::MovingFrameMeanFieldETDPhysics;
using pfc::sim::MovingFrameMeanFieldETDSystem;

namespace {

struct ToyMovingFrame {
  using parameters_type = double;
  Domain domain{};
  Box3i box{};

  void declare_fields(SimulationState &state) const {
    add_declared_field<double>(state, "psi", domain, box, 0);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    return k_laplacian;
  }
  [[nodiscard]] double filter_mf(double) const { return 1.0; }
  [[nodiscard]] double correlation_kernel(double) const { return 0.0; }
  [[nodiscard]] double temperature_variation(double, double) const { return 0.0; }
  [[nodiscard]] double nonlinearity(double psi, double, double, double) const {
    return -psi * psi * psi;
  }
  [[nodiscard]] double free_energy_density(double psi, double, double,
                                           double) const {
    return 0.5 * psi * psi;
  }
};

} // namespace

static_assert(MovingFrameMeanFieldETDPhysics<ToyMovingFrame>);

TEST_CASE("MovingFrameMeanFieldETDSystem zero field stays zero",
          "[moving_frame_etd][unit]") {
  constexpr int N = 8;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);

  ToyMovingFrame physics{};
  physics.domain = domain;
  physics.box = fft.get_inbox_bounds();

  SimulationState state;
  physics.declare_fields(state);
  MovingFrameMeanFieldETDSystem<ToyMovingFrame> sys(physics, fft, state, 0.01);

  REQUIRE(sys.linear_symbol().size() == fft.size_outbox());
  REQUIRE(sys.filter_mf().size() == fft.size_outbox());
  REQUIRE(sys.correlation_kernel().size() == fft.size_outbox());

  auto &psi = state.get_field<double>("psi").vec();
  for (double &v : psi) {
    v = 0.0;
  }
  const double t1 = sys.step(0.0);
  REQUIRE(t1 == Catch::Approx(0.01));
  for (double v : psi) {
    REQUIRE(v == Catch::Approx(0.0).margin(1e-14));
  }
  REQUIRE(sys.last_free_energy_sum() == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(sys.last_free_energy() == Catch::Approx(0.0).margin(1e-14));
}
