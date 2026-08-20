// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

using pfc::Box3i;
using pfc::Domain;
using pfc::SimulationState;
using pfc::sim::add_declared_field;
using pfc::sim::DeclaresFields;
using pfc::sim::FieldDeclaration;
using pfc::sim::FieldElementKind;
using pfc::sim::HasParameters;
using pfc::sim::MovingFrameMeanFieldETDPhysics;
using pfc::sim::PointwisePhysics;
using pfc::sim::PointwiseRhs;
using pfc::sim::SpectralDiagonalPhysics;
using pfc::sim::SpectralETDPhysics;

namespace {

Box3i unit_box() { return Box3i::from_bounds({0, 0, 0}, {0, 0, 0}); }

struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

struct HeatPhysics {
  using parameters_type = double;

  Domain domain = pfc::domain::create({1, 1, 1});
  Box3i box = unit_box();
  double D{1.0};

  void declare_fields(SimulationState &state) const {
    add_declared_field<double>(state, "u", domain, box, 0);
  }

  [[nodiscard]] double rhs(double /*t*/, const HeatGrads &g) const {
    return D * (g.xx + g.yy + g.zz);
  }
};

struct SwiftHohenbergParams {
  double epsilon{0.3};
};

struct SwiftHohenbergPhysics {
  using parameters_type = SwiftHohenbergParams;

  Domain domain = pfc::domain::create({1, 1, 1});
  Box3i box = unit_box();
  SwiftHohenbergParams params{};

  void declare_fields(SimulationState &state) const {
    const FieldDeclaration psi{
        .name = "psi", .element = FieldElementKind::Real, .halo = 0};
    add_declared_field(state, psi, domain, box);
  }

  [[nodiscard]] double linear_symbol(double k_laplacian) const {
    const double one_plus_lap = 1.0 + k_laplacian;
    return params.epsilon - one_plus_lap * one_plus_lap;
  }

  [[nodiscard]] double nonlinearity(double psi) const { return -psi * psi * psi; }
};

struct MovingFrameToy {
  using parameters_type = double;
  Domain domain = pfc::domain::create({1, 1, 1});
  Box3i box = unit_box();

  void declare_fields(SimulationState &state) const {
    add_declared_field<double>(state, "psi", domain, box, 0);
  }
  double linear_symbol(double k_laplacian) const { return k_laplacian; }
  double filter_mf(double) const { return 1.0; }
  double correlation_kernel(double) const { return 0.0; }
  double temperature_variation(double, double) const { return 0.0; }
  double nonlinearity(double psi, double, double, double) const {
    return -psi * psi * psi;
  }
  double free_energy_density(double psi, double, double, double) const {
    return 0.5 * psi * psi;
  }
};

struct LinearOnly {
  double linear_symbol(double k_laplacian) const { return k_laplacian; }
};

struct RhsOnly {
  double rhs(double /*t*/, const HeatGrads &g) const { return g.xx; }
};

} // namespace

static_assert(DeclaresFields<HeatPhysics>);
static_assert(PointwiseRhs<HeatPhysics, HeatGrads>);
static_assert(PointwisePhysics<HeatPhysics, HeatGrads>);
static_assert(HasParameters<HeatPhysics>);
static_assert(!SpectralDiagonalPhysics<HeatPhysics>);
static_assert(!SpectralETDPhysics<HeatPhysics>);

static_assert(DeclaresFields<SwiftHohenbergPhysics>);
static_assert(SpectralDiagonalPhysics<SwiftHohenbergPhysics>);
static_assert(SpectralETDPhysics<SwiftHohenbergPhysics>);
static_assert(HasParameters<SwiftHohenbergPhysics>);
static_assert(!PointwiseRhs<SwiftHohenbergPhysics, HeatGrads>);

static_assert(MovingFrameMeanFieldETDPhysics<MovingFrameToy>);
static_assert(!MovingFrameMeanFieldETDPhysics<SwiftHohenbergPhysics>);

static_assert(!DeclaresFields<LinearOnly>);
static_assert(pfc::sim::SpectralLinearSymbol<LinearOnly>);
static_assert(!SpectralDiagonalPhysics<LinearOnly>);
static_assert(!SpectralETDPhysics<LinearOnly>);

static_assert(PointwiseRhs<RhsOnly, HeatGrads>);
static_assert(!DeclaresFields<RhsOnly>);
static_assert(!PointwisePhysics<RhsOnly, HeatGrads>);

TEST_CASE("declare_fields allocates named host fields", "[physics_concepts][unit]") {
  HeatPhysics heat{};
  SimulationState state;
  heat.declare_fields(state);
  REQUIRE(state.has_field("u"));
  REQUIRE(state.num_fields() == 1);
  REQUIRE(state.get_field<double>("u").size() == 1);

  SwiftHohenbergPhysics sh{};
  SimulationState sh_state;
  sh.declare_fields(sh_state);
  REQUIRE(sh_state.has_field("psi"));
  REQUIRE(sh_state.get_field<double>("psi").size() == 1);
}

TEST_CASE("add_declared_field allocates a complex hat field",
          "[physics_concepts][unit]") {
  SimulationState state;
  const auto domain = pfc::domain::create({2, 1, 1});
  const auto box = Box3i::from_bounds({0, 0, 0}, {1, 0, 0});
  const FieldDeclaration hat{
      .name = "psi_hat", .element = FieldElementKind::Complex, .halo = 0};
  add_declared_field(state, hat, domain, box);
  REQUIRE(state.has_field("psi_hat"));
  REQUIRE(state.get_field<std::complex<double>>("psi_hat").size() == 2);
}

TEST_CASE("Swift-Hohenberg linear symbol and cubic nonlinearity",
          "[physics_concepts][unit]") {
  SwiftHohenbergPhysics sh{};
  sh.params.epsilon = 0.25;
  // k_laplacian = 0 → L = ε - 1
  REQUIRE(sh.linear_symbol(0.0) == Catch::Approx(0.25 - 1.0));
  // k_laplacian = -1 (the SH peak |k|=1) → L = ε
  REQUIRE(sh.linear_symbol(-1.0) == Catch::Approx(0.25));
  REQUIRE(sh.nonlinearity(2.0) == Catch::Approx(-8.0));
}

TEST_CASE("point-wise heat rhs is Laplacian times D", "[physics_concepts][unit]") {
  HeatPhysics heat{};
  heat.D = 2.0;
  const HeatGrads g{.xx = 1.0, .yy = -0.5, .zz = 0.25};
  REQUIRE(heat.rhs(0.0, g) == Catch::Approx(2.0 * (1.0 - 0.5 + 0.25)));
}
