// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/kernel/simulation/steppers/method_composition.hpp>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

using namespace pfc::sim::steppers;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

auto make_decay_rhs(double lambda) {
  return [lambda](double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -lambda * u[i];
    }
  };
}

struct RotationRhs {
  void operator()(
      double /*t*/, std::tuple<std::vector<double> &, std::vector<double> &> u_pack,
      std::tuple<std::vector<double> &, std::vector<double> &> du_pack) const {
    const auto &u = std::get<0>(u_pack);
    const auto &v = std::get<1>(u_pack);
    auto &du = std::get<0>(du_pack);
    auto &dv = std::get<1>(du_pack);
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = v[i];
      dv[i] = -u[i];
    }
  }
};

} // namespace

TEST_CASE("compose_scalar succeeds for Euler with MethodOwned workspace",
          "[method_composition]") {
  constexpr double dt = 0.01;
  constexpr double lambda = 1.0;
  constexpr std::size_t n = 3;
  IntegratorComposeConfig cfg{.dt = dt, .requires_adaptive = false};

  auto composition =
      compose_scalar("euler", cfg, n, make_decay_rhs(lambda));

  REQUIRE(composition.method == RKIntegratorMethod::Euler);
  REQUIRE(composition.workspace_ownership == WorkspaceOwnership::MethodOwned);
  REQUIRE_FALSE(composition.method_state.checkpointable);
  REQUIRE(composition.method_state.state_keys.empty());

  std::vector<double> u{1.0, 2.0, 3.0};
  const auto u_before = u;
  double t = 0.0;
  t = composition.stepper.step(t, u);

  REQUIRE_THAT(t, WithinAbs(dt, 1e-15));
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(u[i] != u_before[i]);
    REQUIRE_THAT(u[i], WithinAbs(u_before[i] + dt * (-lambda * u_before[i]), 1e-12));
  }
}

TEST_CASE("compose_multi succeeds for two-field Euler rotation",
          "[method_composition]") {
  constexpr double dt = 0.001;
  constexpr double u0 = 1.0;
  constexpr double v0 = 0.25;
  IntegratorComposeConfig cfg{.dt = dt, .requires_adaptive = false};
  const std::array<std::size_t, 2> sizes{1, 1};

  auto composition = compose_multi<RotationRhs, 2>("euler", cfg, sizes,
                                                   RotationRhs{});

  REQUIRE(composition.method == RKIntegratorMethod::Euler);
  REQUIRE(composition.workspace_ownership == WorkspaceOwnership::MethodOwned);
  REQUIRE_FALSE(composition.method_state.checkpointable);
  REQUIRE(composition.method_state.state_keys.empty());

  std::vector<double> u{u0};
  std::vector<double> v{v0};
  const double t = composition.stepper.step(0.0, u, v);

  REQUIRE_THAT(t, WithinAbs(dt, 1e-15));
  REQUIRE_THAT(u[0], WithinAbs(u0 + dt * v0, 1e-12));
  REQUIRE_THAT(v[0], WithinAbs(v0 - dt * u0, 1e-12));
}

TEST_CASE("compose_scalar rejects unknown method identifier",
          "[method_composition]") {
  IntegratorComposeConfig cfg{.dt = 0.01, .requires_adaptive = false};
  try {
    (void)compose_scalar("not_a_method", cfg, 2, make_decay_rhs(1.0));
    FAIL("expected ComposeError");
  } catch (const ComposeError &e) {
    REQUIRE(e.kind() == ComposeError::Kind::UnknownIdentifier);
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("not_a_method"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("euler"));
  }
}

TEST_CASE("compose_scalar rejects non-positive dt", "[method_composition]") {
  IntegratorComposeConfig cfg{.dt = 0.0, .requires_adaptive = false};
  try {
    (void)compose_scalar("euler", cfg, 2, make_decay_rhs(1.0));
    FAIL("expected ComposeError");
  } catch (const ComposeError &e) {
    REQUIRE(e.kind() == ComposeError::Kind::InvalidConfiguration);
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("dt must be > 0"));
  }

  cfg.dt = -0.5;
  REQUIRE_THROWS_AS(compose_scalar("euler", cfg, 2, make_decay_rhs(1.0)),
                    ComposeError);
}

TEST_CASE("compose_scalar rejects adaptive requirement for Euler",
          "[method_composition]") {
  IntegratorComposeConfig cfg{.dt = 0.01, .requires_adaptive = true};
  try {
    (void)compose_scalar("euler", cfg, 2, make_decay_rhs(1.0));
    FAIL("expected ComposeError");
  } catch (const ComposeError &e) {
    REQUIRE(e.kind() == ComposeError::Kind::CapabilityMismatch);
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("embedded"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("euler"));
  }
}

TEST_CASE("compose_scalar rejects known id without registered composer",
          "[method_composition]") {
  REQUIRE(resolve_method_id("rk4_classical").has_value());
  IntegratorComposeConfig cfg{.dt = 0.01, .requires_adaptive = false};
  try {
    (void)compose_scalar("rk4_classical", cfg, 2, make_decay_rhs(1.0));
    FAIL("expected ComposeError");
  } catch (const ComposeError &e) {
    REQUIRE(e.kind() == ComposeError::Kind::CapabilityMismatch);
    REQUIRE_THAT(std::string(e.what()),
                 ContainsSubstring("no composer registered"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("euler"));
  }
}

TEST_CASE("compose_scalar accepts RKIntegratorMethod::Euler overload",
          "[method_composition]") {
  IntegratorComposeConfig cfg{.dt = 0.01, .requires_adaptive = false};
  auto composition =
      compose_scalar(RKIntegratorMethod::Euler, cfg, 2, make_decay_rhs(1.0));
  REQUIRE(composition.method == RKIntegratorMethod::Euler);
  REQUIRE(composition.workspace_ownership == WorkspaceOwnership::MethodOwned);
}
