// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>
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
  // Generic packs: MultiExplicitRK stage temps own vectors; du remains tied refs.
  template <class UPack, class DuPack>
  void operator()(double /*t*/, UPack &u_pack, DuPack &du_pack) const {
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

TEST_CASE("registered_method_composers enumerates Euler and three RK builtins",
          "[method_composition]") {
  const auto composers = registered_method_composers();
  REQUIRE(composers.size() == 4);

  const MethodComposerEntry *euler = nullptr;
  const MethodComposerEntry *rk2_mid = nullptr;
  const MethodComposerEntry *rk2_heun = nullptr;
  const MethodComposerEntry *rk4 = nullptr;
  for (const auto &entry : composers) {
    if (entry.id == "euler") {
      euler = &entry;
    } else if (entry.id == "rk2_midpoint") {
      rk2_mid = &entry;
    } else if (entry.id == "rk2_heun") {
      rk2_heun = &entry;
    } else if (entry.id == "rk4_classical") {
      rk4 = &entry;
    }
  }
  REQUIRE(euler != nullptr);
  REQUIRE(rk2_mid != nullptr);
  REQUIRE(rk2_heun != nullptr);
  REQUIRE(rk4 != nullptr);

  REQUIRE(euler->method == RKIntegratorMethod::Euler);
  REQUIRE(rk2_mid->method == RKIntegratorMethod::RK2_Midpoint);
  REQUIRE(rk2_heun->method == RKIntegratorMethod::RK2_Heun);
  REQUIRE(rk4->method == RKIntegratorMethod::RK4_Classical);

  for (const auto *entry : {euler, rk2_mid, rk2_heun, rk4}) {
    REQUIRE(entry->supports_scalar);
    REQUIRE(entry->supports_multi);
    REQUIRE_FALSE(entry->supports_embedded_error);
  }
}

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

TEST_CASE("compose_scalar succeeds for registered RK methods with identity",
          "[method_composition]") {
  const auto method = GENERATE(RKIntegratorMethod::RK2_Midpoint,
                               RKIntegratorMethod::RK2_Heun,
                               RKIntegratorMethod::RK4_Classical);
  constexpr double dt = 0.01;
  constexpr double lambda = 1.0;
  constexpr std::size_t n = 3;
  IntegratorComposeConfig cfg{.dt = dt, .requires_adaptive = false};
  const std::string id = to_string(method);

  auto composition = compose_scalar(id, cfg, n, make_decay_rhs(lambda));

  REQUIRE(composition.method == method);
  REQUIRE(composition.workspace_ownership == WorkspaceOwnership::MethodOwned);
  REQUIRE_FALSE(composition.method_state.checkpointable);

  std::vector<double> u{1.0, 2.0, 3.0};
  const auto u_before = u;
  const double t = composition.stepper.step(0.0, u);
  REQUIRE_THAT(t, WithinAbs(dt, 1e-15));
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(u[i] != u_before[i]);
  }
}

TEST_CASE("compose_scalar RK step matches ExplicitRKStepper + make_tableau",
          "[method_composition]") {
  const auto method = GENERATE(RKIntegratorMethod::RK2_Midpoint,
                               RKIntegratorMethod::RK2_Heun,
                               RKIntegratorMethod::RK4_Classical);
  constexpr double dt = 0.01;
  constexpr double lambda = 1.0;
  constexpr std::size_t n = 3;
  IntegratorComposeConfig cfg{.dt = dt, .requires_adaptive = false};
  const std::vector<double> u0{1.0, 2.0, 3.0};

  auto composed =
      compose_scalar(to_string(method), cfg, n, make_decay_rhs(lambda));
  ExplicitRKStepper reference(dt, n, make_tableau(method),
                              make_decay_rhs(lambda));

  std::vector<double> u_comp = u0;
  std::vector<double> u_ref = u0;
  (void)composed.stepper.step(0.0, u_comp);
  (void)reference.step(0.0, u_ref);

  REQUIRE(composed.method == method);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE_THAT(u_comp[i], WithinAbs(u_ref[i], 1e-14));
  }
}

TEST_CASE("compose_multi succeeds for registered RK methods",
          "[method_composition]") {
  const auto method = GENERATE(RKIntegratorMethod::RK2_Midpoint,
                               RKIntegratorMethod::RK2_Heun,
                               RKIntegratorMethod::RK4_Classical);
  constexpr double dt = 0.001;
  IntegratorComposeConfig cfg{.dt = dt, .requires_adaptive = false};
  const std::array<std::size_t, 2> sizes{1, 1};
  const std::string id = to_string(method);

  auto composition =
      compose_multi<RotationRhs, 2>(id, cfg, sizes, RotationRhs{});
  REQUIRE(composition.method == method);
  REQUIRE(composition.workspace_ownership == WorkspaceOwnership::MethodOwned);

  MultiExplicitRKStepper<RotationRhs, 2> reference(
      dt, sizes, make_tableau(method), RotationRhs{});

  std::vector<double> u_comp{1.0};
  std::vector<double> v_comp{0.25};
  std::vector<double> u_ref{1.0};
  std::vector<double> v_ref{0.25};
  const double t_comp = composition.stepper.step(0.0, u_comp, v_comp);
  const double t_ref = reference.step(0.0, u_ref, v_ref);

  REQUIRE_THAT(t_comp, WithinAbs(t_ref, 1e-15));
  REQUIRE_THAT(u_comp[0], WithinAbs(u_ref[0], 1e-14));
  REQUIRE_THAT(v_comp[0], WithinAbs(v_ref[0], 1e-14));
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

TEST_CASE("compose_scalar rejects adaptive requirement for fixed-step methods",
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

  REQUIRE_THROWS_AS(
      compose_scalar("rk4_classical", cfg, 2, make_decay_rhs(1.0)),
      ComposeError);
}

TEST_CASE("compose_scalar rejects known id without registered composer",
          "[method_composition]") {
  REQUIRE(resolve_method_id("bogacki_shampine32").has_value());
  IntegratorComposeConfig cfg{.dt = 0.01, .requires_adaptive = false};
  try {
    (void)compose_scalar("bogacki_shampine32", cfg, 2, make_decay_rhs(1.0));
    FAIL("expected ComposeError");
  } catch (const ComposeError &e) {
    REQUIRE(e.kind() == ComposeError::Kind::CapabilityMismatch);
    REQUIRE_THAT(std::string(e.what()),
                 ContainsSubstring("no composer registered"));
    REQUIRE_THAT(std::string(e.what()), ContainsSubstring("euler"));
  }
}

TEST_CASE("compose_scalar accepts RKIntegratorMethod overload for builtins",
          "[method_composition]") {
  IntegratorComposeConfig cfg{.dt = 0.01, .requires_adaptive = false};
  auto euler =
      compose_scalar(RKIntegratorMethod::Euler, cfg, 2, make_decay_rhs(1.0));
  REQUIRE(euler.method == RKIntegratorMethod::Euler);
  REQUIRE(euler.workspace_ownership == WorkspaceOwnership::MethodOwned);

  auto rk4 = compose_scalar(RKIntegratorMethod::RK4_Classical, cfg, 2,
                            make_decay_rhs(1.0));
  REQUIRE(rk4.method == RKIntegratorMethod::RK4_Classical);
}
