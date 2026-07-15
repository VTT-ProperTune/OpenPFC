// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/local_field.hpp>

using namespace pfc::sim::steppers;
using Catch::Approx;

TEST_CASE("EulerStepper accumulates correctly with constant RHS", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  std::vector<double> u(n, 0.0);

  // Mock RHS: fills du[i] = 2.0 for all i
  auto rhs = [](double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) {
    for (std::size_t i = 0; i < du.size(); ++i) {
      du[i] = 2.0;
    }
  };

  EulerStepper stepper(dt, n, rhs);

  SECTION("First step: u[i] should be dt * du[i] = 0.1 * 2.0 = 0.2") {
    stepper.step(0.0, u);
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u[i] == Approx(dt * 2.0)); // 0.2
    }
  }

  SECTION("Second step: u[i] should accumulate to 2 * dt * du[i] = 0.4") {
    stepper.step(0.0, u); // t=0.0, u becomes [0.2, 0.2, ...]
    stepper.step(dt, u);  // t=0.1, u becomes [0.4, 0.4, ...]
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u[i] == Approx(2.0 * dt * 2.0)); // 0.4
    }
  }
}

TEST_CASE("EulerStepper returns correct new time", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 5;
  std::vector<double> u(n, 0.0);

  // RHS that does nothing (focus on time return)
  auto dummy_rhs = [](double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 0.0);
  };

  EulerStepper stepper(dt, n, dummy_rhs);

  double t = 0.0;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.0 -> 0.1

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.1 -> 0.2

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.2 -> 0.3

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.3 -> 0.4
}

TEST_CASE("EulerStepper reuses du buffer across steps", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  std::vector<double> u(n, 0.0);

  // Functor-based RHS that only writes to one index per step
  struct SelectiveRhs {
    std::size_t step_counter = 0;
    void operator()(double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) {
      // Write to index 0 on even steps, index 1 on odd steps
      std::size_t write_idx = (step_counter % 2 == 0) ? 0 : 1;
      for (std::size_t i = 0; i < du.size(); ++i) {
        du[i] = (i == write_idx) ? 1.0 : 0.0; // Only one index gets 1.0
      }
      ++step_counter;
    }
  };

  SelectiveRhs rhs;
  EulerStepper stepper(dt, n, rhs);

  SECTION("Step 0: only du[0] written, others preserve 0.0 value-init") {
    stepper.step(0.0, u);
    REQUIRE(u[0] == Approx(dt * 1.0)); // Only index 0 advanced
    for (std::size_t i = 1; i < n; ++i) {
      REQUIRE(u[i] == Approx(0.0)); // Others stay at 0.0
    }
  }

  SECTION("Step 1: only du[1] written, du[0] from previous step ignored in accumulation") {
    stepper.step(0.0, u); // Step 0: writes du[0]=1.0, others 0.0
    stepper.step(dt, u);  // Step 1: writes du[1]=1.0, others 0.0
    REQUIRE(u[0] == Approx(dt * 1.0)); // Only advanced once
    REQUIRE(u[1] == Approx(dt * 1.0)); // Advanced on step 1
    for (std::size_t i = 2; i < n; ++i) {
      REQUIRE(u[i] == Approx(0.0)); // Never advanced
    }
  }
}

TEST_CASE("MultiEulerStepper updates fields independently", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  constexpr std::size_t N = 2;
  std::vector<double> u1(n, 0.0); // First field
  std::vector<double> u2(n, 0.0); // Second field

  // RHS that fills du1[i]=1.0, du2[i]=2.0 for all i
  auto rhs = [](double /*t*/, auto& /*u_pack*/, auto& du_pack) {
    auto& du1 = std::get<0>(du_pack);
    auto& du2 = std::get<1>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
    }
  };

  std::array<std::size_t, N> local_sizes = {n, n};
  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);

  SECTION("First step: each field accumulates independently") {
    stepper.step(0.0, u1, u2);
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u1[i] == Approx(dt * 1.0)); // 0.1
      REQUIRE(u2[i] == Approx(dt * 2.0)); // 0.2
    }
  }

  SECTION("Second step: both fields accumulate independently again") {
    stepper.step(0.0, u1, u2); // t=0.0
    stepper.step(dt, u1, u2);  // t=0.1
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u1[i] == Approx(2.0 * dt * 1.0)); // 0.2
      REQUIRE(u2[i] == Approx(2.0 * dt * 2.0)); // 0.4
    }
  }
}

#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>

TEST_CASE("ExplicitRKStepper accumulates correctly with constant RHS (RK4)", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  std::vector<double> u(n, 0.0);

  auto rhs = [](double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) {
    for (std::size_t i = 0; i < du.size(); ++i) {
      du[i] = 2.0;
    }
  };

  auto tableau = make_rk4_classical<double>();
  ExplicitRKStepper stepper(dt, n, tableau, rhs);

  // Verify RK4: u += dt * sum(b_i * k_i) where k_i = 2.0 for all stages
  // sum(b_i) = 1/6 + 1/3 + 1/3 + 1/6 = 1.0 for classical RK4
  stepper.step(0.0, u);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u[i] == Approx(dt * 2.0)); // 0.2
  }

  stepper.step(dt, u);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u[i] == Approx(2.0 * dt * 2.0)); // 0.4
  }
}

TEST_CASE("ExplicitRKStepper returns correct new time (RK2 midpoint)", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 5;
  std::vector<double> u(n, 0.0);

  auto dummy_rhs = [](double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 0.0);
  };

  auto tableau = make_rk2_midpoint<double>();
  ExplicitRKStepper stepper(dt, n, tableau, dummy_rhs);

  double t = 0.0;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // 0.0 -> 0.1

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // 0.1 -> 0.2

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // 0.2 -> 0.3
}

TEST_CASE("MultiExplicitRKStepper updates fields independently (RK4)", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  constexpr std::size_t N = 2;
  std::vector<double> u1(n, 0.0);
  std::vector<double> u2(n, 0.0);

  auto rhs = [](double /*t*/, auto& /*u_pack*/, auto& du_pack) {
    auto& du1 = std::get<0>(du_pack);
    auto& du2 = std::get<1>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
    }
  };

  std::array<std::size_t, N> local_sizes = {n, n};
  auto tableau = make_rk4_classical<double>();
  MultiExplicitRKStepper<decltype(rhs), N> stepper(dt, local_sizes, tableau, rhs);

  stepper.step(0.0, u1, u2);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u1[i] == Approx(dt * 1.0)); // 0.1
    REQUIRE(u2[i] == Approx(dt * 2.0)); // 0.2
  }

  stepper.step(dt, u1, u2);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u1[i] == Approx(2.0 * dt * 1.0)); // 0.2
    REQUIRE(u2[i] == Approx(2.0 * dt * 2.0)); // 0.4
  }
}

TEST_CASE("ExplicitRKStepper factory with LocalField", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;

  // Mock model with rhs(double t, const G&) -> double
  struct MockModel {
    double rhs(double /*t*/, double /*g*/) const { return 3.0; }
  };

  // Mock evaluator with required interface
  struct MockEval {
    std::size_t size() const { return n; }
    void prepare() {}
    int imin() const { return 0; }
    int imax() const { return static_cast<int>(n); }
    int jmin() const { return 0; }
    int jmax() const { return 1; }
    int kmin() const { return 0; }
    int kmax() const { return 1; }
    std::size_t idx(int ix, int /*iy*/, int /*iz*/) const { return static_cast<std::size_t>(ix); }
    double operator()(int /*ix*/, int /*iy*/, int /*iz*/) const { return 0.0; }
  };

  MockModel model;
  MockEval eval;

  // Create LocalField using named constructor
  auto world = pfc::world::create(pfc::GridSize({static_cast<int>(n), 1, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  pfc::field::LocalField<double> u = pfc::field::LocalField<double>::from_subdomain(decomp, /*rank=*/0, /*halo_width=*/0);

  auto tableau = make_rk4_classical<double>();
  auto stepper = create(u, eval, model, dt, tableau);

  std::vector<double> u_vec(n, 0.0);
  stepper.step(0.0, u_vec);

  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u_vec[i] == Approx(dt * 3.0)); // RK4 sum(b_i) = 1.0
  }
}

TEST_CASE("MultiExplicitRKStepper factory with tuple", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;

  // Mock multi-field model returning tuple
  struct MockModel {
    std::tuple<double, double> rhs(double /*t*/, double /*g*/) const {
      return {1.0, 2.0};
    }
  };

  // Mock composite evaluator
  struct MockEval {
    std::size_t size() const { return n; }
    void prepare() {}
    int imin() const { return 0; }
    int imax() const { return static_cast<int>(n); }
    int jmin() const { return 0; }
    int jmax() const { return 1; }
    int kmin() const { return 0; }
    int kmax() const { return 1; }
    std::size_t idx(int ix, int /*iy*/, int /*iz*/) const { return static_cast<std::size_t>(ix); }
    double operator()(int /*ix*/, int /*iy*/, int /*iz*/) const { return 0.0; }
  };

  MockModel model;
  MockEval eval;

  // Create LocalFields using named constructor
  auto world = pfc::world::create(pfc::GridSize({static_cast<int>(n), 1, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  pfc::field::LocalField<double> u1 = pfc::field::LocalField<double>::from_subdomain(decomp, /*rank=*/0, /*halo_width=*/0);
  pfc::field::LocalField<double> u2 = pfc::field::LocalField<double>::from_subdomain(decomp, /*rank=*/0, /*halo_width=*/0);

  auto fields = std::tie(u1, u2);
  auto tableau = make_rk4_classical<double>();
  auto stepper = create(fields, eval, model, dt, tableau);

  std::vector<double> u1_vec(n, 0.0);
  std::vector<double> u2_vec(n, 0.0);
  stepper.step(0.0, u1_vec, u2_vec);

  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u1_vec[i] == Approx(dt * 1.0)); // 0.1
    REQUIRE(u2_vec[i] == Approx(dt * 2.0)); // 0.2
  }
}
