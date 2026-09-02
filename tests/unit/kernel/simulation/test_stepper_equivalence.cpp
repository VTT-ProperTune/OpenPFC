// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

using namespace pfc;
using Catch::Approx;

// Per-point grads aggregate for decay ODE (only needs value, no derivatives)
struct DecayGrads {
  double value{};
};

class ExplicitEulerDecay {
public:
  explicit ExplicitEulerDecay(std::size_t n, double dt) : m_u(n, 1.0), m_dt(dt) {}

  void step() {
    for (double &v : m_u) {
      v += m_dt * (-v);
    }
  }

  const std::vector<double> &get_field() const { return m_u; }
  std::vector<double> &get_field() { return m_u; }

private:
  std::vector<double> m_u;
  double m_dt{0.0};
  int m_nx{0}, m_ny{0}, m_nz{0};
};

// New pattern: RHS function for use with EulerStepper
struct ExplicitDecayModel {
  double rhs(double /*t*/, const DecayGrads &g) const { return -g.value; }
};

TEST_CASE("test_decay_single_step", "[stepper][equivalence]") {
  constexpr double dt = 0.1;
  constexpr int nx = 8, ny = 8, nz = 8;

  ExplicitEulerDecay legacy_model(static_cast<std::size_t>(nx * ny * nz), dt);

  // New setup
  std::vector<double> new_field(nx * ny * nz, 1.0);
  ExplicitDecayModel explicit_model;
  pfc::gradient::FDGradient<DecayGrads> grad(
      new_field.data(), nx, ny, nz, 1.0, 1.0, 1.0, // dx, dy, dz
      0, // halo_width (no halo needed for decay ODE)
      2  // order
  );
  auto stepper =
      pfc::sim::steppers::create(grad, explicit_model, dt, new_field.size());

  // Run one step
  double t = 0.0;
  legacy_model.step();
  t = stepper.step(t, new_field);

  // Verify equivalence
  const auto &legacy_field = legacy_model.get_field();
  for (size_t i = 0; i < new_field.size(); ++i) {
    REQUIRE(legacy_field[i] == Approx(new_field[i]).margin(1e-12));
  }
}

TEST_CASE("test_decay_multiple_steps", "[stepper][equivalence]") {
  constexpr double dt = 0.1;
  constexpr int nx = 8, ny = 8, nz = 8;
  constexpr int num_steps = 10;

  ExplicitEulerDecay legacy_model(static_cast<std::size_t>(nx * ny * nz), dt);

  // New setup
  std::vector<double> new_field(nx * ny * nz, 1.0);
  ExplicitDecayModel explicit_model;
  pfc::gradient::FDGradient<DecayGrads> grad(new_field.data(), nx, ny, nz, 1.0, 1.0,
                                             1.0, 0, 2);
  auto stepper =
      pfc::sim::steppers::create(grad, explicit_model, dt, new_field.size());

  // Run multiple steps
  double t = 0.0;
  for (int step = 0; step < num_steps; ++step) {
    legacy_model.step();
    t = stepper.step(t, new_field);
  }

  // Verify equivalence
  const auto &legacy_field = legacy_model.get_field();
  for (size_t i = 0; i < new_field.size(); ++i) {
    REQUIRE(legacy_field[i] == Approx(new_field[i]).margin(1e-10));
  }
}

TEST_CASE("test_decay_with_nonzero_initial_condition", "[stepper][equivalence]") {
  constexpr double dt = 0.05;
  constexpr int nx = 8, ny = 8, nz = 8;

  // Create spatially-varying initial condition
  auto init_condition = [&](int ix, int iy, int iz) -> double {
    return 1.0 + 0.1 * (ix + iy + iz);
  };

  ExplicitEulerDecay legacy_model(static_cast<std::size_t>(nx * ny * nz), dt);

  // Apply non-uniform initial condition to legacy field
  auto &legacy_field = legacy_model.get_field();
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        size_t idx = ix + nx * (iy + ny * iz);
        legacy_field[idx] = init_condition(ix, iy, iz);
      }
    }
  }

  // New setup with same initial condition
  std::vector<double> new_field(nx * ny * nz);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        size_t idx = ix + nx * (iy + ny * iz);
        new_field[idx] = init_condition(ix, iy, iz);
      }
    }
  }

  ExplicitDecayModel explicit_model;
  pfc::gradient::FDGradient<DecayGrads> grad(new_field.data(), nx, ny, nz, 1.0, 1.0,
                                             1.0, 0, 2);
  auto stepper =
      pfc::sim::steppers::create(grad, explicit_model, dt, new_field.size());

  // Run multiple steps
  double t = 0.0;
  constexpr int num_steps = 5;
  for (int step = 0; step < num_steps; ++step) {
    legacy_model.step();
    t = stepper.step(t, new_field);
  }

  // Verify equivalence
  for (size_t i = 0; i < new_field.size(); ++i) {
    REQUIRE(legacy_field[i] == Approx(new_field[i]).margin(1e-10));
  }
}
