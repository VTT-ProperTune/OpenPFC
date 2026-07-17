// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

using namespace pfc;
using Catch::Approx;

// Per-point grads aggregate for decay ODE (only needs value, no derivatives)
struct DecayGrads {
  double value{};
};

// Legacy pattern: Model::step(double t) override
class LegacyDecayModel : public Model {
public:
  LegacyDecayModel(fft::IFFT &fft, const World &world)
      : Model(fft, world),
        m_u(get_size(world)[0] * get_size(world)[1] * get_size(world)[2], 1.0) {
    m_nx = get_size(world)[0];
    m_ny = get_size(world)[1];
    m_nz = get_size(world)[2];
  }

  void initialize(double dt) override {
    m_dt = dt;
    // Field already initialized to 1.0 in constructor
  }

  void step(double /*t*/) override {
    for (size_t i = 0; i < m_u.size(); ++i) {
      m_u[i] += m_dt * (-m_u[i]);
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

  // Legacy setup
  auto world = world::create(GridSize({nx, ny, nz}), PhysicalOrigin({0, 0, 0}),
                             GridSpacing({1, 1, 1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  LegacyDecayModel legacy_model(fft, world);
  legacy_model.initialize(dt);

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
  legacy_model.step(t);
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

  // Legacy setup
  auto world = world::create(GridSize({nx, ny, nz}), PhysicalOrigin({0, 0, 0}),
                             GridSpacing({1, 1, 1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  LegacyDecayModel legacy_model(fft, world);
  legacy_model.initialize(dt);

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
    legacy_model.step(t);
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

  // Legacy setup
  auto world = world::create(GridSize({nx, ny, nz}), PhysicalOrigin({0, 0, 0}),
                             GridSpacing({1, 1, 1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  LegacyDecayModel legacy_model(fft, world);
  legacy_model.initialize(dt);

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
    legacy_model.step(t);
    t = stepper.step(t, new_field);
  }

  // Verify equivalence
  for (size_t i = 0; i < new_field.size(); ++i) {
    REQUIRE(legacy_field[i] == Approx(new_field[i]).margin(1e-10));
  }
}
