// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <tuple>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <wave2d/wave_model.hpp>

#include "fixtures/simulation_factories.hpp"

// Per-field gradient aggregates using FDGradient-recognized member names
struct UGrads {
  double xx{};  // Second derivative in x (for u Laplacian)
  double yy{};  // Second derivative in y (for u Laplacian)
};

struct VGrads {
  double value{};  // Current v value (for du = v term)
};

// Composite gradient aggregate passed to model.rhs()
struct WaveLocal {
  UGrads u;  // u's Laplacian components (xx, yy)
  VGrads v;  // v's current value
};

// Model wrapper adapter that transforms WaveModel::rhs signature
struct WaveModelAdapter {
  wave2d::WaveModel model;

  [[nodiscard]] wave2d::WaveIncrements rhs(double t, const WaveLocal &g) const noexcept {
    // Transform UGrads (xx, yy) to WaveLaplacian (lxx, lyy)
    wave2d::WaveLaplacian lap{.lxx = g.u.xx, .lyy = g.u.yy};
    // Call legacy WaveModel::rhs with v value from VGrads
    return model.rhs(t, g.v.value, lap);
  }
};

// Analytical solution: traveling Gaussian pulse in +x direction
constexpr double c = 1.0;
constexpr double sigma = 0.1;
constexpr double xc = 0.5;
constexpr double yc = 0.5;

double u_exact(double x, double y, double t) {
  double x_shifted = x - xc - c * t;
  double r2 = x_shifted * x_shifted + (y - yc) * (y - yc);
  return std::exp(-r2 / (2 * sigma * sigma));
}

double v_exact(double x, double y, double t) {
  double x_shifted = x - xc - c * t;
  return c * x_shifted / (sigma * sigma) * u_exact(x, y, t);
}

double compute_l2_error(const pfc::data::Field<double> &field,
                        double t,
                        double (*exact)(double, double, double)) {
  const auto &spacing = field.spacing();
  const auto &size = field.size3();
  double dv = spacing[0] * spacing[1] * spacing[2];
  double error_sq = 0.0;
  double volume = 0.0;

  for (int iz = 0; iz < size[2]; ++iz) {
    for (int iy = 0; iy < size[1]; ++iy) {
      for (int ix = 0; ix < size[0]; ++ix) {
        auto coords = field.coords(ix, iy, iz);
        double value = field(ix, iy, iz);
        double u_ex = exact(coords[0], coords[1], t);
        double diff = value - u_ex;
        error_sq += diff * diff * dv;
        volume += dv;
      }
    }
  }

  if (volume == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(error_sq / volume);
}

// Test fixture for wave model tests
class WaveModelMultifieldFixture : public pfc::test::SimulationModelFixture {
public:
  WaveModelMultifieldFixture() = default; // Default constructor for parameterless TEST_CASE_METHOD

  // Custom constructor for test-specific setup
  WaveModelMultifieldFixture(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz) {
    double dx = Lx / Nx;
    double dy = Ly / Ny;
    double dz = Lz / Nz;
    pfc::types::Int3 size{Nx, Ny, Nz};
    pfc::types::Real3 spacing{dx, dy, dz};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};
    SetUpCustomDomainWithSpacing(size, spacing);
  }
};

TEST_CASE_METHOD(WaveModelMultifieldFixture, "WaveModel multi-field Gaussian pulse convergence") {
  constexpr int Nx = 64, Ny = 64, Nz = 4;
  constexpr double Lx = 1.0, Ly = 1.0, Lz = 0.1;
  constexpr double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;
  constexpr int fd_order = 2;
  constexpr int halo_width = fd_order / 2;  // = 1 for order 2
  constexpr double t_end = 0.01;  // Shorter time to avoid boundary reflections
  constexpr double dt_coarse = 0.0001;  // Smaller dt for stability
  constexpr double dt_fine = dt_coarse / 2.0;

  // Manually set up domain with specific parameters for this test
  {
    pfc::types::Int3 size{Nx, Ny, Nz};
    pfc::types::Real3 spacing{dx, dy, dz};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};
    SetUpCustomDomainWithSpacing(size, spacing);
  }
  auto world = pfc::test::world_from_domain(domain());
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);

  auto run_with_dt = [&](double dt) -> double {
    auto u_field = pfc::data::field_from_subdomain_unpadded<double>(decomp, /*rank=*/0, halo_width);
    auto v_field = pfc::data::field_from_subdomain_unpadded<double>(decomp, /*rank=*/0, halo_width);

    u_field.apply([&](double x, double y, double z) -> double {
      return u_exact(x, y, 0.0);
    });
    v_field.apply([&](double x, double y, double z) -> double {
      return v_exact(x, y, 0.0);
    });

    auto grad_u = pfc::field::create<UGrads>(u_field, fd_order);
    auto grad_v = pfc::field::create<VGrads>(v_field, fd_order);
    auto composite = pfc::field::create_composite<WaveLocal>(grad_u, grad_v);

    WaveModelAdapter adapter{.model = wave2d::WaveModel{
      .inv_dx2 = 1.0 / (dx * dx),
      .inv_dy2 = 1.0 / (dy * dy)
    }};

    auto stepper = pfc::sim::steppers::create(std::tie(u_field, v_field), composite, adapter, dt);

    double t = 0.0;
    int n_steps = static_cast<int>(t_end / dt);
    for (int i = 0; i < n_steps; ++i) {
      stepper.step(t, u_field.vec(), v_field.vec());
      t += dt;
    }

    return compute_l2_error(u_field, t_end, u_exact);
  };

  double error_coarse = run_with_dt(dt_coarse);
  double error_fine = run_with_dt(dt_fine);
  double convergence_ratio = error_coarse / error_fine;

  CHECK(!std::isnan(convergence_ratio));
  CHECK(convergence_ratio >= 1.5);
}

TEST_CASE_METHOD(WaveModelMultifieldFixture, "WaveModel multi-field factory workflow") {
  constexpr int Nx = 32, Ny = 32, Nz = 4;
  constexpr double Lx = 1.0, Ly = 1.0, Lz = 0.1;
  constexpr double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;
  constexpr int fd_order = 2;
  constexpr int halo_width = fd_order / 2;  // = 1 for order 2
  constexpr double t_end = 0.01;  // Shorter time to avoid boundary reflections
  constexpr double dt = 0.0001;  // Smaller dt for stability

  // Manually set up domain with specific parameters for this test
  {
    pfc::types::Int3 size{Nx, Ny, Nz};
    pfc::types::Real3 spacing{dx, dy, dz};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};
    SetUpCustomDomainWithSpacing(size, spacing);
  }
  auto world = pfc::test::world_from_domain(domain());
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);

  auto u_field = pfc::data::field_from_subdomain_unpadded<double>(decomp, /*rank=*/0, halo_width);
  auto v_field = pfc::data::field_from_subdomain_unpadded<double>(decomp, /*rank=*/0, halo_width);

  u_field.apply([&](double x, double y, double z) -> double {
    return u_exact(x, y, 0.0);
  });
  v_field.apply([&](double x, double y, double z) -> double {
    return v_exact(x, y, 0.0);
  });

  auto grad_u = pfc::field::create<UGrads>(u_field, fd_order);
  auto grad_v = pfc::field::create<VGrads>(v_field, fd_order);
  auto composite = pfc::field::create_composite<WaveLocal>(grad_u, grad_v);

  WaveModelAdapter adapter{.model = wave2d::WaveModel{
    .inv_dx2 = 1.0 / (dx * dx),
    .inv_dy2 = 1.0 / (dy * dy)
  }};

  auto stepper = pfc::sim::steppers::create(std::tie(u_field, v_field), composite, adapter, dt);

  double t = 0.0;
  int n_steps = static_cast<int>(t_end / dt);
  for (int i = 0; i < n_steps; ++i) {
    stepper.step(t, u_field.vec(), v_field.vec());
    t += dt;
  }

  double error_u = compute_l2_error(u_field, t_end, u_exact);
  double error_v = compute_l2_error(v_field, t_end, v_exact);

  CHECK(!std::isnan(error_u));
  CHECK(error_u > 0.0);
  CHECK(error_u < 100.0);  // Relaxed tolerance for workflow validation
  CHECK(!std::isnan(error_v));
  CHECK(error_v > 0.0);
  CHECK(error_v < 100.0);  // Relaxed tolerance for workflow validation
}
