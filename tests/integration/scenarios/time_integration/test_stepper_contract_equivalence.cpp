// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <cmath>
#include <vector>

using Catch::Approx;
using namespace pfc;
using namespace pfc::sim;

// Per-point grads aggregate for diffusion equation
struct DiffusionGrads {
  double xx{};
  double yy{};
  double zz{};
};

// Physics model for explicit Euler diffusion
struct DiffusionModel {
  double D = 1.0;
  [[nodiscard]] double rhs(double /*t*/, const DiffusionGrads& g) const noexcept {
    return D * (g.xx + g.yy + g.zz);
  }
};

// Helper: compute L2 norm of local field
double compute_local_l2(const std::vector<double>& u) {
  double l2 = 0.0;
  for (const auto& v : u) {
    l2 += v * v;
  }
  return l2;
}

// Helper: compute mean of local field
double compute_local_mean(const std::vector<double>& u) {
  double sum = 0.0;
  for (const auto& v : u) {
    sum += v;
  }
  return sum / static_cast<double>(u.size());
}

// Helper: compute max absolute difference between two local fields
double compute_local_max_diff(const std::vector<double>& u1,
                                const std::vector<double>& u2) {
  double max_diff = 0.0;
  for (size_t i = 0; i < u1.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(u1[i] - u2[i]));
  }
  return max_diff;
}

// Helper: apply Gaussian initial condition u(x,y,z) = exp(-r²/(4D))
void apply_gaussian_initial_condition(field::LocalField<double>& u, double D) {
  u.apply([&](double x, double y, double z) {
    double r2 = x * x + y * y + z * z;
    return std::exp(-r2 / (4.0 * D));
  });
}

TEST_CASE("Manual explicit Euler with spectral gradients",
          "[integration][time_integration][stepper_contract]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Setup: world, decomposition, FFT, field
  auto world = world::uniform(32, 1.0);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);
  auto u = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());

  // Parameters
  const double D = 1.0;
  const double dt = 0.0001;
  const int steps = 10;

  // Initial condition: Gaussian u(x,y,z) = exp(-r²/(4D))
  apply_gaussian_initial_condition(u, D);

  // Compute initial mean for mass conservation check
  double local_mean_initial = compute_local_mean(u.vec());
  double global_mean_initial = 0.0;
  MPI_Allreduce(&local_mean_initial, &global_mean_initial, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  global_mean_initial /= static_cast<double>(size);

  // Create spectral gradient evaluator
  auto grad = field::create<DiffusionGrads>(u, fft);

  // Manual explicit Euler time stepping
  std::vector<double> du(u.size(), 0.0);
  DiffusionModel model{D};

  for (int step = 0; step < steps; ++step) {
    // Compute RHS: du = D * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    // for_each_interior calls grad.prepare() internally
    for_each_interior(model, grad, du.data(), 0.0);
    
    // Explicit Euler update: u += dt * du
    for (size_t i = 0; i < u.size(); ++i) {
      u.vec()[i] += dt * du[i];
    }
  }

  // Validate: field is finite
  for (const auto& v : u.vec()) {
    REQUIRE(std::isfinite(v));
  }

  // Validate: mass is conserved (mean equals initial mean)
  double local_mean_final = compute_local_mean(u.vec());
  double global_mean_final = 0.0;
  MPI_Allreduce(&local_mean_final, &global_mean_final, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  global_mean_final /= static_cast<double>(size);
  
  REQUIRE(global_mean_final == Approx(global_mean_initial).margin(1e-10));
}

TEST_CASE("EulerStepper infrastructure with spectral gradients",
          "[integration][time_integration][stepper_contract]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Setup: world, decomposition, FFT, field
  auto world = world::uniform(32, 1.0);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);
  auto u = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());

  // Parameters
  const double D = 1.0;
  const double dt = 0.0001;
  const int steps = 10;

  // Initial condition: Gaussian u(x,y,z) = exp(-r²/(4D))
  apply_gaussian_initial_condition(u, D);

  // Compute initial mean for mass conservation check
  double local_mean_initial = compute_local_mean(u.vec());
  double global_mean_initial = 0.0;
  MPI_Allreduce(&local_mean_initial, &global_mean_initial, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  global_mean_initial /= static_cast<double>(size);

  // Create spectral gradient evaluator
  auto grad = field::create<DiffusionGrads>(u, fft);

  // Create model with RHS
  DiffusionModel model{D};

  // Create EulerStepper using factory function (matches approved proposal signature)
  auto stepper = pfc::sim::steppers::create(u, grad, model, dt);

  // Advance using stepper infrastructure
  double t = 0.0;
  for (int step = 0; step < steps; ++step) {
    t = stepper.step(t, u.vec());
  }

  // Validate: field is finite
  for (const auto& v : u.vec()) {
    REQUIRE(std::isfinite(v));
  }

  // Validate: mass is conserved (mean equals initial mean)
  double local_mean_final = compute_local_mean(u.vec());
  double global_mean_final = 0.0;
  MPI_Allreduce(&local_mean_final, &global_mean_final, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  global_mean_final /= static_cast<double>(size);
  
  REQUIRE(global_mean_final == Approx(global_mean_initial).margin(1e-10));
}

TEST_CASE("Stepper contract equivalence: manual vs infrastructure",
          "[integration][time_integration][stepper_contract]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Common setup
  auto world = world::uniform(32, 1.0);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);
  
  const double D = 1.0;
  const double dt = 0.0001;
  const int steps = 10;

  // Manual implementation
  auto u_manual = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());
  apply_gaussian_initial_condition(u_manual, D);
  auto grad_manual = field::create<DiffusionGrads>(u_manual, fft);
  std::vector<double> du_manual(u_manual.size(), 0.0);
  DiffusionModel model{D};
  
  for (int step = 0; step < steps; ++step) {
    // for_each_interior calls grad_manual.prepare() internally
    for_each_interior(model, grad_manual, du_manual.data(), 0.0);
    for (size_t i = 0; i < u_manual.size(); ++i) {
      u_manual.vec()[i] += dt * du_manual[i];
    }
  }

  // Infrastructure implementation
  auto u_infra = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());
  apply_gaussian_initial_condition(u_infra, D);
  auto grad_infra = field::create<DiffusionGrads>(u_infra, fft);
  auto stepper = pfc::sim::steppers::create(u_infra, grad_infra, model, dt);
  
  double t = 0.0;
  for (int step = 0; step < steps; ++step) {
    t = stepper.step(t, u_infra.vec());
  }

  // Compute L² difference
  double local_l2_diff = 0.0;
  for (size_t i = 0; i < u_manual.size(); ++i) {
    double diff = u_manual.vec()[i] - u_infra.vec()[i];
    local_l2_diff += diff * diff;
  }
  double global_l2_diff = 0.0;
  MPI_Allreduce(&local_l2_diff, &global_l2_diff, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  global_l2_diff = std::sqrt(global_l2_diff);
  
  // Compute mean values for mass conservation check
  double local_mean_manual = compute_local_mean(u_manual.vec());
  double local_mean_infra = compute_local_mean(u_infra.vec());
  double global_mean_manual = 0.0;
  double global_mean_infra = 0.0;
  MPI_Allreduce(&local_mean_manual, &global_mean_manual, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_mean_infra, &global_mean_infra, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  global_mean_manual /= static_cast<double>(size);
  global_mean_infra /= static_cast<double>(size);
  
  // Compute max absolute difference
  double local_max_diff = compute_local_max_diff(u_manual.vec(), u_infra.vec());
  double global_max_diff = 0.0;
  MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  // Validate equivalence within tolerances
  REQUIRE(global_l2_diff < 1e-12);
  REQUIRE(global_mean_manual == Approx(global_mean_infra).margin(1e-14));
  REQUIRE(global_max_diff < 1e-10);
}
