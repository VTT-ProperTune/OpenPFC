// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <iostream>
#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

/** \example 19_explicit_stepper_fd.cpp
 *
 * Explicit forward-Euler time integration of the 3D heat equation
 * \f$\partial u / \partial t = D \nabla^2 u\f\) using a finite-difference
 * gradient evaluator. Demonstrates the complete wiring from model
 * `rhs(double t, const G& g)` through gradient evaluator construction to
 * time loop integration via `pfc::sim::steppers::create()` factory.
 *
 * Key components:
 * - `HeatGrads`: per-point gradient aggregate (only second derivatives needed)
 * - `HeatModel`: pure function `rhs(t, grads)` returning `du/dt`
 * - `FdCpuStack`: field management with halo exchange
 * - `pfc::field::create<HeatGrads>(u, order)`: FD gradient evaluator
 * - `pfc::sim::steppers::create(u, grad, model, dt)`: stepper factory
 *
 * Time loop pattern:
 * ```cpp
 * for (int step = 0; step < n_steps; ++step) {
 *   stack.exchange_halos();  // FD requires halo exchange before each step
 *   t = stepper.step(t, u.vec());
 * }
 * ```
 *
 * Run: `mpirun -np 4 ./19_explicit_stepper_fd`
 */

// Per-point gradient aggregate for heat equation (only second derivatives needed)
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

// Physics model: pure function rhs(t, grads) -> du/dt
struct HeatModel {
  double kD;  // thermal diffusivity

  [[nodiscard]] double rhs(double /*t*/, const HeatGrads& g) const noexcept {
    return kD * (g.xx + g.yy + g.zz);
  }
};

// Helper: compute global L2 norm across all ranks
double compute_l2_norm(const pfc::field::LocalField<double>& u) {
  double local_sum = 0.0;
  u.for_each_owned([&](double /*x*/, double /*y*/, double /*z*/, double val) {
    local_sum += val * val;
  });

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const auto& global_size = u.global_size();
  const double volume = static_cast<double>(global_size[0] * global_size[1] * global_size[2]);
  return std::sqrt(global_sum / volume);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // Problem parameters
  constexpr int N = 32;
  constexpr double dx = 1.0;
  constexpr double D = 1.0;
  constexpr int n_steps = 40;
  constexpr int fd_order = 2;  // second-order central differences

  // Stability: dt <= dx^2 / (6*D) for explicit Euler in 3D
  const double dt = 0.15 * dx * dx / (6.0 * D);

  // Build FD stack (World + Decomposition + halo-aware LocalField + exchanger)
  pfc::sim::stacks::FdCpuStack stack(
    pfc::GridSize{{N, N, N}},
    pfc::PhysicalOrigin{{0.0, 0.0, 0.0}},
    pfc::GridSpacing{{dx, dx, dx}},
    fd_order, rank, nproc);

  // Initialize field with Gaussian initial condition
  auto& u = stack.u();
  const double cx = 0.5 * static_cast<double>(N - 1);
  const double sigma = static_cast<double>(N) / 6.0;
  u.apply([&](double x, double y, double z) {
    const double r2 = (x - cx) * (x - cx) + (y - cx) * (y - cx) + (z - cx) * (z - cx);
    return std::exp(-r2 / (2.0 * sigma * sigma));
  });

  // Construct gradient evaluator: FD with order=2
  auto grad = pfc::field::create<HeatGrads>(u, fd_order);

  // Build stepper via factory: binds model + evaluator + dt
  HeatModel model{D};
  auto stepper = pfc::sim::steppers::create(u, grad, model, dt);

  // Time integration loop
  double t = 0.0;
  double l2_norm = compute_l2_norm(u);
  if (rank == 0) {
    std::cout << "Finite-difference explicit Euler stepper\n";
    std::cout << "Initial L2 norm: " << l2_norm << "\n";
  }

  for (int step = 0; step < n_steps; ++step) {
    // FD requires halo exchange before each step
    stack.exchange_halos();

    // Advance one time step
    t = stepper.step(t, u.vec());

    // Monitor dissipation
    l2_norm = compute_l2_norm(u);
    if (rank == 0 && step % 10 == 0) {
      std::cout << "Step " << step << ", t = " << t
                << ", L2 norm = " << l2_norm << "\n";
    }
  }

  if (rank == 0) {
    std::cout << "Final L2 norm: " << l2_norm << "\n";
    std::cout << "Heat dissipated: " << (1.0 - l2_norm) * 100.0 << "%\n";
  }

  MPI_Finalize();
  return 0;
}
