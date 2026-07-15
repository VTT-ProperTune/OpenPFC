// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <iostream>
#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

/** \example 20_explicit_stepper_spectral.cpp
 *
 * Explicit forward-Euler time integration of the 3D heat equation
 * \f$\partial u / \partial t = D \nabla^2 u\f\) using a spectral gradient
 * evaluator. Demonstrates the same physics as the FD example but with
 * FFT-based gradients and no halo exchange.
 *
 * Key components:
 * - `HeatGrads`: per-point gradient aggregate (same as FD example)
 * - `HeatModel`: pure function `rhs(t, grads)` returning `du/dt` (same as FD)
 * - `SpectralCpuStack`: field management with FFT
 * - `pfc::field::create<HeatGrads>(u, fft)`: spectral gradient evaluator
 * - `pfc::sim::steppers::create(u, grad, model, dt)`: stepper factory (same signature as FD)
 *
 * Time loop pattern (no halo exchange needed for spectral):
 * ```cpp
 * for (int step = 0; step < n_steps; ++step) {
 *   t = stepper.step(t, u.vec());  // FFTs happen inside evaluator.prepare()
 * }
 * ```
 *
 * Run: `mpirun -np 4 ./20_explicit_stepper_spectral`
 */

// Same HeatGrads and HeatModel as FD example - physics unchanged
struct HeatGrads {
  double xx{};
  double yy{};
  double zz{};
};

struct HeatModel {
  double kD;

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

  // Same problem parameters as FD example
  constexpr int N = 32;
  constexpr double dx = 1.0;
  constexpr double D = 1.0;
  constexpr int n_steps = 40;
  const double dt = 0.15 * dx * dx / (6.0 * D);

  // Build spectral stack (World + Decomposition + FFT + inbox-sized LocalField)
  pfc::sim::stacks::SpectralCpuStack stack(
    pfc::GridSize{{N, N, N}},
    pfc::PhysicalOrigin{{0.0, 0.0, 0.0}},
    pfc::GridSpacing{{dx, dx, dx}},
    rank, nproc);

  // Initialize field with same Gaussian initial condition
  auto& u = stack.u();
  const double cx = 0.5 * static_cast<double>(N - 1);
  const double sigma = static_cast<double>(N) / 6.0;
  u.apply([&](double x, double y, double z) {
    const double r2 = (x - cx) * (x - cx) + (y - cx) * (y - cx) + (z - cx) * (z - cx);
    return std::exp(-r2 / (2.0 * sigma * sigma));
  });

  // Construct gradient evaluator: spectral (uses FFT internally)
  auto grad = pfc::field::create<HeatGrads>(u, stack.fft());

  // Build stepper via factory (same signature as FD)
  HeatModel model{D};
  auto stepper = pfc::sim::steppers::create(u, grad, model, dt);

  // Time integration loop (no halo exchange needed for spectral)
  double t = 0.0;
  double l2_norm = compute_l2_norm(u);
  if (rank == 0) {
    std::cout << "Spectral explicit Euler stepper\n";
    std::cout << "Initial L2 norm: " << l2_norm << "\n";
  }

  for (int step = 0; step < n_steps; ++step) {
    // Spectral: no halo exchange, evaluator.prepare() handles FFTs internally
    t = stepper.step(t, u.vec());

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
