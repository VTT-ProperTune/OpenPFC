// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 10_complete_pfc_model.cpp
 * @brief Complete Phase Field Crystal simulation demonstrating OpenPFC 0.2 APIs
 *
 * This comprehensive example demonstrates:
 * 1. Domain - Domain geometry and coordinate system
 * 2. SpectralCPUStack - decomposition, FFT, and host field
 * 3. FFT - Spectral transforms and k-space operations
 * 4. Physics - single-mode PFC on the stack field
 * 5. Initial Conditions - Seed-based nucleation
 * 6. Boundary Conditions - periodic (FFT); FixedBC is available in apps
 * 7. Time - Time stepping and output scheduling
 * 8. SimulationDriver - `pfc::sim::run` time loop
 * 9. ResultsWriter - Parallel output to binary files
 * 10. Full integration - Production-quality PFC simulation
 *
 * Physical System:
 *   Phase Field Crystal (PFC) model for solidification
 *   Single-mode approximation with periodic boundary conditions
 *   Liquid → Solid phase transition driven by undercooling
 *
 * Compile and run:
 *   mpicxx -std=c++20 -I/path/to/openpfc/include 10_complete_pfc_model.cpp \
 *          -L/path/to/openpfc/lib -lopenpfc -lheffte -o 10_complete_pfc_model
 *   mpirun -np 4 ./10_complete_pfc_model
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <numbers>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>
#include <openpfc/kernel/simulation/initial_conditions/single_seed.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc/openpfc.hpp>
#include <openpfc_apps/fixed_bc.hpp>

using namespace pfc;

/**
 * @brief Single-mode PFC: ∂ψ/∂t = ∇²[ε·ψ + ψ³ + (1 + ∇²)²ψ]
 *
 * Semi-implicit spectral step: linear terms via integrating factor, ψ³ explicit.
 */
struct PfcOperators {
  std::vector<double> opL;
  std::vector<double> opN;
};

PfcOperators make_pfc_operators(fft::IHostFFT &fft, const Domain &domain,
                                double epsilon, double dt) {
  PfcOperators ops;
  ops.opL.resize(fft.size_outbox());
  ops.opN.resize(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      get_outbox(fft), domain,
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        const double k2 = ki * ki + kj * kj + kk * kk;
        const double k2_lap = -k2;
        const double L = k2_lap * (epsilon + (1.0 - k2) * (1.0 - k2));
        ops.opL[idx] = std::exp(L * dt);
        if (std::abs(L) > 1e-14) {
          ops.opN[idx] = (ops.opL[idx] - 1.0) / L * k2_lap;
        } else {
          ops.opN[idx] = dt * k2_lap;
        }
      });
  return ops;
}

void pfc_step(fft::IHostFFT &fft, data::Field<double> &psi,
              std::vector<std::complex<double>> &psi_k,
              std::vector<std::complex<double>> &n_k, std::vector<double> &nonlinear,
              const PfcOperators &ops) {
  fft.forward(psi.vec(), psi_k);
  for (std::size_t i = 0; i < psi.vec().size(); ++i) {
    const double v = psi.vec()[i];
    nonlinear[i] = v * v * v;
  }
  fft.forward(nonlinear, n_k);
  for (std::size_t i = 0; i < psi_k.size(); ++i) {
    psi_k[i] = ops.opL[i] * psi_k[i] + ops.opN[i] * n_k[i];
  }
  fft.backward(psi_k, psi.vec());
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n";
    std::cout << "╔═════════════════════════════════════════════════════════════════"
                 "═══╗\n";
    std::cout << "║          Complete Phase Field Crystal Simulation                "
                 "   ║\n";
    std::cout << "║                                                                 "
                 "   ║\n";
    std::cout << "║  Demonstrates OpenPFC 0.2 APIs in a production-style workflow   "
                 "   ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════════"
                 "═══╝\n";
    std::cout << "\nRunning on " << size << " MPI ranks\n";
  }

  try {
    if (rank == 0) {
      std::cout << "\n[1] Creating domain with Domain API...\n";
    }

    auto domain =
        domain::create(GridSize({64, 64, 64}), PhysicalOrigin({0.0, 0.0, 0.0}),
                       GridSpacing({1.0, 1.0, 1.0}));

    if (rank == 0) {
      std::cout << "    Domain: " << domain::get_size(domain, 0) << " x "
                << domain::get_size(domain, 1) << " x "
                << domain::get_size(domain, 2) << " points\n";
    }

    if (rank == 0) {
      std::cout << "\n[2] Building SpectralCPUStack (decomposition + FFT)...\n";
    }

    sim::stacks::SpectralCPUStack stack(std::move(domain), rank, size);
    auto &fft = stack.fft();
    auto &psi = stack.u();

    if (rank == 0) {
      std::cout << "    Input box (real): " << fft.size_inbox() << " points\n";
      std::cout << "    Output box (complex): " << fft.size_outbox() << " points\n";
    }
    std::cout << "    [Rank " << rank << "] Local box size: " << psi.box().size[0]
              << " x " << psi.box().size[1] << " x " << psi.box().size[2] << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "\n[3] Preparing PFC operators...\n";
    }

    constexpr double epsilon = -0.25;
    constexpr double dt = 0.5;
    auto ops = make_pfc_operators(fft, psi.domain(), epsilon, dt);
    std::vector<std::complex<double>> psi_k(fft.size_outbox());
    std::vector<std::complex<double>> n_k(fft.size_outbox());
    std::vector<double> nonlinear(psi.vec().size());

    if (rank == 0) {
      std::cout << "    Model: Phase Field Crystal (single-mode)\n";
      std::cout << "    Epsilon (ε): " << epsilon << " (undercooled)\n";
      std::cout << "    Time step (dt): " << dt << "\n";
    }

    if (rank == 0) {
      std::cout << "\n[4] Applying initial conditions...\n";
    }

    Constant background(0.0);
    background.set_field_name("density");
    background.apply(psi.vec(), psi.domain(), psi.box(), 0.0);

    SingleSeed seed;
    seed.set_field_name("density");
    seed.set_density(0.0);
    seed.set_amplitude(0.3);
    seed.apply(psi.vec(), psi.domain(), psi.box(), 0.0);

    if (rank == 0) {
      std::cout << "    Applied Constant IC: ψ = 0.0 (liquid phase)\n";
      std::cout << "    Applied SingleSeed IC: BCC crystal at origin\n";
    }

    if (rank == 0) {
      std::cout << "\n[5] Boundary conditions: periodic (via FFT)\n";
      std::cout << "    (FixedBC is available in apps/common but unused here)\n";
    }
    FixedBC unused_walls(0.0, 0.0);
    unused_walls.set_field_name("density");
    (void)unused_walls;

    if (rank == 0) {
      std::cout << "\n[6] Configuring time integration and output...\n";
    }

    constexpr double t_start = 0.0;
    constexpr double t_end = 20.0;
    const double saveat = 10.0 * dt;
    Time time({t_start, t_end, dt}, saveat);
    BinaryWriter writer("pfc_output_{:06d}.bin");
    writer.set_domain(domain::get_size(psi.domain()), get_inbox(fft).size,
                      get_inbox(fft).low);

    if (rank == 0) {
      std::cout << "    Time span: [" << t_start << ", " << t_end << "]\n";
      std::cout << "    Output pattern: pfc_output_NNNNNN.bin\n";
    }

    if (rank == 0) {
      std::cout << "\n[7] Running simulation with pfc::sim::run...\n";
      std::cout << std::string(70, '-') << "\n";
    }

    pfc::sim::run(
        time,
        [&](double) { pfc_step(fft, psi, psi_k, n_k, nonlinear, ops); }, {}, {},
        [&](const Time &clock) {
          writer.write(pfc::time::increment(clock), psi.vec());
          if (rank == 0) {
            printf("    Step %4d  |  t = %6.2f  |  Saving output...\n",
                   pfc::time::increment(clock), pfc::time::current(clock));
          }
        });

    if (rank == 0) {
      std::cout << std::string(70, '-') << "\n";
      std::cout << "    Simulation complete!\n";
    }

    if (rank == 0) {
      std::cout << "\n[Post-processing] Computing final statistics...\n";
    }

    const auto &density = psi.vec();
    double local_min = *std::min_element(density.begin(), density.end());
    double local_max = *std::max_element(density.begin(), density.end());
    double local_sum = std::accumulate(density.begin(), density.end(), 0.0);
    int local_size = static_cast<int>(density.size());

    double global_min = 0.0, global_max = 0.0, global_sum = 0.0;
    int global_size = 0;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      double mean = global_sum / static_cast<double>(global_size);
      std::cout << "    Final density field statistics:\n";
      std::cout << "      Mean: " << mean << "\n";
      std::cout << "      Min:  " << global_min << "\n";
      std::cout << "      Max:  " << global_max << "\n";
      std::cout << "\n";
      std::cout << "  ✓ Domain + SpectralCPUStack + FieldModifier + pfc::sim::run\n";
      std::cout << "  ✓ " << pfc::time::increment(time) << " time steps completed\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
