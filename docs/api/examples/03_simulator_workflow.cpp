// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @example 03_simulator_workflow.cpp
 * @brief Demonstrates `pfc::sim::run` for a complete spectral simulation
 *
 * This example shows:
 * - Domain + `SpectralCPUStack` setup
 * - Applying a `FieldModifier` initial condition
 * - Time integration with `pfc::sim::run`
 * - Writing statistics on `Time::do_save()`
 *
 * Time to run: < 5 seconds
 */

#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using namespace pfc;

/**
 * @brief Custom field modifier: Gaussian initial condition
 */
class GaussianIC : public FieldModifier {
private:
  Real3 m_center;
  double m_amplitude;
  double m_sigma;

public:
  GaussianIC(const std::string &field_name, const Real3 &center,
             double amplitude = 1.0, double sigma = 1.0)
      : m_center(center), m_amplitude(amplitude), m_sigma(sigma) {
    set_field_name(field_name);
  }

  void apply(pfc::field::FieldOutput<double> field, const Domain &domain, const Box3i &box,
             double /*t*/) override {
    auto spacing = domain::get_spacing(domain);
    int idx = 0;
    for (int k = box.low[2]; k <= box.high[2]; ++k) {
      for (int j = box.low[1]; j <= box.high[1]; ++j) {
        for (int i = box.low[0]; i <= box.high[0]; ++i) {
          double x = i * spacing[0] - m_center[0];
          double y = j * spacing[1] - m_center[1];
          double z = k * spacing[2] - m_center[2];
          double r2 = x * x + y * y + z * z;
          field[idx++] = m_amplitude * std::exp(-r2 / (2.0 * m_sigma * m_sigma));
        }
      }
    }
  }
};

/**
 * @brief Simple writer that prints statistics
 */
class StatsWriter : public ResultsWriter {
public:
  void set_domain(const std::array<int, 3> & /*arr_global*/,
                  const std::array<int, 3> & /*arr_local*/,
                  const std::array<int, 3> & /*arr_offset*/) override {}

  MPI_Status write(int iteration, const pfc::field::FieldOutput<double> field) override {
    double sum = 0.0, sum2 = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    for (double val : field) {
      sum += val;
      sum2 += val * val;
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    double global_sum, global_sum2, global_min, global_max;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum2, &global_sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&min_val, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_val, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (mpi::get_rank() == 0) {
      size_t total_points = field.size() * static_cast<size_t>(mpi::get_size());
      double mean = global_sum / static_cast<double>(total_points);
      double variance = global_sum2 / static_cast<double>(total_points) - mean * mean;
      double stddev = std::sqrt(std::max(0.0, variance));

      std::cout << "Iteration " << std::setw(4) << iteration << ": "
                << "mean=" << std::fixed << std::setprecision(6) << mean
                << ", std=" << stddev << ", range=[" << global_min << ", "
                << global_max << "]\n";
    }
    MPI_Status st{};
    return st;
  }

  MPI_Status write(int /*iteration*/, pfc::field::FieldView<std::complex<double>> /*field*/) override {
    MPI_Status st{};
    return st;
  }
};

void example_complete_simulation() {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "  Complete Simulation Workflow\n";
  std::cout << std::string(60, '=') << "\n\n";

  auto domain =
      domain::create(GridSize({64, 64, 64}), PhysicalOrigin({0.0, 0.0, 0.0}),
                     GridSpacing({0.1, 0.1, 0.1}));

  if (mpi::get_rank() == 0) {
    std::cout << "Step 1: Created 64³ computational domain\n";
    std::cout << "  Physical size: 6.4 × 6.4 × 6.4\n\n";
  }

  sim::stacks::SpectralCPUStack stack(std::move(domain), mpi::get_rank(),
                                      mpi::get_size());
  auto &fft = stack.fft();
  auto &psi = stack.u();

  if (mpi::get_rank() == 0) {
    std::cout << "Step 2: Built SpectralCPUStack (decomposition + FFT + field)\n";
    std::cout << "  MPI ranks: " << mpi::get_size() << "\n\n";
  }

  Time time({0.0, 10.0, 0.01}, 1.0);
  if (mpi::get_rank() == 0) {
    std::cout << "Step 3: Configured time integration\n";
    std::cout << "  Duration: " << pfc::time::t1(time) << " time units\n";
    std::cout << "  Time step: " << pfc::time::dt(time) << "\n";
    std::cout << "  Save interval: " << pfc::time::saveat(time) << "\n\n";
  }

  GaussianIC ic("concentration", Real3{3.2, 3.2, 3.2}, 1.0, 0.5);
  ic.apply(psi.vec(), psi.domain(), psi.box(), 0.0);
  if (mpi::get_rank() == 0) {
    std::cout << "Step 4: Applied Gaussian initial condition\n";
    std::cout << "  Center: (3.2, 3.2, 3.2), σ = 0.5\n\n";
  }

  StatsWriter writer;
  writer.set_domain(domain::get_size(psi.domain()), get_inbox(fft).size,
                    get_inbox(fft).low);

  constexpr double D = 0.1;
  const double dt = pfc::time::dt(time);
  std::vector<double> propagator(fft.size_outbox());
  std::vector<std::complex<double>> psi_F(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      get_outbox(fft), psi.domain(),
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        const double k2 = ki * ki + kj * kj + kk * kk;
        propagator[idx] = 1.0 / (1.0 + D * dt * k2);
      });

  if (mpi::get_rank() == 0) {
    std::cout << "Step 5: Prepared implicit-Euler diffusion operator\n\n";
    std::cout << "Step 6: Running simulation...\n\n";
  }

  pfc::sim::run(
      time,
      [&](double) {
        fft.forward(psi.vec(), psi_F);
        for (std::size_t i = 0; i < psi_F.size(); ++i) {
          psi_F[i] *= propagator[i];
        }
        fft.backward(psi_F, psi.vec());
      },
      {}, {},
      [&](const Time &clock) {
        writer.write(pfc::time::increment(clock), psi.vec());
      });

  if (mpi::get_rank() == 0) {
    std::cout << "\n✓ Simulation completed successfully!\n";
    std::cout << "  Final time: " << pfc::time::current(time) << "\n";
    std::cout << "  Total steps: " << pfc::time::increment(time) << "\n";
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (mpi::get_rank() == 0) {
    std::cout << "OpenPFC SimulationDriver API Example\n";
    std::cout << "====================================\n";
    std::cout << "\nThis example demonstrates a complete simulation workflow:\n";
    std::cout << "  - SpectralCPUStack setup\n";
    std::cout << "  - Initial conditions via FieldModifier\n";
    std::cout << "  - Time integration loop (`pfc::sim::run`)\n";
    std::cout << "  - Results output on save ticks\n";
  }

  try {
    example_complete_simulation();

    if (mpi::get_rank() == 0) {
      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "  Summary\n";
      std::cout << std::string(60, '=') << "\n\n";
      std::cout << "Key takeaways:\n";
      std::cout << "  ✓ SpectralCPUStack owns domain, FFT, and the host field\n";
      std::cout << "  ✓ FieldModifier::apply takes std::vector<double> + Domain + Box3i\n";
      std::cout << "  ✓ pfc::sim::run drives Time and optional save hooks\n";
      std::cout << "\nSee include/openpfc/kernel/simulation/simulation_driver.hpp.\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error on rank " << mpi::get_rank() << ": " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
