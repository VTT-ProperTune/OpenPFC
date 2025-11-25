// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @example 03_simulator_workflow.cpp
 * @brief Demonstrates Simulator API for running complete simulations
 *
 * This example shows:
 * - Setting up a complete simulation workflow
 * - Registering initial conditions
 * - Adding results writers
 * - Running the main simulation loop
 * - Using Simulator callbacks
 *
 * Expected output:
 * - Simulation progress messages
 * - Time stepping information
 * - Results file creation
 *
 * Time to run: < 5 seconds
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/model.hpp>
#include <openpfc/mpi.hpp>
#include <openpfc/simulator.hpp>
#include <openpfc/time.hpp>

using namespace pfc;

/**
 * @brief Simple diffusion model for demonstration
 *
 * Implements: ∂u/∂t = D∇²u
 * Semi-implicit: u(t+dt) = u(t) / (1 - D*dt*k²)
 */
class DiffusionModel : public Model {
private:
  double m_diffusion_coeff = 0.1;
  std::vector<double> m_propagator; // Precomputed (1 - D*dt*k²)^{-1}

public:
  DiffusionModel(FFT &fft, const World &world) : Model(fft, world) {}

  void initialize(double dt) override {
    if (is_rank0()) {
      std::cout << "Initializing diffusion model with D = " << m_diffusion_coeff
                << "\n";
    }

    // Register fields
    auto &fft = get_fft();
    m_real_field.resize(fft.size_inbox(), 0.0);
    m_complex_field.resize(fft.size_outbox());

    add_real_field("concentration", m_real_field);
    add_complex_field("concentration_k", m_complex_field);

    // Precompute propagator in k-space
    auto outbox = fft::get_outbox(fft);
    auto world = get_world();
    auto size = world::get_size(world);
    auto spacing = world::get_spacing(world);

    m_propagator.resize(fft.size_outbox());

    for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
      for (int j = outbox.low[1]; j <= outbox.high[1]; ++j) {
        for (int k = outbox.low[2]; k <= outbox.high[2]; ++k) {
          size_t idx = (i - outbox.low[0]) * (outbox.high[1] - outbox.low[1] + 1) *
                           (outbox.high[2] - outbox.low[2] + 1) +
                       (j - outbox.low[1]) * (outbox.high[2] - outbox.low[2] + 1) +
                       (k - outbox.low[2]);

          // Wavenumbers
          double kx = (i < size[0] / 2) ? i : i - size[0];
          double ky = (j < size[1] / 2) ? j : j - size[1];
          double kz = k;

          kx *= 2.0 * M_PI / (size[0] * spacing[0]);
          ky *= 2.0 * M_PI / (size[1] * spacing[1]);
          kz *= 2.0 * M_PI / (size[2] * spacing[2]);

          double k2 = kx * kx + ky * ky + kz * kz;

          // Semi-implicit operator
          m_propagator[idx] = 1.0 / (1.0 + m_diffusion_coeff * dt * k2);
        }
      }
    }
  }

  void step(double t) override {
    auto &u = get_real_field("concentration");
    auto &u_k = get_complex_field("concentration_k");
    auto &fft = get_fft();

    // Transform to k-space
    fft.forward(u, u_k);

    // Apply diffusion operator (semi-implicit)
    for (size_t i = 0; i < u_k.size(); ++i) {
      u_k[i] *= m_propagator[i];
    }

    // Transform back
    fft.backward(u_k, u);
  }

private:
  std::vector<double> m_real_field;
  std::vector<std::complex<double>> m_complex_field;
};

/**
 * @brief Custom field modifier: Gaussian initial condition
 */
class GaussianIC : public FieldModifier {
private:
  std::string m_field_name;
  types::Real3 m_center;
  double m_amplitude;
  double m_sigma;

public:
  GaussianIC(const std::string &field_name, const types::Real3 &center,
             double amplitude = 1.0, double sigma = 1.0)
      : m_field_name(field_name), m_center(center), m_amplitude(amplitude),
        m_sigma(sigma) {}

  std::string get_field_name() const override { return m_field_name; }

  void apply(Model &model, double t) override {
    auto &field = model.get_real_field(m_field_name);
    auto &fft = model.get_fft();
    auto inbox = fft::get_inbox(fft);
    auto world = model.get_world();
    auto spacing = world::get_spacing(world);

    for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
      for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
        for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
          size_t idx = (i - inbox.low[0]) * (inbox.high[1] - inbox.low[1] + 1) *
                           (inbox.high[2] - inbox.low[2] + 1) +
                       (j - inbox.low[1]) * (inbox.high[2] - inbox.low[2] + 1) +
                       (k - inbox.low[2]);

          // Position
          double x = i * spacing[0] - m_center[0];
          double y = j * spacing[1] - m_center[1];
          double z = k * spacing[2] - m_center[2];
          double r2 = x * x + y * y + z * z;

          // Gaussian profile
          field[idx] = m_amplitude * std::exp(-r2 / (2.0 * m_sigma * m_sigma));
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
  void write(int iteration, const RealField &field) override {
    // Compute statistics
    double sum = 0.0, sum2 = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    for (double val : field) {
      sum += val;
      sum2 += val * val;
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    // Global reduction
    double global_sum, global_sum2, global_min, global_max;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum2, &global_sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&min_val, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_val, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (mpi::get_rank() == 0) {
      size_t total_points = field.size() * mpi::get_size();
      double mean = global_sum / total_points;
      double variance = global_sum2 / total_points - mean * mean;
      double stddev = std::sqrt(variance);

      std::cout << "Iteration " << std::setw(4) << iteration << ": "
                << "mean=" << std::fixed << std::setprecision(6) << mean
                << ", std=" << stddev << ", range=[" << global_min << ", "
                << global_max << "]\n";
    }
  }

  void write(int iteration, const ComplexField &field) override {
    // Not implemented for this example
  }
};

void example_complete_simulation() {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "  Complete Simulation Workflow\n";
  std::cout << std::string(60, '=') << "\n\n";

  // 1. Create computational domain
  auto world = world::create({64, 64, 64}, {0, 0, 0}, {0.1, 0.1, 0.1});

  if (mpi::get_rank() == 0) {
    std::cout << "Step 1: Created 64³ computational domain\n";
    std::cout << "  Physical size: 6.4 × 6.4 × 6.4\n\n";
  }

  // 2. Set up FFT
  auto decomp = decomposition::create(world, MPI_COMM_WORLD);
  auto fft = fft::create(decomp);

  if (mpi::get_rank() == 0) {
    std::cout << "Step 2: Initialized FFT\n";
    std::cout << "  MPI ranks: " << mpi::get_size() << "\n\n";
  }

  // 3. Create model
  DiffusionModel model(fft, world);

  if (mpi::get_rank() == 0) {
    std::cout << "Step 3: Created diffusion model\n\n";
  }

  // 4. Set up time integration
  // Simulate from t=0 to t=10 with dt=0.01, save every 1.0 time units
  Time time({0.0, 10.0, 0.01}, 1.0);

  if (mpi::get_rank() == 0) {
    std::cout << "Step 4: Configured time integration\n";
    std::cout << "  Duration: " << time.get_tspan()[1] << " time units\n";
    std::cout << "  Time step: " << time.get_dt() << "\n";
    std::cout << "  Save interval: " << time.get_saveat() << "\n\n";
  }

  // 5. Create simulator
  Simulator sim(model, time);
  sim.initialize(); // Calls model.initialize()

  if (mpi::get_rank() == 0) {
    std::cout << "Step 5: Created simulator\n\n";
  }

  // 6. Add initial condition
  types::Real3 center = {3.2, 3.2, 3.2}; // Center of domain
  sim.add_initial_conditions(
      std::make_unique<GaussianIC>("concentration", center, 1.0, 0.5));

  if (mpi::get_rank() == 0) {
    std::cout << "Step 6: Added Gaussian initial condition\n";
    std::cout << "  Center: (" << center[0] << ", " << center[1] << ", " << center[2]
              << ")\n";
    std::cout << "  σ = 0.5\n\n";
  }

  // 7. Add results writer
  sim.add_results_writer("concentration", std::make_unique<StatsWriter>());

  if (mpi::get_rank() == 0) {
    std::cout << "Step 7: Added statistics writer\n\n";
    std::cout << "Step 8: Running simulation...\n\n";
  }

  // 8. Run simulation loop
  while (!sim.done()) {
    sim.step();
  }

  if (mpi::get_rank() == 0) {
    std::cout << "\n✓ Simulation completed successfully!\n";
    std::cout << "  Final time: " << sim.get_time().get_current() << "\n";
    std::cout << "  Total steps: " << sim.get_increment() << "\n";
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (mpi::get_rank() == 0) {
    std::cout << "OpenPFC Simulator API Example\n";
    std::cout << "==============================\n";
    std::cout << "\nThis example demonstrates a complete simulation workflow:\n";
    std::cout << "  - Model setup and initialization\n";
    std::cout << "  - Initial conditions\n";
    std::cout << "  - Time integration loop\n";
    std::cout << "  - Results output\n";
  }

  try {
    example_complete_simulation();

    if (mpi::get_rank() == 0) {
      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "  Summary\n";
      std::cout << std::string(60, '=') << "\n\n";
      std::cout << "Key takeaways:\n";
      std::cout << "  ✓ Simulator orchestrates the entire workflow\n";
      std::cout << "  ✓ Initial conditions applied automatically at t=0\n";
      std::cout << "  ✓ Results writers called at specified intervals\n";
      std::cout << "  ✓ Main loop: while (!sim.done()) { sim.step(); }\n";
      std::cout << "\nSee include/openpfc/simulator.hpp for complete API.\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error on rank " << mpi::get_rank() << ": " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
