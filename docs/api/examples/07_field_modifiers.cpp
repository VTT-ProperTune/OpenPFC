// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 07_field_modifiers.cpp
 * @brief Comprehensive examples of the FieldModifier API
 *
 * This example demonstrates:
 * 1. Creating custom initial conditions
 * 2. Creating custom boundary conditions
 * 3. Space-time varying boundary conditions
 * 4. Composing multiple modifiers
 *
 * Compile and run:
 *   mpicxx -std=c++17 -I/path/to/openpfc/include 07_field_modifiers.cpp \
 *          -L/path/to/openpfc/lib -lopenpfc -lheffte -o 07_field_modifiers
 *   mpirun -np 4 ./07_field_modifiers
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <openpfc/openpfc.hpp>

using namespace pfc;

//==============================================================================
// Helper function for synchronized output
//==============================================================================

void print_section(const std::string &title) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 1: Custom Initial Condition - Gaussian Bump
//==============================================================================

/**
 * @brief Custom initial condition: Gaussian bump in 3D space
 *
 * This demonstrates:
 * - Deriving from FieldModifier
 * - Accessing model geometry (world, fft, inbox)
 * - Coordinate-space operations (world::to_coords)
 * - Spatial field initialization
 */
class GaussianIC : public FieldModifier {
private:
  Real3 m_center;      // Center position of Gaussian
  double m_amplitude;  // Peak amplitude
  double m_width;      // Width parameter (standard deviation)
  double m_background; // Background value

public:
  GaussianIC(Real3 center, double amplitude, double width, double background = 0.0)
      : m_center(center), m_amplitude(amplitude), m_width(width),
        m_background(background) {}

  void apply(Model &model, double time) override {
    // 1. Get the field to modify
    auto &field = model.get_real_field(get_field_name());

    // 2. Get geometry information
    const auto &world = model.get_world();
    const auto &fft = model.get_fft();
    auto inbox = fft::get_inbox(fft);

    // 3. Loop over local subdomain and set Gaussian profile
    int idx = 0;
    for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
      for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
        for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
          // Convert grid indices to physical coordinates
          auto pos = world::to_coords(world, Int3{i, j, k});

          // Compute distance from center
          double dx = pos[0] - m_center[0];
          double dy = pos[1] - m_center[1];
          double dz = pos[2] - m_center[2];
          double r2 = dx * dx + dy * dy + dz * dz;

          // Set Gaussian profile
          field[idx++] =
              m_background + m_amplitude * std::exp(-r2 / (2.0 * m_width * m_width));
        }
      }
    }

    // Note: No MPI communication needed - each rank operates on its subdomain
  }
};

void demo_custom_initial_condition() {
  print_section("SCENARIO 1: Custom Initial Condition - Gaussian Bump");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create a small domain for demonstration
  auto world = world::create(Int3{32, 32, 32}, Real3{1.0, 1.0, 1.0});

  // Create FFT with decomposition
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);

  // Create a simple model with one field
  Model model(world, std::move(fft));
  model.add_real_field("density");

  // Create and apply Gaussian IC
  GaussianIC gaussian_ic(Real3{16.0, 16.0, 16.0}, // Center at domain middle
                         1.0,                     // Amplitude
                         4.0,                     // Width (4 grid units)
                         0.5                      // Background value
  );
  gaussian_ic.set_field_name("density");
  gaussian_ic.apply(model, 0.0); // Apply at t=0

  // Verify field values at a few points
  const auto &field = model.get_real_field("density");
  auto inbox = fft::get_inbox(model.get_fft());

  if (rank == 0) {
    std::cout << "Applied Gaussian IC:\n";
    std::cout << "  Center: (16, 16, 16)\n";
    std::cout << "  Amplitude: 1.0, Width: 4.0, Background: 0.5\n";
    std::cout << "  Local subdomain: [" << inbox.low[0] << ":" << inbox.high[0]
              << ", " << inbox.low[1] << ":" << inbox.high[1] << ", " << inbox.low[2]
              << ":" << inbox.high[2] << "]\n";
    std::cout << "  First few values: ";
    for (int i = 0; i < std::min(5, static_cast<int>(field.size())); i++) {
      std::cout << field[i] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "[Rank " << rank << "] Field initialized with " << field.size()
            << " local points\n";
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 2: Custom Boundary Condition - Dirichlet Fixed Value
//==============================================================================

/**
 * @brief Custom boundary condition: Fixed Dirichlet BC at right boundary
 *
 * This demonstrates:
 * - Boundary condition application
 * - Smooth transition to avoid discontinuities
 * - Selective modification (only near boundaries)
 * - Time-independent BC (ignores time parameter)
 */
class DirichletBC : public FieldModifier {
private:
  double m_value; // Fixed value at boundary
  double m_width; // Transition width
  std::string m_name = "DirichletBC";

public:
  DirichletBC(double value, double width = 5.0) : m_value(value), m_width(width) {}

  const std::string &get_modifier_name() const override { return m_name; }

  void apply(Model &model, double time) override {
    auto &field = model.get_real_field(get_field_name());
    const auto &world = model.get_world();
    const auto &fft = model.get_fft();
    auto inbox = fft::get_inbox(fft);

    // Get domain size in x-direction
    double Lx = world::get_size(world, 0) * world::get_spacing(world, 0);
    double dx = world::get_spacing(world, 0);
    double x0 = world::get_origin(world, 0);

    // Apply BC at right boundary with smooth transition
    int idx = 0;
    for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
      for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
        for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
          double x = x0 + i * dx;

          // Only modify near right boundary
          if (x > Lx - m_width) {
            // Smooth interpolation from field value to BC value
            double s = (x - (Lx - m_width)) / m_width; // 0 to 1
            s = 3 * s * s - 2 * s * s * s;             // Smoothstep function
            field[idx] = field[idx] * (1.0 - s) + m_value * s;
          }
          idx++;
        }
      }
    }
  }
};

void demo_boundary_condition() {
  print_section("SCENARIO 2: Custom Boundary Condition - Dirichlet Fixed Value");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain and model
  auto world = world::create(Int3{64, 16, 16}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  // Initialize with constant value
  auto &field = model.get_real_field("density");
  std::fill(field.begin(), field.end(), 0.5);

  if (rank == 0) {
    std::cout << "Initial field: constant value 0.5\n";
    std::cout << "Applying Dirichlet BC: value = 1.0 at right boundary\n";
    std::cout << "Transition width: 5.0 grid units\n\n";
  }

  // Create and apply Dirichlet BC
  DirichletBC bc(1.0, 5.0);
  bc.set_field_name("density");
  bc.apply(model, 0.0); // Time is irrelevant for this BC

  // Sample field values along x-axis
  auto inbox = fft::get_inbox(model.get_fft());
  double dx = world::get_spacing(world, 0);
  double x0 = world::get_origin(world, 0);

  if (rank == 0) {
    std::cout << "Field values after BC application:\n";
    std::cout << "x-position\tfield value\n";
  }

  // Each rank prints values in its subdomain
  int idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
        double x = x0 + i * dx;
        // Print every 8th point to avoid clutter
        if (i % 8 == 0 && j == inbox.low[1] && k == inbox.low[2]) {
          printf("[Rank %d] x=%.2f\tfield=%.4f\n", rank, x, field[idx]);
        }
        idx++;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 3: Space-Time Varying Boundary Condition
//==============================================================================

/**
 * @brief Space-time varying BC: Oscillating left boundary
 *
 * This demonstrates:
 * - Time-dependent boundary conditions
 * - Using the time parameter in apply()
 * - Periodic forcing at boundaries
 * - Boundary conditions as "driving forces"
 */
class OscillatingBC : public FieldModifier {
private:
  double m_frequency; // Oscillation frequency
  double m_amplitude; // Oscillation amplitude
  double m_mean;      // Mean value
  std::string m_name = "OscillatingBC";

public:
  OscillatingBC(double frequency, double amplitude, double mean = 0.0)
      : m_frequency(frequency), m_amplitude(amplitude), m_mean(mean) {}

  const std::string &get_modifier_name() const override { return m_name; }

  void apply(Model &model, double time) override {
    auto &field = model.get_real_field(get_field_name());
    const auto &fft = model.get_fft();
    auto inbox = fft::get_inbox(fft);

    // Time-varying amplitude (sinusoidal)
    double bc_value =
        m_mean + m_amplitude * std::sin(2.0 * M_PI * m_frequency * time);

    // Apply at left boundary (i=0)
    int idx = 0;
    for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
      for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
        for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
          if (i == 0) {
            field[idx] = bc_value;
          }
          idx++;
        }
      }
    }
  }
};

void demo_space_time_bc() {
  print_section("SCENARIO 3: Space-Time Varying Boundary Condition");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain and model
  auto world = world::create(Int3{32, 16, 16}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  // Initialize field
  auto &field = model.get_real_field("density");
  std::fill(field.begin(), field.end(), 0.5);

  // Create oscillating BC
  OscillatingBC bc(0.5, // frequency = 0.5 Hz (period = 2.0 time units)
                   0.3, // amplitude
                   0.5  // mean value
  );
  bc.set_field_name("density");

  if (rank == 0) {
    std::cout << "Oscillating BC at left boundary (i=0)\n";
    std::cout << "Frequency: 0.5 Hz, Amplitude: 0.3, Mean: 0.5\n";
    std::cout << "BC formula: 0.5 + 0.3*sin(2π*0.5*t)\n\n";
    std::cout << "Time\tBC Value\n";
  }

  // Apply BC at different times
  std::vector<double> times = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};
  for (double t : times) {
    bc.apply(model, t);

    // Check value at i=0 (if this rank owns it)
    auto inbox = fft::get_inbox(model.get_fft());
    if (inbox.low[0] == 0) {
      double bc_val = field[0]; // First point in local subdomain
      if (rank == 0) {
        printf("%.2f\t%.4f\n", t, bc_val);
      }
    }
  }

  if (rank == 0) {
    std::cout << "\nNote: BC value oscillates between 0.2 and 0.8\n";
    std::cout << "      (mean ± amplitude)\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 4: Composing Multiple Modifiers
//==============================================================================

/**
 * @brief Demonstrate composition of multiple field modifiers
 *
 * This shows how to combine:
 * - Constant background IC
 * - Localized perturbation IC
 * - Boundary conditions
 *
 * Modifiers are applied in sequence, each building on the previous.
 */
void demo_composition() {
  print_section("SCENARIO 4: Composing Multiple Modifiers");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain and model
  auto world = world::create(Int3{64, 32, 32}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "Building complex initial state via composition:\n\n";
  }

  // Step 1: Set uniform background using built-in Constant IC
  if (rank == 0) {
    std::cout << "1. Apply constant background: 0.5\n";
  }
  Constant background(0.5);
  background.set_field_name("density");
  background.apply(model, 0.0);

  // Step 2: Add Gaussian perturbation
  if (rank == 0) {
    std::cout << "2. Add Gaussian perturbation at (32, 16, 16)\n";
    std::cout << "   Amplitude: 0.2, Width: 8.0\n";
  }
  GaussianIC perturbation(Real3{32.0, 16.0, 16.0}, 0.2, 8.0, 0.0);
  perturbation.set_field_name("density");
  perturbation.apply(model, 0.0);

  // Step 3: Add second perturbation at different location
  if (rank == 0) {
    std::cout << "3. Add second Gaussian at (48, 16, 16)\n";
    std::cout << "   Amplitude: -0.15, Width: 6.0\n";
  }
  GaussianIC perturbation2(Real3{48.0, 16.0, 16.0}, -0.15, 6.0, 0.0);
  perturbation2.set_field_name("density");
  perturbation2.apply(model, 0.0);

  // Step 4: Apply boundary conditions
  if (rank == 0) {
    std::cout << "4. Apply fixed BC at right boundary: value = 0.3\n";
  }
  DirichletBC bc_right(0.3, 5.0);
  bc_right.set_field_name("density");
  bc_right.apply(model, 0.0);

  // Analyze resulting field
  const auto &field = model.get_real_field("density");
  double min_val = *std::min_element(field.begin(), field.end());
  double max_val = *std::max_element(field.begin(), field.end());
  double sum = std::accumulate(field.begin(), field.end(), 0.0);

  // MPI reductions to get global statistics
  double global_min, global_max, global_sum;
  MPI_Reduce(&min_val, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max_val, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int total_points = world::get_size(world, 0) * world::get_size(world, 1) *
                       world::get_size(world, 2);
    double mean = global_sum / total_points;

    std::cout << "\nResulting field statistics:\n";
    std::cout << "  Min value: " << global_min << "\n";
    std::cout << "  Max value: " << global_max << "\n";
    std::cout << "  Mean value: " << mean << "\n";
    std::cout
        << "\nComposition allows building complex states from simple modifiers!\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// Main: Run all scenarios
//==============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "\n";
    std::cout << "╔═════════════════════════════════════════════════════════════════"
                 "═══╗\n";
    std::cout << "║          OpenPFC FieldModifier API Examples                     "
                 "   ║\n";
    std::cout << "║                                                                 "
                 "   ║\n";
    std::cout << "║  Demonstrates custom initial and boundary conditions            "
                 "   ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════════"
                 "═══╝\n";
  }

  try {
    demo_custom_initial_condition();
    demo_boundary_condition();
    demo_space_time_bc();
    demo_composition();

    if (rank == 0) {
      std::cout << "\n";
      std::cout << "╔═══════════════════════════════════════════════════════════════"
                   "═════╗\n";
      std::cout << "║  Key Takeaways:                                               "
                   "     ║\n";
      std::cout << "║                                                               "
                   "     ║\n";
      std::cout << "║  1. Derive from FieldModifier and implement apply()           "
                   "     ║\n";
      std::cout << "║  2. Access model geometry via get_world(), get_fft()          "
                   "     ║\n";
      std::cout << "║  3. Use time parameter for time-varying BCs                   "
                   "     ║\n";
      std::cout << "║  4. Compose modifiers for complex initial states              "
                   "     ║\n";
      std::cout << "║  5. Each rank operates on its local subdomain automatically   "
                   "     ║\n";
      std::cout << "║                                                               "
                   "     ║\n";
      std::cout << "║  See include/openpfc/initial_conditions/ for built-in ICs     "
                   "     ║\n";
      std::cout << "║  See include/openpfc/boundary_conditions/ for built-in BCs    "
                   "     ║\n";
      std::cout << "╚═══════════════════════════════════════════════════════════════"
                   "═════╝\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
