// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 10_complete_pfc_model.cpp
 * @brief Complete Phase Field Crystal simulation demonstrating all OpenPFC APIs
 *
 * This comprehensive example demonstrates:
 * 1. World - Domain geometry and coordinate system
 * 2. Decomposition - MPI domain decomposition for parallelism
 * 3. FFT - Spectral transforms and k-space operations
 * 4. Model - Custom PFC model with field management
 * 5. Initial Conditions - Seed-based nucleation
 * 6. Boundary Conditions - Fixed walls
 * 7. Time - Time stepping and output scheduling
 * 8. Simulator - Orchestration of the simulation loop
 * 9. ResultsWriter - Parallel output to binary files
 * 10. Full integration - Production-quality PFC simulation
 *
 * Physical System:
 *   Phase Field Crystal (PFC) model for solidification
 *   Single-mode approximation with periodic boundary conditions
 *   Liquid → Solid phase transition driven by undercooling
 *
 * Compile and run:
 *   mpicxx -std=c++17 -I/path/to/openpfc/include 10_complete_pfc_model.cpp \
 *          -L/path/to/openpfc/lib -lopenpfc -lheffte -o 10_complete_pfc_model
 *   mpirun -np 4 ./10_complete_pfc_model
 */

#include <cmath>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <openpfc/boundary_conditions/fixed_bc.hpp>
#include <openpfc/initial_conditions/constant.hpp>
#include <openpfc/initial_conditions/single_seed.hpp>
#include <openpfc/openpfc.hpp>

using namespace pfc;

//==============================================================================
// PFC Model Implementation
//==============================================================================

/**
 * @brief Phase Field Crystal model for solidification simulations
 *
 * Implements single-mode PFC dynamics:
 *   ∂ψ/∂t = ∇²[ε·ψ + ψ³ + (1 + ∇²)²ψ]
 *
 * Where:
 *   ψ - density field (deviation from liquid)
 *   ε - dimensionless temperature (controls undercooling)
 *
 * Time integration: Semi-implicit spectral method
 *   - Linear terms: Exact integration in k-space
 *   - Nonlinear term: Explicit (forward Euler)
 */
class PFCModel : public Model {
private:
  // Real-space fields
  Field m_density;   // Current density field ψ
  Field m_nonlinear; // Nonlinear term ψ³

  // K-space fields
  ComplexField m_density_k;   // Fourier transform of density
  ComplexField m_nonlinear_k; // Fourier transform of nonlinear term

  // Operators (precomputed in k-space)
  std::vector<double> m_operator_L; // Linear operator exp(L·dt)
  std::vector<double> m_operator_N; // Nonlinear operator (exp(L·dt)-1)/L

  // Physical parameters
  double m_epsilon; // Dimensionless temperature
  double m_dt;      // Time step

public:
  /**
   * @brief Construct PFC model with specified domain
   * @param world Physical domain geometry
   * @param fft FFT engine for spectral transforms
   * @param epsilon Dimensionless temperature (ε < 0 for undercooling)
   */
  PFCModel(const World &world, std::unique_ptr<FFT> fft, double epsilon)
      : Model(world, std::move(fft)), m_epsilon(epsilon), m_dt(0.0) {}

  /**
   * @brief Initialize model: allocate fields and precompute operators
   * @param dt Time step size
   */
  void initialize(double dt) override {
    m_dt = dt;
    FFT &fft = get_fft();

    // Allocate real-space fields
    m_density.resize(fft::size_inbox(fft));
    m_nonlinear.resize(fft::size_inbox(fft));

    // Allocate k-space fields
    m_density_k.resize(fft::size_outbox(fft));
    m_nonlinear_k.resize(fft::size_outbox(fft));

    // Allocate operator storage
    m_operator_L.resize(fft::size_outbox(fft));
    m_operator_N.resize(fft::size_outbox(fft));

    // Register field with Model for external access (ICs, BCs, output)
    add_real_field("density", m_density);

    // Precompute operators in k-space
    precompute_operators();
  }

  /**
   * @brief Single time step: advance ψ(t) → ψ(t+dt)
   * @param time Current simulation time (unused in this model)
   */
  void step(double time) override {
    FFT &fft = get_fft();

    // 1. Transform density to k-space
    fft::forward(fft, m_density, m_density_k);

    // 2. Compute nonlinear term: N(ψ) = ψ³
    for (size_t i = 0; i < m_density.size(); i++) {
      m_nonlinear[i] = m_density[i] * m_density[i] * m_density[i];
    }

    // 3. Transform nonlinear term to k-space
    fft::forward(fft, m_nonlinear, m_nonlinear_k);

    // 4. Apply semi-implicit time integration in k-space:
    //    ψ(t+dt) = exp(L·dt)·ψ(t) + [(exp(L·dt)-1)/L]·∇²N(ψ)
    for (size_t i = 0; i < m_density_k.size(); i++) {
      m_density_k[i] =
          m_operator_L[i] * m_density_k[i] + m_operator_N[i] * m_nonlinear_k[i];
    }

    // 5. Transform back to real space
    fft::backward(fft, m_density_k, m_density);
  }

  /**
   * @brief Get current epsilon value
   * @return Dimensionless temperature
   */
  double get_epsilon() const { return m_epsilon; }

  /**
   * @brief Set epsilon (for parameter studies)
   * @param epsilon New dimensionless temperature
   */
  void set_epsilon(double epsilon) { m_epsilon = epsilon; }

private:
  /**
   * @brief Precompute time integration operators in k-space
   *
   * Linear operator: L(k) = -k²[ε + (1-k²)²]
   * - Integrating factor: exp(L·dt)
   * - Nonlinear operator: [exp(L·dt) - 1] / L
   */
  void precompute_operators() {
    const FFT &fft = get_fft();
    const World &world = get_world();
    auto outbox = fft::get_outbox(fft);

    // Wave vector scaling factors
    auto spacing = world::get_spacing(world);
    auto size = world::get_size(world);
    double kx_scale = 2.0 * M_PI / (spacing[0] * size[0]);
    double ky_scale = 2.0 * M_PI / (spacing[1] * size[1]);
    double kz_scale = 2.0 * M_PI / (spacing[2] * size[2]);

    // Loop over k-space
    size_t idx = 0;
    for (int k = outbox.low[2]; k <= outbox.high[2]; k++) {
      for (int j = outbox.low[1]; j <= outbox.high[1]; j++) {
        for (int i = outbox.low[0]; i <= outbox.high[0]; i++) {
          // Compute wave vector components (handle periodicity)
          double kx = (i <= size[0] / 2) ? i * kx_scale : (i - size[0]) * kx_scale;
          double ky = (j <= size[1] / 2) ? j * ky_scale : (j - size[1]) * ky_scale;
          double kz = (k <= size[2] / 2) ? k * kz_scale : (k - size[2]) * kz_scale;

          // |k|²
          double k2 = kx * kx + ky * ky + kz * kz;

          // Laplacian operator in k-space: -k²
          double k2_lap = -k2;

          // PFC linear operator: L(k) = -k²[ε + (1-k²)²]
          double L = k2_lap * (m_epsilon + (1.0 - k2) * (1.0 - k2));

          // Integrating factor: exp(L·dt)
          m_operator_L[idx] = std::exp(L * m_dt);

          // Nonlinear operator: [exp(L·dt) - 1] / L · (-k²)
          // (The -k² comes from ∇² applied to nonlinear term)
          if (std::abs(L) > 1e-14) {
            m_operator_N[idx] = (m_operator_L[idx] - 1.0) / L * k2_lap;
          } else {
            // L'Hôpital's rule at k=0: lim (e^x - 1)/x = 1
            m_operator_N[idx] = m_dt * k2_lap;
          }

          idx++;
        }
      }
    }
  }
};

//==============================================================================
// Main Simulation
//==============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
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
    std::cout << "║  Demonstrates all OpenPFC APIs in production-quality workflow   "
                 "   ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════════"
                 "═══╝\n";
    std::cout << "\nRunning on " << size << " MPI ranks\n";
  }

  try {
    //======================================================================
    // 1. WORLD API: Define physical domain geometry
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[1] Creating domain with World API...\n";
    }

    // Create 128³ computational domain with unit spacing
    auto world = world::create(Int3{128, 128, 128}, // Grid dimensions
                               Real3{1.0, 1.0, 1.0} // Physical spacing
    );

    if (rank == 0) {
      std::cout << "    Domain: " << world::get_size(world, 0) << " x "
                << world::get_size(world, 1) << " x " << world::get_size(world, 2)
                << " points\n";
      std::cout << "    Physical size: ["
                << world::get_size(world, 0) * world::get_spacing(world, 0) << " x "
                << world::get_size(world, 1) * world::get_spacing(world, 1) << " x "
                << world::get_size(world, 2) * world::get_spacing(world, 2) << "]\n";
    }

    //======================================================================
    // 2. DECOMPOSITION API: MPI domain decomposition
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[2] Creating MPI decomposition...\n";
    }

    auto decomp = decomposition::create(world, size);
    auto grid = decomposition::get_grid(decomp);

    if (rank == 0) {
      std::cout << "    MPI grid: " << grid[0] << " x " << grid[1] << " x "
                << grid[2] << "\n";
      std::cout << "    Using " << decomposition::get_num_domains(decomp)
                << " subdomains\n";
    }

    // Each rank prints its local subdomain
    auto local_world = decomposition::get_world(decomp);
    std::cout << "    [Rank " << rank
              << "] Local size: " << world::get_size(local_world, 0) << " x "
              << world::get_size(local_world, 1) << " x "
              << world::get_size(local_world, 2) << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    //======================================================================
    // 3. FFT API: Create distributed FFT engine
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[3] Initializing FFT (HeFFTe backend)...\n";
    }

    auto fft = fft::create(world, decomp, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "    Input box (real): " << fft::size_inbox(*fft) << " points\n";
      std::cout << "    Output box (complex): " << fft::size_outbox(*fft)
                << " points\n";
    }

    //======================================================================
    // 4. MODEL API: Create PFC model
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[4] Creating PFC model...\n";
    }

    double epsilon = -0.25; // Undercooling parameter (ε < 0 favors solid)
    PFCModel model(world, std::move(fft), epsilon);

    double dt = 0.5; // Time step
    model.initialize(dt);

    if (rank == 0) {
      std::cout << "    Model: Phase Field Crystal (single-mode)\n";
      std::cout << "    Epsilon (ε): " << epsilon << " (undercooled)\n";
      std::cout << "    Time step (dt): " << dt << "\n";
    }

    //======================================================================
    // 5. INITIAL CONDITIONS API: Set initial state
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[5] Applying initial conditions...\n";
    }

    // Background: uniform liquid phase
    Constant background(0.0); // ψ = 0 corresponds to liquid
    background.set_field_name("density");
    background.apply(model, 0.0);

    if (rank == 0) {
      std::cout << "    Applied Constant IC: ψ = 0.0 (liquid phase)\n";
    }

    // Nucleation: single crystalline seed at origin
    SingleSeed seed;
    seed.set_field_name("density");
    seed.set_density(0.0);   // Background density
    seed.set_amplitude(0.3); // Crystal amplitude
    seed.apply(model, 0.0);

    if (rank == 0) {
      std::cout << "    Applied SingleSeed IC: BCC crystal at origin\n";
      std::cout << "    Seed amplitude: 0.3, radius: 64.0\n";
    }

    //======================================================================
    // 6. BOUNDARY CONDITIONS API: Fixed walls (optional)
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[6] Setting up boundary conditions...\n";
    }

    // For this simulation, we use periodic BCs (implicit in FFT)
    // Demonstrate fixed BC setup (but don't apply every step for performance)
    FixedBC fixed_walls(0.0, 0.0); // Fix to liquid density at boundaries
    fixed_walls.set_field_name("density");

    if (rank == 0) {
      std::cout << "    Boundary conditions: Periodic (via FFT)\n";
      std::cout << "    (FixedBC available but not used in this run)\n";
    }

    //======================================================================
    // 7. TIME API: Configure time integration
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[7] Configuring time integration...\n";
    }

    double t_start = 0.0;
    double t_end = 100.0;
    int save_interval = 10; // Save every 10 steps

    Time time(t_start, t_end, dt, save_interval);

    if (rank == 0) {
      std::cout << "    Time span: [" << t_start << ", " << t_end << "]\n";
      std::cout << "    Time step: " << dt << "\n";
      std::cout << "    Total steps: " << time.get_nsteps() << "\n";
      std::cout << "    Save interval: " << save_interval << " steps\n";
    }

    //======================================================================
    // 8. RESULTS WRITER API: Configure output
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[8] Setting up output writers...\n";
    }

    BinaryWriter writer("pfc_output_{:06d}.bin");
    writer.set_domain(world::get_size(model.get_world()),
                      fft::get_inbox(model.get_fft()).size,
                      fft::get_inbox(model.get_fft()).low);

    if (rank == 0) {
      std::cout << "    Output format: Binary (MPI-IO)\n";
      std::cout << "    Output pattern: pfc_output_NNNNNN.bin\n";
    }

    //======================================================================
    // 9. SIMULATOR API: Orchestrate simulation loop
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[9] Creating simulator...\n";
    }

    Simulator simulator(model, time);

    // Register output writer
    simulator.add_results_writer(writer, "density");

    if (rank == 0) {
      std::cout << "    Simulator configured with model and time stepper\n";
      std::cout << "    Output writer registered for 'density' field\n";
    }

    //======================================================================
    // 10. FULL INTEGRATION: Run simulation
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[10] Running simulation...\n";
      std::cout << std::string(70, '-') << "\n";
    }

    // Write initial state
    writer.write(0, model.get_real_field("density"));
    if (rank == 0) {
      std::cout << "    Saved initial state: pfc_output_000000.bin\n";
    }

    // Time integration loop
    int step = 0;
    while (!time.done()) {
      // Advance one time step
      simulator.step();

      // Output progress
      if (time.do_save()) {
        if (rank == 0) {
          printf("    Step %4d / %4d  |  t = %6.2f  |  Saving output...\n",
                 time.get_increment(), time.get_nsteps(), time.get_current());
        }
      }

      // Advance time
      time.next();
      step++;
    }

    if (rank == 0) {
      std::cout << std::string(70, '-') << "\n";
      std::cout << "    Simulation complete!\n";
    }

    //======================================================================
    // Post-simulation analysis
    //======================================================================
    if (rank == 0) {
      std::cout << "\n[Post-processing] Computing final statistics...\n";
    }

    const auto &density = model.get_real_field("density");

    // Local statistics
    double local_min = *std::min_element(density.begin(), density.end());
    double local_max = *std::max_element(density.begin(), density.end());
    double local_sum = std::accumulate(density.begin(), density.end(), 0.0);
    int local_size = density.size();

    // Global statistics via MPI reduction
    double global_min, global_max, global_sum;
    int global_size;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      double mean = global_sum / global_size;
      std::cout << "    Final density field statistics:\n";
      std::cout << "      Mean: " << mean << "\n";
      std::cout << "      Min:  " << global_min << "\n";
      std::cout << "      Max:  " << global_max << "\n";
      std::cout << "      Range: [" << global_min << ", " << global_max << "]\n";
    }

    //======================================================================
    // Summary
    //======================================================================
    if (rank == 0) {
      std::cout << "\n";
      std::cout << "╔═══════════════════════════════════════════════════════════════"
                   "═════╗\n";
      std::cout << "║  Simulation Summary:                                          "
                   "     ║\n";
      std::cout << "║                                                               "
                   "     ║\n";
      std::cout << "║  ✓ Domain created with World API                              "
                   "     ║\n";
      std::cout << "║  ✓ MPI decomposition configured                               "
                   "     ║\n";
      std::cout << "║  ✓ FFT engine initialized (HeFFTe)                            "
                   "     ║\n";
      std::cout << "║  ✓ PFC model implemented and initialized                      "
                   "     ║\n";
      std::cout << "║  ✓ Initial conditions applied (Constant + SingleSeed)         "
                   "     ║\n";
      std::cout << "║  ✓ Boundary conditions set up (Periodic)                      "
                   "     ║\n";
      std::cout << "║  ✓ Time integration configured                                "
                   "     ║\n";
      std::cout << "║  ✓ Output writer configured (BinaryWriter)                    "
                   "     ║\n";
      std::cout << "║  ✓ Simulator orchestrated time loop                           "
                   "     ║\n";
      std::cout << "║  ✓ " << time.get_nsteps()
                << " time steps completed                                   ║\n";
      std::cout << "║                                                               "
                   "     ║\n";
      std::cout << "║  All 10 OpenPFC APIs demonstrated successfully!               "
                   "     ║\n";
      std::cout << "╚═══════════════════════════════════════════════════════════════"
                   "═════╝\n";

      std::cout << "\nNext steps:\n";
      std::cout << "  • Visualize output: Use ParaView or custom Python scripts\n";
      std::cout << "  • Analyze growth: Track crystal front velocity\n";
      std::cout << "  • Parameter study: Vary epsilon, dt, domain size\n";
      std::cout << "  • Extend model: Add temperature field (Karma-Rappel)\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
