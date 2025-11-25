// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 09_initial_conditions.cpp
 * @brief Comprehensive examples of built-in Initial Condition classes
 *
 * This example demonstrates:
 * 1. Constant IC - uniform field initialization
 * 2. SingleSeed IC - single crystalline seed
 * 3. SeedGrid IC - regular grid of seeds
 * 4. RandomSeeds IC - random seed distribution
 * 5. FileReader IC - restart from checkpoint
 * 6. Composition patterns - combining multiple ICs
 *
 * Compile and run:
 *   mpicxx -std=c++17 -I/path/to/openpfc/include 09_initial_conditions.cpp \
 *          -L/path/to/openpfc/lib -lopenpfc -lheffte -o 09_initial_conditions
 *   mpirun -np 4 ./09_initial_conditions
 */

#include <iostream>
#include <memory>
#include <mpi.h>
#include <openpfc/initial_conditions/constant.hpp>
#include <openpfc/initial_conditions/file_reader.hpp>
#include <openpfc/initial_conditions/random_seeds.hpp>
#include <openpfc/initial_conditions/seed_grid.hpp>
#include <openpfc/initial_conditions/single_seed.hpp>
#include <openpfc/openpfc.hpp>
#include <openpfc/results_writer.hpp>

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

// Helper to compute field statistics
void print_field_stats(const Field &field, const std::string &name, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  double local_sum = 0.0;
  double local_min = field.empty() ? 0.0 : field[0];
  double local_max = field.empty() ? 0.0 : field[0];

  for (const auto &val : field) {
    local_sum += val;
    local_min = std::min(local_min, val);
    local_max = std::max(local_max, val);
  }

  double global_sum, global_min, global_max;
  int local_size = field.size();
  int global_size;

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, 0, comm);

  if (rank == 0) {
    double mean = global_sum / global_size;
    std::cout << name << " statistics:\n";
    std::cout << "  Mean:  " << mean << "\n";
    std::cout << "  Min:   " << global_min << "\n";
    std::cout << "  Max:   " << global_max << "\n";
    std::cout << "  Range: [" << global_min << ", " << global_max << "]\n";
  }
}

//==============================================================================
// SCENARIO 1: Constant IC - Uniform Field
//==============================================================================

void demo_constant_ic() {
  print_section("SCENARIO 1: Constant IC - Uniform Field");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain and model
  auto world = world::create(Int3{64, 64, 64}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "Constant IC sets uniform value throughout domain\n";
    std::cout << "Use case: Homogeneous background, equilibrium state\n\n";
  }

  // Create and apply Constant IC
  Constant ic(0.5); // Density = 0.5
  ic.set_field_name("density");
  ic.apply(model, 0.0);

  if (rank == 0) {
    std::cout << "Applied Constant IC with value = 0.5\n\n";
  }

  // Verify uniformity
  const auto &field = model.get_real_field("density");
  print_field_stats(field, "Field after Constant IC", MPI_COMM_WORLD);

  // Demonstrate setter methods
  if (rank == 0) {
    std::cout << "\nConstant IC API:\n";
    std::cout << "  Constant(value)           - Constructor\n";
    std::cout << "  set_density(value)        - Set value\n";
    std::cout << "  get_density()             - Get current value\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 2: SingleSeed IC - Single Crystalline Nucleus
//==============================================================================

void demo_single_seed() {
  print_section("SCENARIO 2: SingleSeed IC - Single Crystalline Nucleus");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create larger domain for crystal growth
  auto world = world::create(Int3{128, 128, 128}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "SingleSeed IC creates spherical crystalline seed at origin\n";
    std::cout << "Use case: Single crystal growth, dendrite solidification\n\n";
  }

  // First apply background
  Constant background(0.285); // Liquid phase density
  background.set_field_name("density");
  background.apply(model, 0.0);

  // Add crystalline seed
  SingleSeed seed;
  seed.set_field_name("density");
  seed.set_density(0.285);  // Base density
  seed.set_amplitude(0.15); // Crystal amplitude
  seed.apply(model, 0.0);

  if (rank == 0) {
    std::cout << "Configuration:\n";
    std::cout << "  Background density: 0.285 (liquid)\n";
    std::cout << "  Seed density: 0.285\n";
    std::cout << "  Seed amplitude: 0.15 (crystal modulation)\n";
    std::cout << "  Seed location: origin (0, 0, 0)\n";
    std::cout << "  Seed radius: 64.0 (hardcoded)\n\n";
  }

  const auto &field = model.get_real_field("density");
  print_field_stats(field, "Field with single seed", MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\nSingleSeed IC API:\n";
    std::cout << "  set_density(rho)         - Base density\n";
    std::cout << "  set_amplitude(amp)       - Crystal modulation amplitude\n";
    std::cout << "  get_density()            - Get base density\n";
    std::cout << "  get_amplitude()          - Get amplitude\n";
    std::cout << "\nNote: Seed is BCC crystal structure with 6 wave vectors\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 3: SeedGrid IC - Regular Grid of Seeds
//==============================================================================

void demo_seed_grid() {
  print_section("SCENARIO 3: SeedGrid IC - Regular Grid of Seeds");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain
  auto world = world::create(Int3{128, 128, 128}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "SeedGrid IC creates regular array of crystalline seeds\n";
    std::cout << "Use case: Polycrystalline microstructure, grain growth\n\n";
  }

  // Apply background
  Constant background(0.285);
  background.set_field_name("density");
  background.apply(model, 0.0);

  // Create seed grid
  SeedGrid grid;
  grid.set_field_name("density");
  grid.set_Nx(1);           // Seeds only in y-z plane (for visualization)
  grid.set_Ny(3);           // 3 seeds in y
  grid.set_Nz(3);           // 3 seeds in z (total: 1x3x3 = 9 seeds)
  grid.set_X0(0.0);         // x position
  grid.set_radius(15.0);    // Seed radius
  grid.set_density(0.285);  // Base density
  grid.set_amplitude(0.15); // Crystal amplitude
  grid.apply(model, 0.0);

  if (rank == 0) {
    std::cout << "Configuration:\n";
    std::cout << "  Grid: 1x3x3 = 9 seeds\n";
    std::cout << "  Seed radius: 15.0\n";
    std::cout << "  Base density: 0.285\n";
    std::cout << "  Amplitude: 0.15\n";
    std::cout << "  Seeds distributed evenly in y-z plane\n";
    std::cout << "  Random perturbation: ±20% of radius\n";
    std::cout
        << "  Random orientation: each seed has unique crystal orientation\n\n";
  }

  const auto &field = model.get_real_field("density");
  print_field_stats(field, "Field with seed grid", MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\nSeedGrid IC API:\n";
    std::cout << "  set_Nx(nx), set_Ny(ny), set_Nz(nz) - Grid dimensions\n";
    std::cout << "  set_X0(x0)                         - x position\n";
    std::cout << "  set_radius(r)                      - Seed radius\n";
    std::cout << "  set_density(rho)                   - Base density\n";
    std::cout << "  set_amplitude(amp)                 - Crystal amplitude\n";
    std::cout << "\nSeeds are randomly perturbed in position and orientation\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 4: RandomSeeds IC - Random Distribution
//==============================================================================

void demo_random_seeds() {
  print_section("SCENARIO 4: RandomSeeds IC - Random Distribution");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain (should match what RandomSeeds expects)
  auto world = world::create(Int3{256, 256, 256}, Real3{1.0, 1.0, 1.0});
  auto origin = Real3{-128.0, -128.0, -128.0}; // Centered at origin
  world = world::create(Int3{256, 256, 256}, Real3{1.0, 1.0, 1.0}, origin);

  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "RandomSeeds IC places seeds at random locations\n";
    std::cout << "Use case: Realistic polycrystalline microstructure\n";
    std::cout << "          Homogeneous nucleation simulations\n\n";
  }

  // Apply background
  Constant background(0.285);
  background.set_field_name("density");
  background.apply(model, 0.0);

  // Apply random seeds
  RandomSeeds seeds;
  seeds.set_field_name("density");
  seeds.set_density(0.285);
  seeds.set_amplitude(0.15);
  seeds.apply(model, 0.0);

  if (rank == 0) {
    std::cout << "Configuration:\n";
    std::cout << "  Number of seeds: 150 (hardcoded)\n";
    std::cout << "  Seed radius: 20.0 (hardcoded)\n";
    std::cout << "  Distribution: random in x ∈ [-128+r, -128+3r)\n";
    std::cout << "                uniform in y,z ∈ [-128, 128]\n";
    std::cout << "  Base density: 0.285\n";
    std::cout << "  Amplitude: 0.15\n";
    std::cout << "  RNG seed: 42 (reproducible)\n\n";
  }

  const auto &field = model.get_real_field("density");
  print_field_stats(field, "Field with random seeds", MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\nRandomSeeds IC API:\n";
    std::cout << "  set_density(rho)      - Base density\n";
    std::cout << "  set_amplitude(amp)    - Crystal amplitude\n";
    std::cout << "  get_density()         - Get base density\n";
    std::cout << "  get_amplitude()       - Get amplitude\n";
    std::cout << "\nNote: Number of seeds, radius, and distribution are hardcoded\n";
    std::cout << "      Consider making these configurable in future versions\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 5: FileReader IC - Restart from Checkpoint
//==============================================================================

void demo_file_reader() {
  print_section("SCENARIO 5: FileReader IC - Restart from Checkpoint");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain and model
  auto world = world::create(Int3{64, 64, 64}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "FileReader IC loads field from binary checkpoint file\n";
    std::cout << "Use case: Restart simulations, continue from saved state\n\n";
  }

  // Step 1: Create initial state and save it
  if (rank == 0) {
    std::cout << "Step 1: Creating initial state...\n";
  }

  Constant background(0.3);
  background.set_field_name("density");
  background.apply(model, 0.0);

  SingleSeed seed;
  seed.set_field_name("density");
  seed.set_density(0.3);
  seed.set_amplitude(0.1);
  seed.apply(model, 0.0);

  // Save to file
  std::string checkpoint_file = "checkpoint_test.bin";
  BinaryWriter writer(checkpoint_file);
  writer.set_domain(world::get_size(world), fft::get_inbox(model.get_fft()).size,
                    fft::get_inbox(model.get_fft()).low);
  writer.write(0, model.get_real_field("density"));

  if (rank == 0) {
    std::cout << "  Saved checkpoint to: " << checkpoint_file << "\n\n";
  }

  // Compute stats before clearing
  const auto &field_before = model.get_real_field("density");
  double sum_before = 0.0;
  for (const auto &val : field_before) sum_before += val;

  // Step 2: Clear field
  if (rank == 0) {
    std::cout << "Step 2: Clearing field (simulating restart)...\n";
  }
  auto &field = model.get_real_field("density");
  std::fill(field.begin(), field.end(), 0.0);

  if (rank == 0) {
    std::cout << "  Field cleared (all zeros)\n\n";
  }

  // Step 3: Restore from checkpoint
  if (rank == 0) {
    std::cout << "Step 3: Restoring from checkpoint...\n";
  }

  FileReader reader(checkpoint_file);
  reader.set_field_name("density");
  reader.apply(model, 0.0);

  if (rank == 0) {
    std::cout << "  Loaded checkpoint from: " << checkpoint_file << "\n\n";
  }

  // Verify restoration
  const auto &field_after = model.get_real_field("density");
  double sum_after = 0.0;
  for (const auto &val : field_after) sum_after += val;

  double global_sum_before, global_sum_after;
  MPI_Reduce(&sum_before, &global_sum_before, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&sum_after, &global_sum_after, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Verification:\n";
    std::cout << "  Sum before save:  " << global_sum_before << "\n";
    std::cout << "  Sum after load:   " << global_sum_after << "\n";
    std::cout << "  Match: "
              << (std::abs(global_sum_before - global_sum_after) < 1e-10 ? "YES ✓"
                                                                         : "NO ✗")
              << "\n\n";

    std::cout << "FileReader IC API:\n";
    std::cout << "  FileReader(filename)     - Constructor\n";
    std::cout << "  set_filename(filename)   - Set file to read\n";
    std::cout << "  get_filename()           - Get filename\n";
    std::cout << "\nNote: File format must match BinaryWriter output\n";
    std::cout << "      Domain size and decomposition must be consistent\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//==============================================================================
// SCENARIO 6: Composition - Combining Multiple ICs
//==============================================================================

void demo_composition() {
  print_section("SCENARIO 6: Composition - Combining Multiple ICs");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create domain
  auto world = world::create(Int3{128, 128, 128}, Real3{1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, 4);
  auto fft = fft::create(world, decomp, MPI_COMM_WORLD);
  Model model(world, std::move(fft));
  model.add_real_field("density");

  if (rank == 0) {
    std::cout << "Demonstrating composition of multiple initial conditions\n";
    std::cout << "Pattern: Background → Modification → Refinement\n\n";
  }

  // Step 1: Uniform background
  if (rank == 0) {
    std::cout << "Step 1: Apply uniform background (Constant IC)\n";
  }
  Constant background(0.285);
  background.set_field_name("density");
  background.apply(model, 0.0);
  print_field_stats(model.get_real_field("density"), "  After background",
                    MPI_COMM_WORLD);

  // Step 2: Add single seed at center
  if (rank == 0) {
    std::cout << "\nStep 2: Add central crystalline seed (SingleSeed IC)\n";
  }
  SingleSeed central_seed;
  central_seed.set_field_name("density");
  central_seed.set_density(0.285);
  central_seed.set_amplitude(0.15);
  central_seed.apply(model, 0.0);
  print_field_stats(model.get_real_field("density"), "  After central seed",
                    MPI_COMM_WORLD);

  // Step 3: Add smaller grid of seeds around it
  if (rank == 0) {
    std::cout << "\nStep 3: Add secondary seed grid (SeedGrid IC)\n";
    std::cout << "  (Note: This overwrites region where seeds are placed)\n";
  }
  SeedGrid secondary_grid;
  secondary_grid.set_field_name("density");
  secondary_grid.set_Nx(1);
  secondary_grid.set_Ny(2);
  secondary_grid.set_Nz(2);
  secondary_grid.set_X0(50.0); // Offset in x
  secondary_grid.set_radius(10.0);
  secondary_grid.set_density(0.285);
  secondary_grid.set_amplitude(0.12); // Slightly different amplitude
  secondary_grid.apply(model, 0.0);
  print_field_stats(model.get_real_field("density"), "  After secondary grid",
                    MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\nComposition principles:\n";
    std::cout << "  1. ICs are applied in order (order matters!)\n";
    std::cout << "  2. Later ICs can overwrite earlier ones\n";
    std::cout << "  3. Use Constant IC for background, then add features\n";
    std::cout << "  4. SingleSeed, SeedGrid only modify points inside seeds\n";
    std::cout << "  5. FileReader completely overwrites field\n\n";

    std::cout << "Common patterns:\n";
    std::cout << "  • Constant → SingleSeed:  Liquid + nucleation\n";
    std::cout << "  • Constant → SeedGrid:    Polycrystalline structure\n";
    std::cout << "  • FileReader alone:       Pure restart\n";
    std::cout
        << "  • Constant → FileReader:  Not recommended (FileReader overwrites)\n";
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
    std::cout << "║          OpenPFC Initial Conditions Examples                    "
                 "   ║\n";
    std::cout << "║                                                                 "
                 "   ║\n";
    std::cout << "║  Demonstrates all built-in initial condition classes            "
                 "   ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════════"
                 "═══╝\n";
  }

  try {
    demo_constant_ic();
    demo_single_seed();
    demo_seed_grid();
    demo_random_seeds();
    demo_file_reader();
    demo_composition();

    if (rank == 0) {
      std::cout << "\n";
      std::cout << "╔═══════════════════════════════════════════════════════════════"
                   "═════╗\n";
      std::cout << "║  Summary of Built-in Initial Conditions:                      "
                   "     ║\n";
      std::cout << "║                                                               "
                   "     ║\n";
      std::cout << "║  Constant      - Uniform field value                          "
                   "     ║\n";
      std::cout << "║  SingleSeed    - Single crystalline nucleus at origin         "
                   "    ║\n";
      std::cout << "║  SeedGrid      - Regular array of crystalline seeds           "
                   "     ║\n";
      std::cout << "║  RandomSeeds   - Random distribution of seeds                 "
                   "     ║\n";
      std::cout << "║  FileReader    - Load from binary checkpoint file             "
                   "     ║\n";
      std::cout << "║                                                               "
                   "     ║\n";
      std::cout << "║  All inherit from FieldModifier - can create custom ICs!      "
                   "     ║\n";
      std::cout << "║  See 07_field_modifiers.cpp for custom IC examples            "
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
