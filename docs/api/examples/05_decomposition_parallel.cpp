// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 05_decomposition_parallel.cpp
 * @brief Comprehensive demonstration of OpenPFC Decomposition API
 *
 * This example showcases domain decomposition for distributed-memory parallelism:
 * - Manual grid specification (explicit layout)
 * - Automatic grid selection (optimal communication)
 * - Rank-local vs global coordinate mapping
 * - Subdomain queries and properties
 * - Integration with MPI parallel execution
 *
 * The Decomposition class partitions a global World into non-overlapping
 * subdomains, each owned by one MPI rank. This enables scalable parallel
 * simulations on HPC clusters.
 *
 * **Requirements**: MPI library (must be run with mpirun/mpiexec)
 *
 * **Usage**:
 * ```bash
 * # Compile (assuming build system configured)
 * cd build && make 05_decomposition_parallel
 *
 * # Run with 4 MPI ranks
 * mpirun -np 4 ./05_decomposition_parallel
 * ```
 */

#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <sstream>

using namespace pfc;

// Helper: Get rank-specific output prefix
std::string rank_prefix(int rank) {
  std::ostringstream oss;
  oss << "[Rank " << rank << "] ";
  return oss.str();
}

// Helper: Synchronized printing (avoid interleaved output)
void print_sync(int rank, int size, const std::string &message) {
  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      std::cout << message << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

// ============================================================================
// Scenario 1: Manual Grid Specification
// ============================================================================

void scenario_manual_grid() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 1: Manual Grid Specification ===\n\n";
    std::cout << "Total MPI ranks: " << size << "\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create global domain
  auto world = world::create({128, 128, 128}, {1.0, 1.0, 1.0});

  if (rank == 0) {
    auto global_size = world::get_size(world);
    std::cout << "Global domain: [" << global_size[0] << ", " << global_size[1]
              << ", " << global_size[2] << "]\n";
    std::cout << "Physical volume: " << world::physical_volume(world)
              << " units³\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create decomposition with 2x2x1 grid (suitable for 4 ranks)
  auto decomp = decomposition::create(world, {2, 2, 1});

  if (rank == 0) {
    auto grid = decomposition::get_grid(decomp);
    int num_domains = decomposition::get_num_domains(decomp);
    std::cout << "Decomposition grid: [" << grid[0] << ", " << grid[1] << ", "
              << grid[2] << "]\n";
    std::cout << "Total subdomains: " << num_domains << "\n\n";

    if (num_domains != size) {
      std::cout << "⚠️  Warning: Number of subdomains (" << num_domains
                << ") != MPI size (" << size << ")\n";
      std::cout << "    Some ranks will be idle or domain assignment may be "
                   "incorrect.\n\n";
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Each rank queries its local subdomain
  if (rank < decomposition::get_num_domains(decomp)) {
    auto local_world = decomposition::get_subworld(decomp, rank);
    auto local_size = world::get_size(local_world);
    auto local_origin = world::get_origin(local_world);
    auto local_bounds = world::get_upper_bounds(local_world);

    std::ostringstream oss;
    oss << rank_prefix(rank) << "Local subdomain:\n";
    oss << rank_prefix(rank) << "  Size: [" << local_size[0] << ", " << local_size[1]
        << ", " << local_size[2] << "]\n";
    oss << rank_prefix(rank) << "  Origin: [" << std::fixed << std::setprecision(2)
        << local_origin[0] << ", " << local_origin[1] << ", " << local_origin[2]
        << "]\n";
    oss << rank_prefix(rank) << "  Upper bound: [" << local_bounds[0] << ", "
        << local_bounds[1] << ", " << local_bounds[2] << "]\n\n";

    print_sync(rank, size, oss.str());
  } else {
    std::ostringstream oss;
    oss << rank_prefix(rank) << "Idle (no subdomain assigned)\n\n";
    print_sync(rank, size, oss.str());
  }
}

// ============================================================================
// Scenario 2: Automatic Grid Selection
// ============================================================================

void scenario_automatic_grid() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 2: Automatic Grid Selection ===\n\n";
    std::cout << "MPI size: " << size << "\n";
    std::cout << "Using proc_setup_min_surface for optimal grid selection\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create global domain
  auto world = world::create({256, 256, 256}, {0.5, 0.5, 0.5});

  // Let algorithm choose optimal grid pattern
  auto decomp = decomposition::create(world, size);

  if (rank == 0) {
    auto grid = decomposition::get_grid(decomp);
    int num_domains = decomposition::get_num_domains(decomp);

    std::cout << "Auto-selected grid: [" << grid[0] << ", " << grid[1] << ", "
              << grid[2] << "]\n";
    std::cout << "Total subdomains: " << num_domains << "\n";
    std::cout << "Product: " << grid[0] << " × " << grid[1] << " × " << grid[2]
              << " = " << (grid[0] * grid[1] * grid[2]) << "\n\n";

    // Show subdomain sizes
    auto subworlds = decomposition::get_subworlds(decomp);
    std::cout << "Subdomain sizes:\n";
    for (int i = 0; i < std::min(4, num_domains); ++i) {
      auto sz = world::get_size(subworlds[i]);
      std::cout << "  Rank " << i << ": [" << sz[0] << ", " << sz[1] << ", " << sz[2]
                << "]\n";
    }
    if (num_domains > 4) {
      std::cout << "  ... (total " << num_domains << " subdomains)\n";
    }
    std::cout << "\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Each rank gets its local world
  auto local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_vol = world::physical_volume(local_world);

  std::ostringstream oss;
  oss << rank_prefix(rank) << "Size: [" << local_size[0] << ", " << local_size[1]
      << ", " << local_size[2] << "], ";
  oss << "Volume: " << std::fixed << std::setprecision(2) << local_vol
      << " units³\n";

  print_sync(rank, size, oss.str());

  if (rank == 0) {
    std::cout << "\n";
  }
}

// ============================================================================
// Scenario 3: Global vs Local Coordinate Mapping
// ============================================================================

void scenario_coordinate_mapping() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 3: Global vs Local Coordinate Mapping ===\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create decomposition
  auto global_world = world::create({100, 100, 100}, {1.0, 1.0, 1.0});
  auto decomp = decomposition::create(global_world, size);
  auto local_world = decomposition::get_subworld(decomp, rank);

  // Compute center point coordinates
  auto local_size = world::get_size(local_world);
  Int3 local_center = {local_size[0] / 2, local_size[1] / 2, local_size[2] / 2};

  // Local coordinates (within subdomain)
  auto local_coords = world::to_coords(local_world, local_center);

  // Global coordinates (within full domain)
  auto global_origin = world::get_origin(local_world);
  Real3 global_coords = {global_origin[0] + local_coords[0],
                         global_origin[1] + local_coords[1],
                         global_origin[2] + local_coords[2]};

  std::ostringstream oss;
  oss << rank_prefix(rank) << "Center point:\n";
  oss << rank_prefix(rank) << "  Local indices: [" << local_center[0] << ", "
      << local_center[1] << ", " << local_center[2] << "]\n";
  oss << rank_prefix(rank) << "  Local coords: [" << std::fixed
      << std::setprecision(2) << local_coords[0] << ", " << local_coords[1] << ", "
      << local_coords[2] << "]\n";
  oss << rank_prefix(rank) << "  Global coords: [" << global_coords[0] << ", "
      << global_coords[1] << ", " << global_coords[2] << "]\n\n";

  print_sync(rank, size, oss.str());
}

// ============================================================================
// Scenario 4: Decomposition Properties and Queries
// ============================================================================

void scenario_properties() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 4: Decomposition Properties and Queries ===\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create various decompositions
  auto world = world::create({200, 200, 100}, {0.5, 0.5, 1.0});

  if (rank == 0) {
    std::cout << "Global domain: 200×200×100, spacing: [0.5, 0.5, 1.0]\n\n";

    // Test different grid patterns
    std::vector<Int3> test_grids = {
        {1, 1, size},     // Slab decomposition (Z-direction only)
        {2, 2, size / 4}, // Block decomposition (if size=4)
        {size, 1, 1}      // Pencil decomposition (X-direction only)
    };

    for (size_t i = 0; i < test_grids.size(); ++i) {
      auto grid = test_grids[i];
      if (grid[0] * grid[1] * grid[2] != size) continue; // Skip invalid

      auto decomp = decomposition::create(world, grid);
      auto num = decomposition::get_num_domains(decomp);

      std::cout << "Grid [" << grid[0] << ", " << grid[1] << ", " << grid[2]
                << "]:\n";
      std::cout << "  Total domains: " << num << "\n";

      // Show first subdomain as example
      auto subworld_0 = decomposition::get_subworld(decomp, 0);
      auto sz = world::get_size(subworld_0);
      std::cout << "  Example subdomain (rank 0): [" << sz[0] << ", " << sz[1]
                << ", " << sz[2] << "]\n\n";
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Each rank queries its configuration
  auto decomp = decomposition::create(world, size);
  auto local = decomposition::get_subworld(decomp, rank);
  auto grid = decomposition::get_grid(decomp);

  auto local_size = world::get_size(local);
  auto local_spacing = world::get_spacing(local);
  int local_points = local_size[0] * local_size[1] * local_size[2];

  std::ostringstream oss;
  oss << rank_prefix(rank) << "Configuration:\n";
  oss << rank_prefix(rank) << "  Grid position: Rank " << rank << " of "
      << (grid[0] * grid[1] * grid[2]) << "\n";
  oss << rank_prefix(rank) << "  Local size: [" << local_size[0] << ", "
      << local_size[1] << ", " << local_size[2] << "]\n";
  oss << rank_prefix(rank) << "  Local spacing: [" << std::fixed
      << std::setprecision(2) << local_spacing[0] << ", " << local_spacing[1] << ", "
      << local_spacing[2] << "]\n";
  oss << rank_prefix(rank) << "  Grid points: " << local_points << "\n\n";

  print_sync(rank, size, oss.str());
}

// ============================================================================
// Scenario 5: Load Balance Analysis
// ============================================================================

void scenario_load_balance() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 5: Load Balance Analysis ===\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create decomposition
  auto world = world::create({256, 256, 256}, {1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, size);
  auto local = decomposition::get_subworld(decomp, rank);

  // Count local grid points
  auto local_size = world::get_size(local);
  int local_points = local_size[0] * local_size[1] * local_size[2];

  // Gather all point counts to rank 0
  std::vector<int> all_points(size);
  MPI_Gather(&local_points, 1, MPI_INT, all_points.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    // Compute statistics
    int min_points = *std::min_element(all_points.begin(), all_points.end());
    int max_points = *std::max_element(all_points.begin(), all_points.end());
    double avg_points = 0.0;
    for (int p : all_points) avg_points += p;
    avg_points /= size;

    double imbalance = (max_points - min_points) / avg_points * 100.0;

    std::cout << "Load Balance Statistics:\n";
    std::cout << "  Min points per rank: " << min_points << "\n";
    std::cout << "  Max points per rank: " << max_points << "\n";
    std::cout << "  Avg points per rank: " << std::fixed << std::setprecision(1)
              << avg_points << "\n";
    std::cout << "  Imbalance: " << std::setprecision(2) << imbalance << "%\n\n";

    if (imbalance < 1.0) {
      std::cout << "✓ Excellent load balance\n";
    } else if (imbalance < 5.0) {
      std::cout << "✓ Good load balance\n";
    } else {
      std::cout << "⚠ Significant load imbalance detected\n";
    }
    std::cout << "\n";
  }
}

// ============================================================================
// Scenario 6: Different Domain Aspect Ratios
// ============================================================================

void scenario_aspect_ratios() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 6: Domain Aspect Ratios ===\n\n";
    std::cout
        << "Comparing automatic grid selection for different domain shapes\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  struct TestCase {
    Int3 size;
    const char *description;
  };

  std::vector<TestCase> cases = {{{256, 256, 256}, "Cube (1:1:1)"},
                                 {{512, 256, 256}, "Elongated X (2:1:1)"},
                                 {{256, 256, 128}, "Flat slab (2:2:1)"},
                                 {{1024, 256, 64}, "Thin film (16:4:1)"}};

  for (const auto &test : cases) {
    auto world = world::create(test.size, {1.0, 1.0, 1.0});
    auto decomp = decomposition::create(world, size);
    auto grid = decomposition::get_grid(decomp);

    if (rank == 0) {
      std::cout << test.description << ":\n";
      std::cout << "  Domain: [" << test.size[0] << ", " << test.size[1] << ", "
                << test.size[2] << "]\n";
      std::cout << "  Selected grid: [" << grid[0] << ", " << grid[1] << ", "
                << grid[2] << "]\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::cout << "Note: Algorithm adapts grid to domain aspect ratio\n";
    std::cout << "      to minimize communication surface area.\n\n";
  }
}

// ============================================================================
// Main: Run All Scenarios
// ============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    if (rank == 0) {
      std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
      std::cout << "║  OpenPFC Decomposition API - MPI Parallel Demonstration  ║\n";
      std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    }

    scenario_manual_grid();
    scenario_automatic_grid();
    scenario_coordinate_mapping();
    scenario_properties();
    scenario_load_balance();
    scenario_aspect_ratios();

    if (rank == 0) {
      std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
      std::cout << "║  All scenarios completed successfully!                   ║\n";
      std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    }

    MPI_Finalize();
    return 0;

  } catch (const std::exception &e) {
    std::cerr << rank_prefix(rank) << "❌ Error: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
