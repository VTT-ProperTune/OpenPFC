// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 06_results_writers.cpp
 * @brief Demonstration of OpenPFC ResultsWriter API for simulation output
 *
 * This example shows how to use ResultsWriter for various output scenarios:
 * - Binary format output (checkpointing)
 * - Multiple writers (full field + statistics)
 * - Custom writer implementation
 * - Parallel I/O with MPI
 *
 * ResultsWriter provides a unified interface for writing simulation data to
 * files, with built-in support for parallel MPI-IO and domain decomposition.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/results_writer.hpp>

using namespace pfc;

// Helper: Create test field data (sine wave)
RealField create_test_field(const World &world, double time) {
  auto size = world::get_size(world);
  auto spacing = world::get_spacing(world);
  auto origin = world::get_origin(world);

  int total = size[0] * size[1] * size[2];
  RealField field(total);

  for (int k = 0; k < size[2]; ++k) {
    for (int j = 0; j < size[1]; ++j) {
      for (int i = 0; i < size[0]; ++i) {
        int idx = i + size[0] * (j + size[1] * k);

        double x = origin[0] + i * spacing[0];
        double y = origin[1] + j * spacing[1];
        double z = origin[2] + k * spacing[2];

        // Sine wave pattern that evolves with time
        field[idx] =
            std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y) * std::sin(time);
      }
    }
  }

  return field;
}

// Custom statistics writer (writes to CSV on rank 0 only)
class StatsWriter {
private:
  std::string m_filename;
  std::ofstream m_file;

public:
  StatsWriter(const std::string &filename) : m_filename(filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      m_file.open(filename);
      m_file << "step,time,min,max,mean,std_dev\n";
    }
  }

  void write_statistics(int step, double time, const RealField &local_data) {
    // Compute local statistics
    double local_min = *std::min_element(local_data.begin(), local_data.end());
    double local_max = *std::max_element(local_data.begin(), local_data.end());
    double local_sum = std::accumulate(local_data.begin(), local_data.end(), 0.0);
    int local_count = local_data.size();

    // Global reduction
    double global_min, global_max, global_sum;
    int global_count;

    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      double mean = global_sum / global_count;

      // Simplified std dev calculation
      double std_dev = (global_max - global_min) / 4.0; // Approximation

      m_file << step << "," << std::fixed << std::setprecision(6) << time << ","
             << global_min << "," << global_max << "," << mean << "," << std_dev
             << "\n";
      m_file.flush();
    }
  }

  ~StatsWriter() {
    if (m_file.is_open()) {
      m_file.close();
    }
  }
};

// ============================================================================
// Scenario 1: Basic Binary Output
// ============================================================================

void scenario_basic_binary() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 1: Basic Binary Output ===\n\n";
  }

  // Create domain
  auto world = world::create({64, 64, 64}, {1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, size);
  auto local_world = decomposition::get_subworld(decomp, rank);

  // Setup writer
  auto writer = std::make_unique<BinaryWriter>("output/field_%04d.bin");

  auto global_size = world::get_size(world);
  auto local_size = world::get_size(local_world);
  auto local_origin = world::get_origin(local_world);
  auto spacing = world::get_spacing(world);

  // Compute offset in grid points
  std::array<int, 3> offset = {static_cast<int>(local_origin[0] / spacing[0]),
                               static_cast<int>(local_origin[1] / spacing[1]),
                               static_cast<int>(local_origin[2] / spacing[2])};

  writer->set_domain(global_size, local_size, offset);

  if (rank == 0) {
    std::cout << "Writing 5 time steps...\n";
  }

  // Write several time steps
  for (int step = 0; step < 5; ++step) {
    double time = step * 0.1;
    auto field = create_test_field(local_world, time);

    writer->write(step, field);

    if (rank == 0) {
      std::cout << "  Wrote step " << step << " (t=" << time << ")\n";
    }
  }

  if (rank == 0) {
    std::cout << "\nFiles created: field_0000.bin through field_0004.bin\n";
  }
}

// ============================================================================
// Scenario 2: Multiple Writers (Binary + Statistics)
// ============================================================================

void scenario_multiple_writers() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 2: Multiple Writers ===\n\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create domain
  auto world = world::create({128, 128, 128}, {1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, size);
  auto local_world = decomposition::get_subworld(decomp, rank);

  // Binary writer (full field, save every 10 steps)
  auto binary_writer = std::make_unique<BinaryWriter>("output/checkpoint_%04d.bin");

  auto global_size = world::get_size(world);
  auto local_size = world::get_size(local_world);
  auto local_origin = world::get_origin(local_world);
  auto spacing = world::get_spacing(world);

  std::array<int, 3> offset = {static_cast<int>(local_origin[0] / spacing[0]),
                               static_cast<int>(local_origin[1] / spacing[1]),
                               static_cast<int>(local_origin[2] / spacing[2])};

  binary_writer->set_domain(global_size, local_size, offset);

  // Statistics writer (CSV, every step)
  auto stats_writer = std::make_unique<StatsWriter>("output/statistics.csv");

  if (rank == 0) {
    std::cout << "Running simulation with dual output:\n";
    std::cout << "  - Full field: every 10 steps (binary)\n";
    std::cout << "  - Statistics: every step (CSV)\n\n";
  }

  // Simulate 50 steps
  for (int step = 0; step < 50; ++step) {
    double time = step * 0.01;
    auto field = create_test_field(local_world, time);

    // Always write statistics
    stats_writer->write_statistics(step, time, field);

    // Periodically save full field
    if (step % 10 == 0) {
      binary_writer->write(step, field);

      if (rank == 0) {
        std::cout << "  [" << step << "] Full field checkpoint saved\n";
      }
    }
  }

  if (rank == 0) {
    std::cout << "\nOutput files:\n";
    std::cout << "  - checkpoint_0000.bin, checkpoint_0010.bin, ..., "
                 "checkpoint_0040.bin\n";
    std::cout << "  - statistics.csv (50 rows)\n";
  }
}

// ============================================================================
// Scenario 3: Restart from Checkpoint
// ============================================================================

void scenario_checkpoint_restart() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== Scenario 3: Checkpoint & Restart Pattern ===\n\n";
    std::cout << "Demonstrating checkpoint workflow:\n";
  }

  // Create domain
  auto world = world::create({64, 64, 64}, {1.0, 1.0, 1.0});
  auto decomp = decomposition::create(world, size);
  auto local_world = decomposition::get_subworld(decomp, rank);

  // Setup checkpoint writer
  auto checkpoint = std::make_unique<BinaryWriter>("output/restart_%04d.bin");

  auto global_size = world::get_size(world);
  auto local_size = world::get_size(local_world);
  auto local_origin = world::get_origin(local_world);
  auto spacing = world::get_spacing(world);

  std::array<int, 3> offset = {static_cast<int>(local_origin[0] / spacing[0]),
                               static_cast<int>(local_origin[1] / spacing[1]),
                               static_cast<int>(local_origin[2] / spacing[2])};

  checkpoint->set_domain(global_size, local_size, offset);

  // Simulate with checkpointing
  const int checkpoint_interval = 100;
  const int total_steps = 250;

  if (rank == 0) {
    std::cout << "  1. Running initial simulation (0-250 steps)\n";
    std::cout << "     Checkpointing every " << checkpoint_interval << " steps\n";
  }

  for (int step = 0; step <= total_steps; ++step) {
    double time = step * 0.01;
    auto field = create_test_field(local_world, time);

    if (step % checkpoint_interval == 0) {
      checkpoint->write(step, field);

      if (rank == 0) {
        std::cout << "     [" << step << "] Checkpoint saved\n";
      }
    }
  }

  if (rank == 0) {
    std::cout << "\n  2. To restart from step 200:\n";
    std::cout << "     - Load restart_0200.bin using BinaryReader\n";
    std::cout << "     - Resume simulation from step 201\n";
    std::cout << "     - Metadata (time, step) stored separately\n\n";

    std::cout << "Files created:\n";
    std::cout << "  - restart_0000.bin (initial condition)\n";
    std::cout << "  - restart_0100.bin\n";
    std::cout << "  - restart_0200.bin\n";
  }
}

// ============================================================================
// Main: Run All Scenarios
// ============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  try {
    if (rank == 0) {
      std::cout << "╔═══════════════════════════════════════════════════╗\n";
      std::cout << "║  OpenPFC ResultsWriter API Demonstration        ║\n";
      std::cout << "╚═══════════════════════════════════════════════════╝\n";
    }

    scenario_basic_binary();
    MPI_Barrier(MPI_COMM_WORLD);

    scenario_multiple_writers();
    MPI_Barrier(MPI_COMM_WORLD);

    scenario_checkpoint_restart();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
      std::cout << "║  All scenarios completed successfully!           ║\n";
      std::cout << "╚═══════════════════════════════════════════════════╝\n";
    }

    MPI_Finalize();
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[Rank " << rank << "] ❌ Error: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
