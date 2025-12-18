// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_scalability.cpp
 * @brief Scalability study application for Tungsten model
 *
 * Runs different model sizes with different configurations (CPU/CUDA, precision, MPI
 * ranks) and stores performance metrics to CSV for analysis.
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <vector>

#include "tungsten_model.hpp"
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include "tungsten_cuda_model.hpp"
#include <openpfc/fft_cuda.hpp>
#endif

using namespace pfc;

struct ScalabilityResult {
  std::string backend;   // "CPU" or "CUDA"
  std::string precision; // "float" or "double"
  int size_x, size_y, size_z;
  int mpi_ranks;
  int num_iterations;
  double setup_time;         // seconds (allocation + prepare_operators)
  double total_time;         // seconds (total iteration time)
  double fft_time;           // seconds (total FFT time across all iterations)
  double other_time;         // seconds (total non-FFT time across all iterations)
  double time_per_iteration; // seconds (average per iteration)
  double fft_time_per_iteration;   // seconds (average FFT time per iteration)
  double other_time_per_iteration; // seconds (average non-FFT time per iteration)
  double memory_used;              // MB (approximate)
};

class ScalabilityStudy {
private:
  std::vector<ScalabilityResult> results;
  std::string output_file;

public:
  ScalabilityStudy(const std::string &csv_file) : output_file(csv_file) {
    // Write CSV header
    std::ofstream out(output_file);
    out << "backend,precision,size_x,size_y,size_z,mpi_ranks,iterations,"
        << "setup_time_sec,total_time_sec,fft_time_sec,other_time_sec,"
        << "time_per_iteration_sec,fft_time_per_iteration_sec,other_time_per_"
           "iteration_sec,memory_mb\n";
    out.close();
  }

  void run_cpu_test(int size_x, int size_y, int size_z, int num_iterations) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
      std::cout << "Running CPU test: size=(" << size_x << "," << size_y << ","
                << size_z << "), ranks=" << size << ", iterations=" << num_iterations
                << std::endl;
    }

    try {
      // Create world and decomposition
      auto world = world::create(GridSize({size_x, size_y, size_z}),
                                 PhysicalOrigin({0.0, 0.0, 0.0}),
                                 GridSpacing({1.0, 1.0, 1.0}));
      auto decomp = decomposition::create(world, size);
      auto fft = fft::create(decomp, rank);

      // Create model
      Tungsten model(fft, world);
      model.params.set_n0(-0.4);
      model.params.set_T(0.5);

      // Measure setup time (allocation + prepare_operators)
      MPI_Barrier(MPI_COMM_WORLD);
      double setup_start = MPI_Wtime();
      double dt = 0.01;
      model.initialize(dt);
      MPI_Barrier(MPI_COMM_WORLD);
      double setup_time = MPI_Wtime() - setup_start;

      // Initialize field
      auto &psi = model.get_real_field("psi");
      for (size_t i = 0; i < psi.size(); ++i) {
        psi[i] = -0.4 + 0.1 * std::sin(2.0 * M_PI * i / psi.size());
      }

      // Warm-up run
      for (int i = 0; i < 3; ++i) {
        model.step(0.0);
      }

      // Reset FFT timing before iterations
      fft.reset_fft_time();

      // Synchronize before timing
      MPI_Barrier(MPI_COMM_WORLD);
      double start_time = MPI_Wtime();

      // Run iterations
      for (int i = 0; i < num_iterations; ++i) {
        model.step(0.0);
      }

      // Synchronize after timing
      MPI_Barrier(MPI_COMM_WORLD);
      double end_time = MPI_Wtime();
      double total_time = end_time - start_time;
      double fft_time = fft.get_fft_time();
      double other_time = total_time - fft_time;
      double time_per_iter = total_time / num_iterations;
      double fft_time_per_iter = fft_time / num_iterations;
      double other_time_per_iter = other_time / num_iterations;

      // Estimate memory (rough approximation)
      auto size_inbox = fft.size_inbox();
      auto size_outbox = fft.size_outbox();
      double memory_mb =
          (size_inbox * sizeof(double) * 3 +               // psi, psiMF, psiN
           size_outbox * sizeof(double) * 3 +              // operators
           size_outbox * sizeof(std::complex<double>) * 3) // FFT fields
          / (1024.0 * 1024.0);

      if (rank == 0) {
        ScalabilityResult result;
        result.backend = "CPU";
        result.precision = "double"; // CPU always uses double
        result.size_x = size_x;
        result.size_y = size_y;
        result.size_z = size_z;
        result.mpi_ranks = size;
        result.num_iterations = num_iterations;
        result.setup_time = setup_time;
        result.total_time = total_time;
        result.fft_time = fft_time;
        result.other_time = other_time;
        result.time_per_iteration = time_per_iter;
        result.fft_time_per_iteration = fft_time_per_iter;
        result.other_time_per_iteration = other_time_per_iter;
        result.memory_used = memory_mb;

        save_result(result);
        std::cout << "  CPU (double): " << time_per_iter * 1000 << " ms/iteration"
                  << " (FFT: " << fft_time_per_iter * 1000
                  << " ms, Other: " << other_time_per_iter * 1000
                  << " ms, Setup: " << setup_time * 1000 << " ms)" << std::endl;
      }
    } catch (const std::exception &e) {
      if (rank == 0) {
        std::cerr << "Error in CPU test: " << e.what() << std::endl;
      }
    }
  }

#if defined(OpenPFC_ENABLE_CUDA)
  template <typename RealType>
  void run_cuda_test_impl(int size_x, int size_y, int size_z, int num_iterations,
                          const std::string &precision_name) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
      std::cout << "Running CUDA test (" << precision_name << "): size=(" << size_x
                << "," << size_y << "," << size_z << "), ranks=" << size
                << ", iterations=" << num_iterations << std::endl;
    }

    try {
      // Create world and decomposition
      auto world = world::create(GridSize({size_x, size_y, size_z}),
                                 PhysicalOrigin({0.0, 0.0, 0.0}),
                                 GridSpacing({1.0, 1.0, 1.0}));
      auto decomp = decomposition::create(world, size);

      // Create model with specified precision
      TungstenCUDA<RealType> model(fft::create(decomp, rank), world);
      model.params.set_n0(-0.4);
      model.params.set_T(0.5);

      // Measure setup time (allocation + prepare_operators)
      MPI_Barrier(MPI_COMM_WORLD);
      double setup_start = MPI_Wtime();
      double dt = 0.01;
      model.initialize(dt);
      MPI_Barrier(MPI_COMM_WORLD);
      double setup_time = MPI_Wtime() - setup_start;

      // Initialize field
      auto &psi_gpu = model.get_psi();
      std::vector<RealType> psi_init(psi_gpu.size());
      for (size_t i = 0; i < psi_init.size(); ++i) {
        psi_init[i] =
            static_cast<RealType>(-0.4) +
            static_cast<RealType>(0.1) *
                std::sin(static_cast<RealType>(2.0 * M_PI * i / psi_init.size()));
      }
      psi_gpu.copy_from_host(psi_init);

      // Warm-up run
      for (int i = 0; i < 3; ++i) {
        model.step(0.0);
      }

      // Reset FFT timing before iterations
      auto &cuda_fft = model.get_cuda_fft();
      cuda_fft.reset_fft_time();

      // Synchronize before timing
      MPI_Barrier(MPI_COMM_WORLD);
      double start_time = MPI_Wtime();

      // Run iterations
      for (int i = 0; i < num_iterations; ++i) {
        model.step(0.0);
      }

      // Synchronize after timing
      MPI_Barrier(MPI_COMM_WORLD);
      double end_time = MPI_Wtime();
      double total_time = end_time - start_time;
      double fft_time = cuda_fft.get_fft_time();
      double other_time = total_time - fft_time;
      double time_per_iter = total_time / num_iterations;
      double fft_time_per_iter = fft_time / num_iterations;
      double other_time_per_iter = other_time / num_iterations;

      // Estimate memory (rough approximation)
      auto size_inbox = cuda_fft.size_inbox();
      auto size_outbox = cuda_fft.size_outbox();
      double memory_mb =
          (size_inbox * sizeof(RealType) * 3 +               // psi, psiMF, psiN
           size_outbox * sizeof(RealType) * 3 +              // operators
           size_outbox * sizeof(std::complex<RealType>) * 3) // FFT fields
          / (1024.0 * 1024.0);

      if (rank == 0) {
        ScalabilityResult result;
        result.backend = "CUDA";
        result.precision = precision_name;
        result.size_x = size_x;
        result.size_y = size_y;
        result.size_z = size_z;
        result.mpi_ranks = size;
        result.num_iterations = num_iterations;
        result.setup_time = setup_time;
        result.total_time = total_time;
        result.fft_time = fft_time;
        result.other_time = other_time;
        result.time_per_iteration = time_per_iter;
        result.fft_time_per_iteration = fft_time_per_iter;
        result.other_time_per_iteration = other_time_per_iter;
        result.memory_used = memory_mb;

        save_result(result);
        std::cout << "  CUDA (" << precision_name << "): " << time_per_iter * 1000
                  << " ms/iteration"
                  << " (FFT: " << fft_time_per_iter * 1000
                  << " ms, Other: " << other_time_per_iter * 1000
                  << " ms, Setup: " << setup_time * 1000 << " ms)" << std::endl;
      }
    } catch (const std::exception &e) {
      if (rank == 0) {
        std::cerr << "Error in CUDA test (" << precision_name << "): " << e.what()
                  << std::endl;
      }
    }
  }

  void run_cuda_test(int size_x, int size_y, int size_z, int num_iterations) {
    // Test both float and double precision
    run_cuda_test_impl<double>(size_x, size_y, size_z, num_iterations, "double");
    run_cuda_test_impl<float>(size_x, size_y, size_z, num_iterations, "float");
  }
#endif

private:
  void save_result(const ScalabilityResult &result) {
    std::ofstream out(output_file, std::ios::app);
    out << std::fixed << std::setprecision(6);
    out << result.backend << "," << result.precision << "," << result.size_x << ","
        << result.size_y << "," << result.size_z << "," << result.mpi_ranks << ","
        << result.num_iterations << "," << result.setup_time << ","
        << result.total_time << "," << result.fft_time << "," << result.other_time
        << "," << result.time_per_iteration << "," << result.fft_time_per_iteration
        << "," << result.other_time_per_iteration << "," << result.memory_used
        << "\n";
    out.close();
  }
};

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Default configuration
  std::string output_file = "tungsten_scalability_results.csv";
  int num_iterations = 30;

  // Parse command line arguments
  if (argc > 1) {
    output_file = argv[1];
  }
  if (argc > 2) {
    num_iterations = std::stoi(argv[2]);
  }

  if (rank == 0) {
    std::cout << "Tungsten Scalability Study" << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Iterations per test: " << num_iterations << std::endl;
    std::cout << "MPI ranks: " << std::flush;
  }

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    std::cout << size << std::endl << std::endl;
  }

  ScalabilityStudy study(output_file);

  // Determine scaling mode from environment or command line
  std::string scaling_mode = "strong"; // default
  if (argc > 3) {
    scaling_mode = argv[3];
  } else if (const char *env_mode = std::getenv("SCALING_MODE")) {
    scaling_mode = env_mode;
  }

  if (rank == 0) {
    std::cout << "Scaling mode: " << scaling_mode << std::endl;
  }

  if (scaling_mode == "gpu") {
    // GPU scaling: test with different numbers of GPUs (1 and 2)
    // Test different sizes to see GPU performance
    std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };

    if (rank == 0) {
      std::cout << "GPU scaling test with " << size << " MPI rank(s)" << std::endl;
    }

    for (const auto &[sx, sy, sz] : sizes) {
#if defined(OpenPFC_ENABLE_CUDA)
      study.run_cuda_test(sx, sy, sz, num_iterations);
#endif
      if (rank == 0) {
        std::cout << std::endl;
      }
    }
  } else if (scaling_mode == "cpu") {
    // CPU scaling: test with different numbers of CPU cores
    // Test different sizes to see CPU performance
    std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };

    if (rank == 0) {
      std::cout << "CPU scaling test with " << size << " MPI rank(s)" << std::endl;
    }

    for (const auto &[sx, sy, sz] : sizes) {
      study.run_cpu_test(sx, sy, sz, num_iterations);
      if (rank == 0) {
        std::cout << std::endl;
      }
    }
  } else if (scaling_mode == "strong") {
    // Strong scaling: fixed problem size, varying number of ranks
    // Test with different numbers of ranks for same problem size
    const int base_size = 256;
    std::vector<int> rank_counts = {1, 2, 4,
                                    8}; // Will be adjusted based on available ranks

    if (rank == 0) {
      std::cout << "Strong scaling: fixed size " << base_size << "^3" << std::endl;
    }

    // For now, just test with current number of ranks
    // In a real scenario, you'd submit multiple jobs with different rank counts
    study.run_cpu_test(base_size, base_size, base_size, num_iterations);
#if defined(OpenPFC_ENABLE_CUDA)
    study.run_cuda_test(base_size, base_size, base_size, num_iterations);
#endif
  } else if (scaling_mode == "weak") {
    // Weak scaling: problem size scales with number of ranks
    // Each rank gets roughly the same amount of work
    int size_per_rank = 64; // Base size per rank
    int total_size =
        size_per_rank * static_cast<int>(std::cbrt(size)); // Scale with cube root

    if (rank == 0) {
      std::cout << "Weak scaling: size per rank " << size_per_rank << ", total size "
                << total_size << "^3" << std::endl;
    }

    study.run_cpu_test(total_size, total_size, total_size, num_iterations);
#if defined(OpenPFC_ENABLE_CUDA)
    study.run_cuda_test(total_size, total_size, total_size, num_iterations);
#endif
  } else {
    // Default: test different model sizes (up to 1024^3 for large memory systems)
    std::vector<std::tuple<int, int, int>> sizes = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        // Also test some non-cubic sizes
        {128, 128, 64},
        {256, 256, 128},
        {512, 512, 256},
    };

    for (const auto &[sx, sy, sz] : sizes) {
      // Run CPU test
      study.run_cpu_test(sx, sy, sz, num_iterations);

#if defined(OpenPFC_ENABLE_CUDA)
      // Run CUDA test (tests both float and double)
      study.run_cuda_test(sx, sy, sz, num_iterations);
#endif

      if (rank == 0) {
        std::cout << std::endl;
      }
    }
  }

  if (rank == 0) {
    std::cout << "Scalability study complete. Results saved to: " << output_file
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}
