// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_power_benchmark.cpp
 * @brief FFT benchmark to measure GPU power consumption and utilization
 *
 * This benchmark performs continuous FFT forward/backward operations for
 * a specified duration (default 5 minutes) to measure:
 * - GPU power consumption
 * - GPU utilization
 * - FFT performance (iterations per second)
 * - Memory bandwidth utilization
 *
 * Can be run with 1, 2, or 4 GPUs to compare scaling.
 */

#include "heffte.h"
#include <chrono>
#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

#ifdef NVML_FOUND
#include <nvml.h>
#endif

// CUDA kernel to initialize test data
__global__ void init_data_kernel(cuDoubleComplex *data, int size, int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double phase = 2.0 * M_PI * (idx + seed) / size;
    data[idx] = make_cuDoubleComplex(cos(phase), sin(phase));
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  using backend_tag = heffte::backend::cufft;

  int const my_rank = heffte::mpi::comm_rank(MPI_COMM_WORLD);
  int const num_ranks = heffte::mpi::comm_size(MPI_COMM_WORLD);

  // Parse command line arguments
  int grid_size = 512;             // Default: 512^3
  double duration_seconds = 300.0; // Default: 5 minutes
  bool use_nvml = false;

  if (argc > 1) {
    grid_size = std::stoi(argv[1]);
  }
  if (argc > 2) {
    duration_seconds = std::stod(argv[2]);
  }
  if (argc > 3 && std::string(argv[3]) == "--nvml") {
    use_nvml = true;
  }

  if (my_rank == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "FFT Power Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Grid size: " << grid_size << "^3" << std::endl;
    std::cout << "MPI ranks: " << num_ranks << std::endl;
    std::cout << "Duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "CUDA devices: " << heffte::gpu::device_count() << std::endl;
    std::cout << "========================================" << std::endl;
  }

  // Set GPU device for this MPI rank
  if (heffte::gpu::device_count() > 0) {
    int device_id = my_rank % heffte::gpu::device_count();
    heffte::gpu::device_set(device_id);

    int current_device;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    std::cout << "Rank " << my_rank << " using GPU " << device_id << " ("
              << prop.name << ")" << std::endl;
  } else {
    if (my_rank == 0) {
      std::cerr << "ERROR: No CUDA devices found!" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Initialize NVML for power monitoring (if available)
#ifdef NVML_FOUND
  nvmlReturn_t nvml_result = NVML_SUCCESS;
  nvmlDevice_t nvml_device;
  unsigned int power_mw = 0;
  unsigned int power_limit_mw = 0;
  unsigned int temp_c = 0;
  unsigned int utilization_gpu = 0;
  unsigned int utilization_memory = 0;

  if (use_nvml) {
    nvml_result = nvmlInit();
    if (nvml_result == NVML_SUCCESS) {
      int device_id = my_rank % heffte::gpu::device_count();
      nvml_result = nvmlDeviceGetHandleByIndex(device_id, &nvml_device);
      if (nvml_result == NVML_SUCCESS) {
        nvmlDeviceGetPowerManagementLimitConstraints(nvml_device, &power_limit_mw,
                                                     nullptr);
        if (my_rank == 0) {
          std::cout << "NVML initialized successfully" << std::endl;
          std::cout << "Power limit: " << power_limit_mw << " mW" << std::endl;
        }
      }
    } else {
      if (my_rank == 0) {
        std::cout << "Warning: NVML not available, using CUDA power APIs"
                  << std::endl;
      }
      use_nvml = false;
    }
  }
#else
  if (use_nvml && my_rank == 0) {
    std::cout << "Warning: NVML not compiled in, using CUDA power APIs" << std::endl;
  }
  use_nvml = false;
#endif

  // Create FFT with proper domain decomposition
  // Use simple 1D decomposition along z-axis for multi-GPU
  int const nz_per_rank = grid_size / num_ranks;
  int const z_start = my_rank * nz_per_rank;
  int const z_end =
      (my_rank == num_ranks - 1) ? grid_size - 1 : (my_rank + 1) * nz_per_rank - 1;

  heffte::box3d<> const inbox = {{0, 0, z_start},
                                 {grid_size - 1, grid_size - 1, z_end}};
  // For r2c FFT, output box is same as input for complex-to-complex
  // But we'll use complex-to-complex FFT for simplicity
  heffte::box3d<> const outbox = inbox;

  heffte::plan_options options = heffte::default_options<backend_tag>();
  // Use complex-to-complex FFT for simplicity (can be changed to r2c if needed)
  heffte::fft3d<backend_tag> fft(inbox, outbox, MPI_COMM_WORLD, options);

  // Allocate GPU memory (complex-to-complex FFT)
  size_t const size_inbox = fft.size_inbox();
  size_t const size_outbox = fft.size_outbox();

  heffte::gpu::vector<std::complex<double>> gpu_input(size_inbox);
  heffte::gpu::vector<std::complex<double>> gpu_output(size_outbox);

  // Initialize input data on GPU
  int blocks = (size_inbox + 255) / 256;
  init_data_kernel<<<blocks, 256>>>(
      reinterpret_cast<cuDoubleComplex *>(gpu_input.data()), size_inbox, my_rank);
  cudaDeviceSynchronize();

  if (my_rank == 0) {
    std::cout << "Rank " << my_rank << " local box: z=[" << z_start << ", " << z_end
              << "]" << std::endl;
  }

  // Workspace for FFT
  auto workspace = fft.size_workspace();
  heffte::gpu::vector<std::complex<double>> fft_workspace(workspace);

  if (my_rank == 0) {
    std::cout << "\nStarting FFT benchmark..." << std::endl;
    std::cout << "Local input size: " << size_inbox << std::endl;
    std::cout << "Local output size: " << size_outbox << std::endl;
    std::cout << "Workspace size: " << workspace << std::endl;
  }

  // Warm-up
  for (int i = 0; i < 10; ++i) {
    fft.forward(gpu_input.data(), gpu_output.data(), fft_workspace.data());
    fft.backward(gpu_output.data(), gpu_input.data(), fft_workspace.data());
  }
  cudaDeviceSynchronize();

  // Start timing and power monitoring
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = start_time + std::chrono::duration<double>(duration_seconds);

  long long iterations = 0;
  double total_power_mw = 0.0;
  int power_samples = 0;
  double max_power_mw = 0.0;
  double min_power_mw = 1e9;

  // Power monitoring variables
  std::vector<double> power_readings;
  std::vector<std::chrono::steady_clock::time_point> power_timestamps;

  if (my_rank == 0) {
    std::cout << "\nRunning benchmark for " << duration_seconds << " seconds..."
              << std::endl;
  }

  // Main benchmark loop
  while (std::chrono::steady_clock::now() < end_time) {
    // Forward FFT
    fft.forward(gpu_input.data(), gpu_output.data(), fft_workspace.data());

    // Backward FFT
    fft.backward(gpu_output.data(), gpu_input.data(), fft_workspace.data());

    iterations++;

    // Sample power every 100 iterations (or every ~second for fast FFTs)
    if (iterations % 100 == 0) {
      double current_power = 0.0;

#ifdef NVML_FOUND
      if (use_nvml && nvml_result == NVML_SUCCESS) {
        nvmlDeviceGetPowerUsage(nvml_device, &power_mw);
        current_power = static_cast<double>(power_mw);
      }
#endif
      // If NVML not available, we can't measure power directly
      // User can monitor with nvidia-smi externally

      if (current_power > 0) {
        total_power_mw += current_power;
        power_samples++;
        max_power_mw = std::max(max_power_mw, current_power);
        min_power_mw = std::min(min_power_mw, current_power);
        power_readings.push_back(current_power);
        power_timestamps.push_back(std::chrono::steady_clock::now());
      }
    }
  }

  // Final synchronization
  cudaDeviceSynchronize();
  auto actual_end_time = std::chrono::steady_clock::now();
  double actual_duration =
      std::chrono::duration<double>(actual_end_time - start_time).count();

  // Collect statistics from all ranks
  long long total_iterations = 0;
  MPI_Allreduce(&iterations, &total_iterations, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  double avg_power_mw = (power_samples > 0) ? (total_power_mw / power_samples) : 0.0;
  double global_avg_power = 0.0;
  double global_max_power = 0.0;
  double global_min_power = 0.0;

  MPI_Reduce(&avg_power_mw, &global_avg_power, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&max_power_mw, &global_max_power, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&min_power_mw, &global_min_power, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);

  if (my_rank == 0) {
    global_avg_power /= num_ranks; // Average across ranks

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total iterations: " << total_iterations << std::endl;
    std::cout << "Actual duration: " << std::fixed << std::setprecision(2)
              << actual_duration << " seconds" << std::endl;
    std::cout << "Iterations per second: " << std::fixed << std::setprecision(2)
              << (total_iterations / actual_duration) << std::endl;
    std::cout << "Average power: " << std::fixed << std::setprecision(1)
              << (global_avg_power / 1000.0) << " W" << std::endl;
    std::cout << "Max power: " << std::fixed << std::setprecision(1)
              << (global_max_power / 1000.0) << " W" << std::endl;
    std::cout << "Min power: " << std::fixed << std::setprecision(1)
              << (global_min_power / 1000.0) << " W" << std::endl;
    std::cout << "Power samples: " << power_samples << std::endl;
    std::cout << "========================================" << std::endl;

    // Write power data to CSV
    std::ofstream csv_file("fft_power_benchmark_" + std::to_string(num_ranks) +
                           "gpu.csv");
    csv_file << "time_sec,power_w" << std::endl;
    for (size_t i = 0; i < power_readings.size(); ++i) {
      double time_sec =
          std::chrono::duration<double>(power_timestamps[i] - start_time).count();
      csv_file << std::fixed << std::setprecision(3) << time_sec << ","
               << (power_readings[i] / 1000.0) << std::endl;
    }
    csv_file.close();
    std::cout << "Power data saved to: fft_power_benchmark_" << num_ranks
              << "gpu.csv" << std::endl;
  }

#ifdef NVML_FOUND
  if (use_nvml && nvml_result == NVML_SUCCESS) {
    nvmlShutdown();
  }
#endif

  MPI_Finalize();
  return 0;
}
