// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cufft_direct_benchmark.cpp
 * @brief Direct cuFFT benchmark to measure GPU power consumption
 *
 * This benchmark uses cuFFT directly (not HeFFTe) to perform 3D FFTs
 * on a single GPU. This helps isolate whether low power usage is due to
 * HeFFTe/MPI overhead or the FFT computation itself.
 *
 * Performs continuous FFT forward/backward operations for a specified
 * duration to measure:
 * - GPU power consumption
 * - FFT performance (iterations per second)
 * - Memory bandwidth utilization
 */

#include <chrono>
#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef NVML_FOUND
#include <nvml.h>
#endif

// CUDA kernel to initialize test data
__global__ void init_data_kernel(cuDoubleComplex *data, int nx, int ny, int nz,
                                 int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nx * ny * nz;
  if (idx < total) {
    int k = idx / (nx * ny);
    int j = (idx % (nx * ny)) / nx;
    int i = idx % nx;
    double phase = 2.0 * M_PI * (i + j + k + seed) / (nx + ny + nz);
    data[idx] = make_cuDoubleComplex(cos(phase), sin(phase));
  }
}

int main(int argc, char **argv) {
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

  std::cout << "========================================" << std::endl;
  std::cout << "Direct cuFFT Power Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Grid size: " << grid_size << "^3" << std::endl;
  std::cout << "Duration: " << duration_seconds << " seconds" << std::endl;

  // Get CUDA device info
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "ERROR: No CUDA devices found!" << std::endl;
    return 1;
  }

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  std::cout << "CUDA device: " << device_id << " (" << prop.name << ")" << std::endl;
  std::cout << "Total device memory: " << (prop.totalGlobalMem / (1024 * 1024))
            << " MB" << std::endl;
  std::cout << "========================================" << std::endl;

  // Initialize NVML for power monitoring (if available)
#ifdef NVML_FOUND
  nvmlReturn_t nvml_result = NVML_SUCCESS;
  nvmlDevice_t nvml_device;
  unsigned int power_mw = 0;
  unsigned int power_limit_mw = 0;

  if (use_nvml) {
    nvml_result = nvmlInit();
    if (nvml_result == NVML_SUCCESS) {
      nvml_result = nvmlDeviceGetHandleByIndex(device_id, &nvml_device);
      if (nvml_result == NVML_SUCCESS) {
        nvmlDeviceGetPowerManagementLimitConstraints(nvml_device, &power_limit_mw,
                                                     nullptr);
        std::cout << "NVML initialized successfully" << std::endl;
        std::cout << "Power limit: " << power_limit_mw << " mW" << std::endl;
      }
    } else {
      std::cout << "Warning: NVML not available, using CUDA power APIs" << std::endl;
      use_nvml = false;
    }
  }
#else
  if (use_nvml) {
    std::cout << "Warning: NVML not compiled in, using CUDA power APIs" << std::endl;
  }
  use_nvml = false;
#endif

  // Allocate GPU memory for 3D FFT
  // cuFFT uses complex-to-complex format
  size_t grid_elements = static_cast<size_t>(grid_size) * grid_size * grid_size;
  size_t data_size = grid_elements * sizeof(cuDoubleComplex);

  cuDoubleComplex *d_data = nullptr;
  cudaError_t err = cudaMalloc(&d_data, data_size);
  if (err != cudaSuccess) {
    std::cerr << "ERROR: cudaMalloc failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  std::cout << "Allocated " << (data_size / (1024 * 1024)) << " MB on GPU"
            << std::endl;

  // Initialize input data on GPU
  int threads_per_block = 256;
  int blocks = (grid_elements + threads_per_block - 1) / threads_per_block;
  init_data_kernel<<<blocks, threads_per_block>>>(d_data, grid_size, grid_size,
                                                  grid_size, 0);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_data);
    return 1;
  }

  // Create cuFFT plan for 3D FFT
  cufftHandle plan;
  cufftResult cufft_err =
      cufftPlan3d(&plan, grid_size, grid_size, grid_size, CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    std::cerr << "ERROR: cufftPlan3d failed" << std::endl;
    cudaFree(d_data);
    return 1;
  }

  // Enable execution time measurement for cuFFT
  cufftSetAutoAllocation(plan, 0); // We'll manage workspace ourselves
  size_t work_size = 0;
  cufftMakePlan3d(plan, grid_size, grid_size, grid_size, CUFFT_Z2Z, &work_size);
  std::cout << "cuFFT workspace size: " << (work_size / (1024 * 1024)) << " MB"
            << std::endl;

  std::cout << "\nStarting FFT benchmark..." << std::endl;

  // Warm-up
  for (int i = 0; i < 10; ++i) {
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE);
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

  std::cout << "\nRunning benchmark for " << duration_seconds << " seconds..."
            << std::endl;

  // Main benchmark loop
  while (std::chrono::steady_clock::now() < end_time) {
    // Forward FFT
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);

    // Backward FFT
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE);

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

  // Cleanup
  cufftDestroy(plan);
  cudaFree(d_data);

  // Print results
  double avg_power_mw = (power_samples > 0) ? (total_power_mw / power_samples) : 0.0;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Benchmark Results" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Total iterations: " << iterations << std::endl;
  std::cout << "Actual duration: " << std::fixed << std::setprecision(2)
            << actual_duration << " seconds" << std::endl;
  std::cout << "Iterations per second: " << std::fixed << std::setprecision(2)
            << (iterations / actual_duration) << std::endl;
  std::cout << "Average power: " << std::fixed << std::setprecision(1)
            << (avg_power_mw / 1000.0) << " W" << std::endl;
  std::cout << "Max power: " << std::fixed << std::setprecision(1)
            << (max_power_mw / 1000.0) << " W" << std::endl;
  std::cout << "Min power: " << std::fixed << std::setprecision(1)
            << (min_power_mw / 1000.0) << " W" << std::endl;
  std::cout << "Power samples: " << power_samples << std::endl;
  std::cout << "========================================" << std::endl;

  // Write power data to CSV
  if (power_samples > 0) {
    std::ofstream csv_file("cufft_direct_benchmark_1gpu.csv");
    csv_file << "time_sec,power_w" << std::endl;
    for (size_t i = 0; i < power_readings.size(); ++i) {
      double time_sec =
          std::chrono::duration<double>(power_timestamps[i] - start_time).count();
      csv_file << std::fixed << std::setprecision(3) << time_sec << ","
               << (power_readings[i] / 1000.0) << std::endl;
    }
    csv_file.close();
    std::cout << "Power data saved to: cufft_direct_benchmark_1gpu.csv" << std::endl;
  } else {
    std::cout << "Warning: No power samples collected. Use --nvml flag or monitor "
                 "with nvidia-smi externally."
              << std::endl;
  }

#ifdef NVML_FOUND
  if (use_nvml && nvml_result == NVML_SUCCESS) {
    nvmlShutdown();
  }
#endif

  return 0;
}
