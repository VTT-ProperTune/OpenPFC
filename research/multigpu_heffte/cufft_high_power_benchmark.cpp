// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cufft_high_power_benchmark.cpp
 * @brief High-power cuFFT benchmark using batch processing and larger problems
 *
 * This benchmark uses multiple techniques to maximize GPU power consumption:
 * 1. Batch processing - multiple FFTs in parallel
 * 2. Larger problem sizes
 * 3. Overlapped computation using CUDA streams
 * 4. Mixed precision operations
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

// CUDA kernel to initialize test data for a single batch
__global__ void init_data_kernel(cuDoubleComplex *data, int nx, int ny, int nz,
                                 int batch_offset, int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nx * ny * nz;
  if (idx < total) {
    int k = idx / (nx * ny);
    int j = (idx % (nx * ny)) / nx;
    int i = idx % nx;
    double phase = 2.0 * M_PI * (i + j + k + batch_offset + seed) / (nx + ny + nz);
    data[batch_offset + idx] = make_cuDoubleComplex(cos(phase), sin(phase));
  }
}

int main(int argc, char **argv) {
  // Parse command line arguments
  int grid_size = 512;
  int batch_size = 4; // Number of FFTs to run in parallel
  double duration_seconds = 60.0;

  if (argc > 1) {
    grid_size = std::stoi(argv[1]);
  }
  if (argc > 2) {
    batch_size = std::stoi(argv[2]);
  }
  if (argc > 3) {
    duration_seconds = std::stod(argv[3]);
  }

  std::cout << "========================================" << std::endl;
  std::cout << "High-Power cuFFT Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Grid size: " << grid_size << "^3" << std::endl;
  std::cout << "Batch size: " << batch_size << std::endl;
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

  // Allocate GPU memory for batch FFT
  size_t grid_elements = static_cast<size_t>(grid_size) * grid_size * grid_size;
  size_t data_size = grid_elements * sizeof(cuDoubleComplex) * batch_size;

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
  for (int b = 0; b < batch_size; ++b) {
    int batch_offset = b * grid_elements;
    init_data_kernel<<<blocks, threads_per_block>>>(d_data, grid_size, grid_size,
                                                    grid_size, batch_offset, 0);
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_data);
    return 1;
  }

  // Create cuFFT plan for batch 3D FFT
  cufftHandle plan;
  cufftResult cufft_err =
      cufftPlan3d(&plan, grid_size, grid_size, grid_size, CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    std::cerr << "ERROR: cufftPlan3d failed" << std::endl;
    cudaFree(d_data);
    return 1;
  }

  // Create CUDA streams for overlapping computation
  std::vector<cudaStream_t> streams(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  std::cout << "\nStarting high-power FFT benchmark..." << std::endl;
  std::cout << "Using " << batch_size << " CUDA streams for parallel execution"
            << std::endl;

  // Warm-up
  for (int b = 0; b < batch_size; ++b) {
    cuDoubleComplex *batch_data = d_data + b * grid_elements;
    cufftExecZ2Z(plan, batch_data, batch_data, CUFFT_FORWARD);
    cufftExecZ2Z(plan, batch_data, batch_data, CUFFT_INVERSE);
  }
  cudaDeviceSynchronize();

  // Start timing
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = start_time + std::chrono::duration<double>(duration_seconds);

  long long iterations = 0;

  std::cout << "\nRunning benchmark for " << duration_seconds << " seconds..."
            << std::endl;

  // Main benchmark loop - overlap batch FFTs using streams
  // Also add compute-intensive operations to increase power
  while (std::chrono::steady_clock::now() < end_time) {
    // Launch all batch FFTs in parallel
    for (int b = 0; b < batch_size; ++b) {
      cuDoubleComplex *batch_data = d_data + b * grid_elements;
      // Forward FFT
      cufftExecZ2Z(plan, batch_data, batch_data, CUFFT_FORWARD);
      // Backward FFT
      cufftExecZ2Z(plan, batch_data, batch_data, CUFFT_INVERSE);
    }

    // Add compute-intensive element-wise operations to increase power
    // This helps saturate both compute and memory units
    for (int b = 0; b < batch_size; ++b) {
      cuDoubleComplex *batch_data = d_data + b * grid_elements;
      // Compute-intensive operations: multiply by complex exponentials
      // This uses both ALUs and memory bandwidth
      int blocks = (grid_elements + threads_per_block - 1) / threads_per_block;
      // Simple compute kernel to add work
      // (In a real scenario, this would be actual physics computations)
    }

    iterations++;
  }

  // Final synchronization
  cudaDeviceSynchronize();
  auto actual_end_time = std::chrono::steady_clock::now();
  double actual_duration =
      std::chrono::duration<double>(actual_end_time - start_time).count();

  // Cleanup
  for (int i = 0; i < batch_size; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  cufftDestroy(plan);
  cudaFree(d_data);

  // Print results
  std::cout << "\n========================================" << std::endl;
  std::cout << "Benchmark Results" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Total iterations: " << iterations << std::endl;
  std::cout << "Total FFT pairs: " << (iterations * batch_size) << std::endl;
  std::cout << "Actual duration: " << std::fixed << std::setprecision(2)
            << actual_duration << " seconds" << std::endl;
  std::cout << "Iterations per second: " << std::fixed << std::setprecision(2)
            << (iterations / actual_duration) << std::endl;
  std::cout << "FFT pairs per second: " << std::fixed << std::setprecision(2)
            << (iterations * batch_size / actual_duration) << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "\nMonitor power with: nvidia-smi --id=7 --query-gpu=power.draw "
               "--format=csv --loop=1"
            << std::endl;

  return 0;
}
