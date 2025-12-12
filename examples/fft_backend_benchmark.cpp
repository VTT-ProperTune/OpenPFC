// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_backend_benchmark.cpp
 * @brief Benchmark CPU (FFTW) vs GPU (CUDA) FFT performance
 *
 * This example demonstrates:
 * - Runtime FFT backend selection
 * - Performance measurement using std::chrono
 * - Speedup comparison between CPU and GPU
 * - Proper usage of DataBuffer for GPU operations
 *
 * Compile with:
 *   cmake -B build -DOpenPFC_ENABLE_CUDA=ON
 *   cmake --build build --target fft_backend_benchmark
 *
 * Run:
 *   mpirun -np 1 ./examples/fft_backend_benchmark
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "openpfc/core/databuffer.hpp"
#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/fft.hpp"

using namespace pfc;

// Benchmark configuration
constexpr int GRID_SIZE = 128;  // 128³ = 2,097,152 points
constexpr int NUM_ITERATIONS = 10;  // Number of iterations for averaging

/**
 * @brief Benchmark FFT performance for a given backend
 *
 * @param backend The FFT backend to test (FFTW or CUDA)
 * @param world The computational domain
 * @param decomp Domain decomposition
 * @param rank_id MPI rank ID
 * @return Average time per forward+backward transform pair (in milliseconds)
 */
double benchmark_fft(fft::Backend backend, const World &world,
                     const decomposition::Decomposition &decomp, int rank_id) {
  
  std::string backend_name = (backend == fft::Backend::FFTW) ? "FFTW (CPU)" : "CUDA (GPU)";
  std::cout << "\n========================================\n";
  std::cout << "Benchmarking: " << backend_name << "\n";
  std::cout << "========================================\n";
  
  // Create FFT with selected backend
  auto fft = fft::create_with_backend(decomp, rank_id, backend);
  
  std::cout << "Grid size: " << GRID_SIZE << "³ = "
            << (GRID_SIZE * GRID_SIZE * GRID_SIZE) << " points\n";
  std::cout << "Real data size: " << fft->size_inbox() << " (local)\n";
  std::cout << "Complex data size: " << fft->size_outbox() << " (local)\n";
  std::cout << "Iterations: " << NUM_ITERATIONS << "\n\n";

  if (backend == fft::Backend::FFTW) {
    // CPU backend: use std::vector
    std::vector<double> real_data(fft->size_inbox());
    std::vector<std::complex<double>> complex_data(fft->size_outbox());
    
    // Initialize with some test data
    for (size_t i = 0; i < real_data.size(); ++i) {
      real_data[i] = std::sin(2.0 * M_PI * i / real_data.size());
    }
    
    // Warmup
    std::cout << "Warmup...";
    fft->forward(real_data, complex_data);
    fft->backward(complex_data, real_data);
    std::cout << " done.\n";
    
    // Benchmark
    std::cout << "Running benchmark...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
      fft->forward(real_data, complex_data);
      fft->backward(complex_data, real_data);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_ms = duration.count() / (1000.0 * NUM_ITERATIONS);
    
    std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Average time per forward+backward: " << std::fixed
              << std::setprecision(3) << avg_time_ms << " ms\n";
    
    return avg_time_ms;
    
  } else {
#if defined(OpenPFC_ENABLE_CUDA)
    // GPU backend: use DataBuffer
    using RealBufferGPU = core::DataBuffer<backend::CudaTag, double>;
    using ComplexBufferGPU = core::DataBuffer<backend::CudaTag, std::complex<double>>;
    
    RealBufferGPU real_data(fft->size_inbox());
    ComplexBufferGPU complex_data(fft->size_outbox());
    
    // Initialize on host, copy to device
    std::vector<double> host_data(fft->size_inbox());
    for (size_t i = 0; i < host_data.size(); ++i) {
      host_data[i] = std::sin(2.0 * M_PI * i / host_data.size());
    }
    real_data.copy_from_host(host_data);
    
    // Get the FFT_Impl with CUDA backend
    auto* fft_cuda = dynamic_cast<fft::FFT_Impl<heffte::backend::cufft>*>(fft.get());
    if (!fft_cuda) {
      throw std::runtime_error("Failed to cast to CUDA FFT implementation");
    }
    
    // Warmup
    std::cout << "Warmup...";
    fft_cuda->forward(real_data, complex_data);
    fft_cuda->backward(complex_data, real_data);
    cudaDeviceSynchronize();  // Ensure GPU work is complete
    std::cout << " done.\n";
    
    // Benchmark
    std::cout << "Running benchmark...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
      fft_cuda->forward(real_data, complex_data);
      fft_cuda->backward(complex_data, real_data);
    }
    
    cudaDeviceSynchronize();  // Ensure all GPU work is complete
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_ms = duration.count() / (1000.0 * NUM_ITERATIONS);
    
    std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Average time per forward+backward: " << std::fixed
              << std::setprecision(3) << avg_time_ms << " ms\n";
    
    return avg_time_ms;
#else
    throw std::runtime_error("CUDA support not compiled in");
#endif
  }
}

int main(int argc, char *argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (rank == 0) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  FFT Backend Performance Benchmark     ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";
    std::cout << "\nMPI ranks: " << size << "\n";
  }
  
  try {
    // Create computational domain (128³ grid)
    auto world = world::create({GRID_SIZE, GRID_SIZE, GRID_SIZE},
                               {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0});
    
    // Create domain decomposition
    auto decomp = decomposition::create(world, size);
    
    if (rank == 0) {
      std::cout << "Domain: " << GRID_SIZE << " × " << GRID_SIZE << " × "
                << GRID_SIZE << " = "
                << (GRID_SIZE * GRID_SIZE * GRID_SIZE) << " grid points\n";
    }
    
    // Benchmark CPU (FFTW)
    double cpu_time_ms = 0.0;
    if (rank == 0) {
      cpu_time_ms = benchmark_fft(fft::Backend::FFTW, world, decomp, rank);
    }
    
#if defined(OpenPFC_ENABLE_CUDA)
    // Benchmark GPU (CUDA)
    double gpu_time_ms = 0.0;
    if (rank == 0) {
      gpu_time_ms = benchmark_fft(fft::Backend::CUDA, world, decomp, rank);
    }
    
    // Report results
    if (rank == 0) {
      std::cout << "\n========================================\n";
      std::cout << "Performance Summary\n";
      std::cout << "========================================\n";
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "CPU (FFTW) time:  " << cpu_time_ms << " ms\n";
      std::cout << "GPU (CUDA) time:  " << gpu_time_ms << " ms\n";
      
      double speedup = cpu_time_ms / gpu_time_ms;
      std::cout << "\nSpeedup: " << std::setprecision(2) << speedup << "x\n";
      
      if (speedup > 1.0) {
        std::cout << "✓ GPU is " << speedup << "x faster than CPU\n";
      } else {
        std::cout << "✗ CPU is " << (1.0 / speedup) << "x faster than GPU\n";
        std::cout << "  (Note: GPU may be slower for small problems due to overhead)\n";
      }
      
      // Performance metrics
      size_t total_points = GRID_SIZE * GRID_SIZE * GRID_SIZE;
      double cpu_throughput = total_points / (cpu_time_ms * 1e-3) / 1e6;  // Mpoints/s
      double gpu_throughput = total_points / (gpu_time_ms * 1e-3) / 1e6;  // Mpoints/s
      
      std::cout << "\nThroughput:\n";
      std::cout << "  CPU: " << std::setprecision(1) << cpu_throughput << " Mpoints/s\n";
      std::cout << "  GPU: " << std::setprecision(1) << gpu_throughput << " Mpoints/s\n";
      
      std::cout << "\n========================================\n";
      std::cout << "Recommendation:\n";
      std::cout << "========================================\n";
      if (speedup > 2.0) {
        std::cout << "Use CUDA backend for production runs.\n";
        std::cout << "Set 'backend = \"cuda\"' in your config file.\n";
      } else if (speedup > 1.2) {
        std::cout << "CUDA provides modest speedup.\n";
        std::cout << "Consider GPU for large-scale problems (>256³).\n";
      } else {
        std::cout << "FFTW (CPU) is sufficient for this problem size.\n";
        std::cout << "GPU overhead dominates for smaller problems.\n";
      }
    }
#else
    if (rank == 0) {
      std::cout << "\n========================================\n";
      std::cout << "CUDA Support Not Enabled\n";
      std::cout << "========================================\n";
      std::cout << "To enable GPU benchmarking, rebuild with:\n";
      std::cout << "  cmake -DOpenPFC_ENABLE_CUDA=ON -B build\n";
      std::cout << "\nCPU (FFTW) time: " << cpu_time_ms << " ms\n";
    }
#endif
    
  } catch (const std::exception &e) {
    if (rank == 0) {
      std::cerr << "\nError: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }
  
  MPI_Finalize();
  return 0;
}
