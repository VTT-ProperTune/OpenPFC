// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_backend_benchmark.cpp
 * @brief Benchmark CPU (FFTW) vs GPU (CUDA / HIP) FFT performance
 *
 * This example demonstrates:
 * - Runtime FFT backend selection
 * - Performance measurement using std::chrono
 * - Speedup comparison between CPU and GPU
 * - Proper usage of DataBuffer for GPU operations
 *
 * Compile with CUDA:
 *   cmake -B build -DOpenPFC_ENABLE_CUDA=ON
 * Compile with HIP:
 *   cmake -B build -DOpenPFC_ENABLE_HIP=ON
 *   cmake --build build --target fft_backend_benchmark
 *
 * Run:
 *   mpirun -np 1 ./examples/fft_backend_benchmark
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#endif
#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <openpfc/runtime/cuda/fft_cuda.hpp>
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <hip/hip_runtime.h>
#include <openpfc/runtime/hip/fft_hip.hpp>
#endif

using namespace pfc;

constexpr int GRID_SIZE = 128;     // 128³ = 2,097,152 points
constexpr int NUM_ITERATIONS = 10; // Number of iterations for averaging

static const char *backend_label(fft::Backend backend) {
  switch (backend) {
  case fft::Backend::FFTW:
    return "FFTW (CPU)";
  case fft::Backend::CUDA:
    return "CUDA (GPU)";
  case fft::Backend::HIP:
    return "HIP (GPU)";
  }
  return "unknown";
}

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
template <typename Tag, typename GpuFft, typename Sync>
double benchmark_gpu_fft(GpuFft &fft, Sync &&sync) {
  using RealBufferGPU = core::DataBuffer<Tag, double>;
  using ComplexBufferGPU = core::DataBuffer<Tag, std::complex<double>>;

  RealBufferGPU real_data(fft.size_inbox());
  ComplexBufferGPU complex_data(fft.size_outbox());

  std::vector<double> host_data(fft.size_inbox());
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = std::sin(2.0 * std::numbers::pi * i / host_data.size());
  }
  real_data.copy_from_host(host_data);

  std::cout << "Warmup...";
  fft.forward(real_data, complex_data);
  fft.backward(complex_data, real_data);
  sync();
  std::cout << " done.\n";

  std::cout << "Running benchmark...\n";
  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
    fft.forward(real_data, complex_data);
    fft.backward(complex_data, real_data);
  }

  sync();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double avg_time_ms = duration.count() / (1000.0 * NUM_ITERATIONS);

  std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
  std::cout << "Average time per forward+backward: " << std::fixed
            << std::setprecision(3) << avg_time_ms << " ms\n";

  return avg_time_ms;
}
#endif

/**
 * @brief Benchmark FFT performance for a given backend
 *
 * @param backend The FFT backend to test (FFTW, CUDA, or HIP)
 * @param world The computational domain
 * @param decomp Domain decomposition
 * @param rank_id MPI rank ID
 * @return Average time per forward+backward transform pair (in milliseconds)
 */
double benchmark_fft(fft::Backend backend, const Domain &world,
                     const decomposition::Decomposition &decomp, int rank_id) {
  (void)world;
  std::cout << "\n========================================\n";
  std::cout << "Benchmarking: " << backend_label(backend) << "\n";
  std::cout << "========================================\n";

  std::cout << "Grid size: " << GRID_SIZE
            << "³ = " << (GRID_SIZE * GRID_SIZE * GRID_SIZE) << " points\n";
  std::cout << "Iterations: " << NUM_ITERATIONS << "\n\n";

  if (backend == fft::Backend::FFTW) {
    auto fft = fft::create_with_backend(decomp, rank_id, backend);
    std::cout << "Real data size: " << fft->size_inbox() << " (local)\n";
    std::cout << "Complex data size: " << fft->size_outbox() << " (local)\n";
    std::vector<double> real_data(fft->size_inbox());
    std::vector<std::complex<double>> complex_data(fft->size_outbox());

    for (size_t i = 0; i < real_data.size(); ++i) {
      real_data[i] = std::sin(2.0 * std::numbers::pi * i / real_data.size());
    }

    std::cout << "Warmup...";
    fft->forward(real_data, complex_data);
    fft->backward(complex_data, real_data);
    std::cout << " done.\n";

    std::cout << "Running benchmark...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
      fft->forward(real_data, complex_data);
      fft->backward(complex_data, real_data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_ms = duration.count() / (1000.0 * NUM_ITERATIONS);

    std::cout << "Total time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Average time per forward+backward: " << std::fixed
              << std::setprecision(3) << avg_time_ms << " ms\n";

    return avg_time_ms;
  }

  if (backend == fft::Backend::CUDA) {
#if defined(OpenPFC_ENABLE_CUDA)
    auto gpu = fft::create_cuda(decomp, rank_id);
    std::cout << "Real data size: " << gpu.size_inbox() << " (local)\n";
    std::cout << "Complex data size: " << gpu.size_outbox() << " (local)\n";
    return benchmark_gpu_fft<backend::CudaTag>(gpu,
                                               [] { cudaDeviceSynchronize(); });
#else
    throw std::runtime_error("CUDA support not compiled in");
#endif
  }

  if (backend == fft::Backend::HIP) {
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
    auto gpu = fft::create_hip(decomp, rank_id, MPI_COMM_WORLD);
    std::cout << "Real data size: " << gpu.size_inbox() << " (local)\n";
    std::cout << "Complex data size: " << gpu.size_outbox() << " (local)\n";
    return benchmark_gpu_fft<backend::HipTag>(gpu,
                                              [] { hipDeviceSynchronize(); });
#else
    throw std::runtime_error("HIP spectral support not compiled in");
#endif
  }

  throw std::runtime_error("Unsupported FFT backend requested");
}

static void print_gpu_summary(const char *gpu_name, const char *config_key,
                              double cpu_time_ms, double gpu_time_ms) {
  std::cout << "\n========================================\n";
  std::cout << "Performance Summary (" << gpu_name << ")\n";
  std::cout << "========================================\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "CPU (FFTW) time:  " << cpu_time_ms << " ms\n";
  std::cout << "GPU (" << gpu_name << ") time:  " << gpu_time_ms << " ms\n";

  double speedup = cpu_time_ms / gpu_time_ms;
  std::cout << "\nSpeedup: " << std::setprecision(2) << speedup << "x\n";

  if (speedup > 1.0) {
    std::cout << "GPU is " << speedup << "x faster than CPU\n";
  } else {
    std::cout << "CPU is " << (1.0 / speedup) << "x faster than GPU\n";
    std::cout
        << "  (Note: GPU may be slower for small problems due to overhead)\n";
  }

  size_t total_points = static_cast<size_t>(GRID_SIZE) * GRID_SIZE * GRID_SIZE;
  double cpu_throughput = total_points / (cpu_time_ms * 1e-3) / 1e6;
  double gpu_throughput = total_points / (gpu_time_ms * 1e-3) / 1e6;

  std::cout << "\nThroughput:\n";
  std::cout << "  CPU: " << std::setprecision(1) << cpu_throughput
            << " Mpoints/s\n";
  std::cout << "  GPU: " << std::setprecision(1) << gpu_throughput
            << " Mpoints/s\n";

  std::cout << "\n========================================\n";
  std::cout << "Recommendation:\n";
  std::cout << "========================================\n";
  if (speedup > 2.0) {
    std::cout << "Use the " << gpu_name << " backend for production runs.\n";
    std::cout << "Set 'backend = \"" << config_key << "\"' in your config file.\n";
  } else if (speedup > 1.2) {
    std::cout << gpu_name << " provides modest speedup.\n";
    std::cout << "Consider GPU for large-scale problems (>256³).\n";
  } else {
    std::cout << "FFTW (CPU) is sufficient for this problem size.\n";
    std::cout << "GPU overhead dominates for smaller problems.\n";
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n========================================\n";
    std::cout << "  FFT Backend Performance Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "\nMPI ranks: " << size << "\n";
  }

  try {
    auto world =
        domain::create_world_from_bounds({GRID_SIZE, GRID_SIZE, GRID_SIZE},
                                          {1.0, 1.0, 1.0}, {128.0, 128.0, 128.0});

    auto decomp = decomposition::create(world, size);

    if (rank == 0) {
      std::cout << "Domain: " << GRID_SIZE << " × " << GRID_SIZE << " × "
                << GRID_SIZE << " = " << (GRID_SIZE * GRID_SIZE * GRID_SIZE)
                << " grid points\n";
    }

    double cpu_time_ms = 0.0;
    if (rank == 0) {
      cpu_time_ms = benchmark_fft(fft::Backend::FFTW, world.domain_, decomp, rank);
    }

    bool ran_gpu = false;

#if defined(OpenPFC_ENABLE_CUDA)
    if (rank == 0) {
      const double cuda_time_ms =
          benchmark_fft(fft::Backend::CUDA, world.domain_, decomp, rank);
      print_gpu_summary("CUDA", "cuda", cpu_time_ms, cuda_time_ms);
      ran_gpu = true;
    }
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
    if (rank == 0) {
      const double hip_time_ms =
          benchmark_fft(fft::Backend::HIP, world.domain_, decomp, rank);
      print_gpu_summary("HIP", "hip", cpu_time_ms, hip_time_ms);
      ran_gpu = true;
    }
#endif

    if (!ran_gpu && rank == 0) {
      std::cout << "\n========================================\n";
      std::cout << "GPU Support Not Enabled\n";
      std::cout << "========================================\n";
      std::cout << "To enable GPU benchmarking, rebuild with CUDA or HIP:\n";
      std::cout << "  cmake -DOpenPFC_ENABLE_CUDA=ON ...\n";
      std::cout << "  cmake -DOpenPFC_ENABLE_HIP=ON ...\n";
      std::cout << "\nCPU (FFTW) time: " << cpu_time_ms << " ms\n";
    }

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
