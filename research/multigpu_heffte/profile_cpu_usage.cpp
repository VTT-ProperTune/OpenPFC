// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file profile_cpu_usage.cpp
 * @brief Profile CPU usage during FFT operations to identify bottlenecks
 *
 * This tool helps identify what the CPU is doing during GPU operations.
 * It can be used with perf or strace to see system calls and CPU activity.
 */

#include "heffte.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <thread>
#include <unistd.h> // for getpid()
#include <vector>

using backend_tag = heffte::backend::cufft;

// Global flag to track CPU activity
std::atomic<bool> gpu_busy{false};
std::atomic<long long> sync_calls{0};
std::atomic<long long> fft_calls{0};

// CPU monitoring thread
void cpu_monitor_thread() {
  auto start = std::chrono::steady_clock::now();
  long long last_sync = 0;
  long long last_fft = 0;

  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start).count();

    long long current_sync = sync_calls.load();
    long long current_fft = fft_calls.load();

    if (elapsed > 0.1) {                                     // After warmup
      long long sync_rate = (current_sync - last_sync) * 10; // per second
      long long fft_rate = (current_fft - last_fft) * 10;

      std::cerr << "[CPU Monitor] t=" << std::fixed << std::setprecision(2)
                << elapsed << "s, GPU_busy=" << (gpu_busy.load() ? "YES" : "NO")
                << ", Sync_calls=" << current_sync << " (" << sync_rate << "/s)"
                << ", FFT_calls=" << current_fft << " (" << fft_rate << "/s)"
                << std::endl;
    }

    last_sync = current_sync;
    last_fft = current_fft;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int const my_rank = heffte::mpi::comm_rank(MPI_COMM_WORLD);
  int const num_ranks = heffte::mpi::comm_size(MPI_COMM_WORLD);

  // Parse command line arguments
  int grid_size = 512;
  double duration_seconds = 60.0; // Shorter for profiling
  bool enable_monitor = false;

  if (argc > 1) {
    grid_size = std::stoi(argv[1]);
  }
  if (argc > 2) {
    duration_seconds = std::stod(argv[2]);
  }
  if (argc > 3 && std::string(argv[3]) == "--monitor") {
    enable_monitor = true;
  }

  if (my_rank == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "CPU Usage Profiler for FFT Operations" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Grid size: " << grid_size << "^3" << std::endl;
    std::cout << "MPI ranks: " << num_ranks << std::endl;
    std::cout << "Duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nUse 'perf top -p <PID>' or 'strace -p <PID>' to monitor"
              << std::endl;
    std::cout << "PID: " << getpid() << std::endl;
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
  }

  // Start CPU monitoring thread
  std::thread monitor;
  if (enable_monitor && my_rank == 0) {
    monitor = std::thread(cpu_monitor_thread);
  }

  // Create FFT with proper domain decomposition
  int const nz_per_rank = grid_size / num_ranks;
  int const z_start = my_rank * nz_per_rank;
  int const z_end =
      (my_rank == num_ranks - 1) ? grid_size - 1 : (my_rank + 1) * nz_per_rank - 1;

  heffte::box3d<> const inbox = {{0, 0, z_start},
                                 {grid_size - 1, grid_size - 1, z_end}};
  heffte::box3d<> const outbox = inbox;

  heffte::plan_options options = heffte::default_options<backend_tag>();
  heffte::fft3d<backend_tag> fft(inbox, outbox, MPI_COMM_WORLD, options);

  // Allocate GPU memory
  size_t const size_inbox = fft.size_inbox();
  heffte::gpu::vector<std::complex<double>> gpu_input(size_inbox);
  heffte::gpu::vector<std::complex<double>> gpu_output(size_inbox);
  auto workspace = fft.size_workspace();
  heffte::gpu::vector<std::complex<double>> fft_workspace(workspace);

  // Initialize data
  std::vector<std::complex<double>> host_data(size_inbox);
  for (size_t i = 0; i < size_inbox; ++i) {
    host_data[i] = std::complex<double>(1.0, 0.0);
  }
  // Copy to GPU
  cudaMemcpy(gpu_input.data(), host_data.data(),
             size_inbox * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  if (my_rank == 0) {
    std::cout << "\nStarting profiled FFT operations..." << std::endl;
    std::cout << "Monitor CPU with: perf top -p " << getpid() << std::endl;
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    gpu_busy = true;
    fft.forward(gpu_input.data(), gpu_output.data(), fft_workspace.data());
    fft.backward(gpu_output.data(), gpu_input.data(), fft_workspace.data());
    fft_calls += 2;
    sync_calls++;
    cudaDeviceSynchronize();
    gpu_busy = false;
  }

  // Main loop with profiling
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = start_time + std::chrono::duration<double>(duration_seconds);

  long long iterations = 0;
  std::chrono::duration<double> total_sync_time{0};
  std::chrono::duration<double> total_fft_time{0};

  while (std::chrono::steady_clock::now() < end_time) {
    // Time FFT operations
    auto fft_start = std::chrono::steady_clock::now();
    gpu_busy = true;

    fft.forward(gpu_input.data(), gpu_output.data(), fft_workspace.data());
    fft_calls++;

    fft.backward(gpu_output.data(), gpu_input.data(), fft_workspace.data());
    fft_calls++;

    auto fft_end = std::chrono::steady_clock::now();
    total_fft_time += (fft_end - fft_start);

    // Time synchronization
    auto sync_start = std::chrono::steady_clock::now();
    sync_calls++;
    cudaDeviceSynchronize();
    auto sync_end = std::chrono::steady_clock::now();
    total_sync_time += (sync_end - sync_start);
    gpu_busy = false;

    iterations++;
  }

  if (enable_monitor && my_rank == 0) {
    monitor.detach(); // Let it run until process ends
  }

  if (my_rank == 0) {
    double total_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
            .count();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Profiling Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time
              << " s" << std::endl;
    std::cout << "FFT time: " << std::fixed << std::setprecision(2)
              << total_fft_time.count() << " s ("
              << (100.0 * total_fft_time.count() / total_time) << "%)" << std::endl;
    std::cout << "Sync time: " << std::fixed << std::setprecision(2)
              << total_sync_time.count() << " s ("
              << (100.0 * total_sync_time.count() / total_time) << "%)" << std::endl;
    std::cout << "Other time: " << std::fixed << std::setprecision(2)
              << (total_time - total_fft_time.count() - total_sync_time.count())
              << " s ("
              << (100.0 *
                  (total_time - total_fft_time.count() - total_sync_time.count()) /
                  total_time)
              << "%)" << std::endl;
    std::cout << "Sync calls: " << sync_calls.load() << std::endl;
    std::cout << "FFT calls: " << fft_calls.load() << std::endl;
    std::cout << "========================================" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
