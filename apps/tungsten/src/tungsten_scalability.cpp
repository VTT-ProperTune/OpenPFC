// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_scalability.cpp
 * @brief Scalability study for 0.2 tungsten ETD sessions (CPU / CUDA / HIP)
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <tungsten/tungsten_etd_session.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <tungsten/tungsten_etd_gpu_session.hpp>
#endif

using nlohmann::json;

namespace {

json model_params() {
  return {{"n0", -0.4},         {"n_sol", -0.047},
          {"n_vap", -0.464},    {"T", 0.5},
          {"T0", 156000.0},     {"Bx", 0.8582},
          {"alpha", 0.50},      {"alpha_farTol", 0.001},
          {"alpha_highOrd", 4}, {"lambda", 0.22},
          {"stabP", 0.2},       {"shift_u", 0.3341},
          {"shift_s", 0.1898},  {"p2", 1.0},
          {"p3", -0.5},         {"p4", 0.333333333},
          {"q20", -0.0037},     {"q21", 1.0},
          {"q30", -12.4567},    {"q31", 20.0},
          {"q40", 45.0}};
}

json session_settings(int sx, int sy, int sz, double dt) {
  return {
      {"model", {{"name", "tungsten"}, {"params", model_params()}}},
      {"domain",
       {{"Lx", sx},
        {"Ly", sy},
        {"Lz", sz},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", dt}, {"dt", dt}, {"saveat", 1.0e9}}}};
}

void fill_sine(double *d, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = -0.4 + 0.1 * std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                                 static_cast<double>(n));
  }
}

} // namespace

struct ScalabilityResult {
  std::string backend;
  std::string precision;
  int size_x, size_y, size_z;
  int mpi_ranks;
  int num_iterations;
  double setup_time;
  double total_time;
  double fft_time;
  double other_time;
  double time_per_iteration;
  double fft_time_per_iteration;
  double other_time_per_iteration;
  double memory_used;
};

class ScalabilityStudy {
  std::string output_file;

public:
  explicit ScalabilityStudy(std::string csv_file)
      : output_file(std::move(csv_file)) {
    std::ofstream out(output_file);
    out << "backend,precision,size_x,size_y,size_z,mpi_ranks,iterations,"
        << "setup_time_sec,total_time_sec,fft_time_sec,other_time_sec,"
        << "time_per_iteration_sec,fft_time_per_iteration_sec,other_time_per_"
           "iteration_sec,memory_mb\n";
  }

  template <class Session, class FillPsi>
  void run_backend(const char *backend, int size_x, int size_y, int size_z,
                   int num_iterations, FillPsi &&fill) {
    int rank = 0;
    int nproc = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if (rank == 0) {
      std::cout << "Running " << backend << " test: size=(" << size_x << ","
                << size_y << "," << size_z << "), ranks=" << nproc
                << ", iterations=" << num_iterations << '\n';
    }
    try {
      const json settings = session_settings(size_x, size_y, size_z, 0.01);
      MPI_Barrier(MPI_COMM_WORLD);
      const double setup_start = MPI_Wtime();
      Session session(settings, rank, nproc, MPI_COMM_WORLD);
      fill(session);
      MPI_Barrier(MPI_COMM_WORLD);
      const double setup_time = MPI_Wtime() - setup_start;

      for (int i = 0; i < 3; ++i) {
        session.step_physics();
      }
      pfc::fft::reset_fft_time(session.fft());
      MPI_Barrier(MPI_COMM_WORLD);
      const double start_time = MPI_Wtime();
      for (int i = 0; i < num_iterations; ++i) {
        session.step_physics();
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const double total_time = MPI_Wtime() - start_time;
      const double fft_time = pfc::fft::get_fft_time(session.fft());
      const double other_time = total_time - fft_time;
      const double niter = static_cast<double>(num_iterations);

      if (rank == 0) {
        ScalabilityResult result;
        result.backend = backend;
        result.precision = "double";
        result.size_x = size_x;
        result.size_y = size_y;
        result.size_z = size_z;
        result.mpi_ranks = nproc;
        result.num_iterations = num_iterations;
        result.setup_time = setup_time;
        result.total_time = total_time;
        result.fft_time = fft_time;
        result.other_time = other_time;
        result.time_per_iteration = total_time / niter;
        result.fft_time_per_iteration = fft_time / niter;
        result.other_time_per_iteration = other_time / niter;
        result.memory_used = 0.0;
        save_result(result);
        std::cout << "  " << backend
                  << " (double): " << result.time_per_iteration * 1000.0
                  << " ms/iteration"
                  << " (FFT: " << result.fft_time_per_iteration * 1000.0
                  << " ms, Other: " << result.other_time_per_iteration * 1000.0
                  << " ms, Setup: " << setup_time * 1000.0 << " ms)" << '\n';
      }
    } catch (const std::exception &e) {
      if (rank == 0) {
        std::cerr << "Error in " << backend << " test: " << e.what() << '\n';
      }
    }
  }

  void run_cpu_test(int size_x, int size_y, int size_z, int num_iterations) {
    run_backend<tungsten::TungstenETDSession>("CPU", size_x, size_y, size_z,
                                              num_iterations, [](auto &session) {
                                                auto &psi = session.psi().vec();
                                                fill_sine(psi.data(), psi.size());
                                              });
  }

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
  void run_cuda_test(int size_x, int size_y, int size_z, int num_iterations) {
    run_backend<tungsten::TungstenETDCUDASession>(
        "CUDA", size_x, size_y, size_z, num_iterations, [](auto &session) {
          session.psi().with_host_view(
              [](double *d, std::size_t n) { fill_sine(d, n); });
        });
  }
#endif

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
  void run_hip_test(int size_x, int size_y, int size_z, int num_iterations) {
    run_backend<tungsten::TungstenETDHIPSession>(
        "HIP", size_x, size_y, size_z, num_iterations, [](auto &session) {
          session.psi().with_host_view(
              [](double *d, std::size_t n) { fill_sine(d, n); });
        });
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
  }
};

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string output_file = "tungsten_scalability_results.csv";
  int num_iterations = 30;
  if (argc > 1) {
    output_file = argv[1];
  }
  if (argc > 2) {
    try {
      num_iterations = std::stoi(argv[2]);
      if (num_iterations <= 0) {
        throw std::invalid_argument("num_iterations must be positive");
      }
    } catch (const std::exception &e) {
      if (rank == 0) {
        std::cerr << "Error: Invalid num_iterations argument '" << argv[2] << "' -- "
                  << e.what() << "\n";
        std::cerr << "Usage: mpirun -n <ranks> " << argv[0]
                  << " [output_file] [num_iterations] [scaling_mode]\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    std::cout << "Tungsten Scalability Study\nOutput file: " << output_file
              << "\nIterations per test: " << num_iterations
              << "\nMPI ranks: " << size << "\n\n";
  }

  ScalabilityStudy study(output_file);

  std::string scaling_mode = "strong";
  if (argc > 3) {
    scaling_mode = argv[3];
  } else if (const char *env_mode = std::getenv("SCALING_MODE")) {
    scaling_mode = env_mode;
  }
  if (rank == 0) {
    std::cout << "Scaling mode: " << scaling_mode << '\n';
  }

  auto run_gpu = [&](int sx, int sy, int sz) {
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
    study.run_cuda_test(sx, sy, sz, num_iterations);
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
    study.run_hip_test(sx, sy, sz, num_iterations);
#endif
  };

  if (scaling_mode == "gpu") {
    const std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}};
    for (const auto &[sx, sy, sz] : sizes) {
      run_gpu(sx, sy, sz);
    }
  } else if (scaling_mode == "cpu") {
    const std::vector<std::tuple<int, int, int>> sizes = {
        {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}};
    for (const auto &[sx, sy, sz] : sizes) {
      study.run_cpu_test(sx, sy, sz, num_iterations);
    }
  } else if (scaling_mode == "strong") {
    const int base_size = 256;
    study.run_cpu_test(base_size, base_size, base_size, num_iterations);
    run_gpu(base_size, base_size, base_size);
  } else if (scaling_mode == "weak") {
    const int size_per_rank = 64;
    const int total_size =
        size_per_rank * static_cast<int>(std::cbrt(static_cast<double>(size)));
    study.run_cpu_test(total_size, total_size, total_size, num_iterations);
    run_gpu(total_size, total_size, total_size);
  } else {
    const std::vector<std::tuple<int, int, int>> sizes = {
        {64, 64, 64},       {128, 128, 128}, {256, 256, 256}, {512, 512, 512},
        {1024, 1024, 1024}, {128, 128, 64},  {256, 256, 128}, {512, 512, 256}};
    for (const auto &[sx, sy, sz] : sizes) {
      study.run_cpu_test(sx, sy, sz, num_iterations);
      run_gpu(sx, sy, sz);
    }
  }

  if (rank == 0) {
    std::cout << "Scalability study complete. Results saved to: " << output_file
              << '\n';
  }
  MPI_Finalize();
  return 0;
}
