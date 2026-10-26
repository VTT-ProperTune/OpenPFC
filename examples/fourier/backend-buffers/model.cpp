// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/fft_gpu.hpp>
#endif

struct Parameters {
  std::array<int, 3> shape;
  std::array<double, 3> length;
  int cycles;
  std::string backend;

  static Parameters read(int argc, char **argv) {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const double pi = std::numbers::pi;
    Parameters p{input.value("shape", std::array{16, 18, 20}),
                 input.value("length", std::array{2 * pi, 3 * pi, pi}),
                 input.value("cycles", 3),
                 input.value("backend", std::string("cpu"))};
    for (int d = 0; d < 3; ++d)
      if (p.shape[d] < 8 || p.shape[d] > 64 || !std::isfinite(p.length[d]) ||
          p.length[d] < 0.1 || p.length[d] > 100)
        throw std::invalid_argument(
            "Use each shape in [8,64] and length in [0.1,100]");
    if (p.cycles < 1 || p.cycles > 25 ||
        (p.backend != "cpu" && p.backend != "cuda" && p.backend != "hip"))
      throw std::invalid_argument(
          "Use cycles in [1,25] and backend cpu, cuda or hip");
    return p;
  }
};

// start snippet signal
double signal(double x, double y, double z, const Parameters &p) {
  const double tau = 2 * std::numbers::pi;
  return 1 + 0.3 * std::cos(tau * x / p.length[0]) +
         0.2 * std::sin(2 * tau * y / p.length[1]) * std::cos(tau * z / p.length[2]);
}
// end snippet signal

struct Result {
  std::vector<double> field;
  std::vector<std::complex<double>> modes;
  pfc::Box3i outbox;
};

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
// start snippet buffers
template <class Tag, class FFT, class Synchronize>
Result round_trip_device(FFT &fft, const std::vector<double> &initial, int cycles,
                         Synchronize synchronize) {
  pfc::core::DataBuffer<Tag, double> real(fft.size_inbox());
  pfc::core::DataBuffer<Tag, std::complex<double>> modes(fft.size_outbox());
  real.copy_from_host(initial);
  for (int pass = 0; pass < cycles; ++pass) {
    fft.forward(real, modes);
    fft.backward(modes, real);
  }
  synchronize();
  return {real.to_host(), modes.to_host(), fft.get_outbox_bounds()};
}
// end snippet buffers
#endif

Result compute(const Parameters &p,
               const pfc::decomposition::Decomposition &decomposition,
               const std::vector<double> &initial,
               const pfc::mpi::communicator &comm) {
  if (p.backend == "cpu") {
    // start snippet cpu
    auto fft = pfc::fft::create(decomposition, comm.rank());
    auto real = initial;
    std::vector<std::complex<double>> modes(fft.size_outbox());
    for (int pass = 0; pass < p.cycles; ++pass) {
      fft.forward(real, modes);
      fft.backward(modes, real);
    }
    return {std::move(real), std::move(modes), fft.get_outbox_bounds()};
    // end snippet cpu
  }
  pfc::runtime::gpu::bind_local_device(comm);
  if (p.backend == "cuda") {
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
    // start snippet cuda
    auto fft = pfc::fft::create_cuda(decomposition, comm.rank(), comm);
    return round_trip_device<pfc::backend::CUDATag>(fft, initial, p.cycles, [] {
      if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA FFT execution failed");
    });
    // end snippet cuda
#else
    throw std::invalid_argument("CUDA spectral support is not compiled in");
#endif
  }
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
  auto fft = pfc::fft::create_hip(decomposition, comm.rank(), comm);
  return round_trip_device<pfc::backend::HIPTag>(fft, initial, p.cycles, [] {
    if (hipDeviceSynchronize() != hipSuccess)
      throw std::runtime_error("HIP FFT execution failed");
  });
#else
  throw std::invalid_argument("HIP spectral support is not compiled in");
#endif
}

void write_outputs(const pfc::data::Field<double> &original,
                   const pfc::Domain &domain, const Result &result,
                   const Parameters &p, const pfc::mpi::communicator &comm) {
  std::filesystem::create_directories("results");
  std::ofstream field("results/field_rank_" + std::to_string(comm.rank()) + ".csv");
  std::ofstream modes("results/modes_rank_" + std::to_string(comm.rank()) + ".csv");
  if (!field || !modes) throw std::runtime_error("Cannot write FFT output");
  field << std::setprecision(17) << "i,j,k,x,y,z,initial,u\n";
  modes << std::setprecision(17) << "i,j,k,real,imag\n";
  const auto low = original.box().low;
  double local_error = 0;
  std::array<double, 2> local{};
  original.for_each_owned([&](int i, int j, int k) {
    const auto x = original.coords(i, j, k);
    const double value = result.field[original.idx(i, j, k)];
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite FFT result");
    local_error = std::max(local_error, std::abs(value - original(i, j, k)));
    local[0] += value;
    local[1] += value * value;
    field << i + low[0] << ',' << j + low[1] << ',' << k + low[2] << ',' << x[0]
          << ',' << x[1] << ',' << x[2] << ',' << original(i, j, k) << ',' << value
          << '\n';
  });
  pfc::fft::kspace::for_each_kpoint(
      result.outbox, domain,
      [&](std::size_t at, double, double, double, int i, int j, int k) {
        const auto value = result.modes[at];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
          throw std::runtime_error("Nonfinite Fourier coefficient");
        modes << i << ',' << j << ',' << k << ',' << value.real() << ','
              << value.imag() << '\n';
      });
  if (!field || !modes) throw std::runtime_error("Cannot finish FFT output");
  double error = 0;
  std::array<double, 2> sums{};
  MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(local.data(), sums.data(), 2, MPI_DOUBLE, MPI_SUM, comm);
  if (error > 1e-10)
    throw std::runtime_error("Round-trip field error exceeds 1e-10");
  if (comm.rank() == 0) {
    const double cells = double(p.shape[0]) * p.shape[1] * p.shape[2];
    std::ofstream summary("results/summary.json");
    summary << std::setw(2) << nlohmann::json{{"backend", p.backend},
                                              {"ranks", comm.size()},
                                              {"shape", p.shape},
                                              {"length", p.length},
                                              {"cycles", p.cycles},
                                              {"max_error", error},
                                              {"mean", sums[0] / cells},
                                              {"mean_square", sums[1] / cells}}
            << '\n';
    if (!summary) throw std::runtime_error("Cannot finish FFT summary");
    std::cout << p.backend << " round trips: maximum field error=" << error << '\n';
  }
}

int main(int argc, char **argv) {
  try {
    const auto p = Parameters::read(argc, argv);
    if (p.backend != "cpu") pfc::runtime::gpu::bind_local_device_before_mpi();
    pfc::mpi::environment environment;
    pfc::mpi::communicator comm;
    try {
      const std::array spacing{p.length[0] / p.shape[0], p.length[1] / p.shape[1],
                               p.length[2] / p.shape[2]};
      const auto domain =
          pfc::domain::create(pfc::GridSize{p.shape}, pfc::PhysicalOrigin{{0, 0, 0}},
                              pfc::GridSpacing{spacing});
      const auto decomposition = pfc::decomposition::create(domain, comm.size());
      auto original =
          pfc::data::field_from_subdomain<double>(decomposition, comm.rank(), 0);
      original.apply(
          [&](double x, double y, double z) { return signal(x, y, z, p); });
      // Every rank constructs its plan and executes the same collective sequence.
      const auto result = compute(p, decomposition, original.vec(), comm);
      write_outputs(original, domain, result, p, comm);
    } catch (const std::exception &error) {
      if (comm.rank() == 0) std::cerr << error.what() << '\n';
      MPI_Abort(comm, 1);
      return 1;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
