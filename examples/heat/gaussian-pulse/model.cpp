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
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

struct Parameters {
  int n = 32;
  int steps = 50;
  double length = 12;
  double width = 1;
  double diffusivity = 1;
  double dt = 0.01;

  static Parameters read(int argc, char **argv) {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    Parameters p{input.value("n", 32),
                 input.value("steps", 50),
                 input.value("length", 12.0),
                 input.value("width", 1.0),
                 input.value("diffusivity", 1.0),
                 input.value("dt", 0.01)};
    if (p.n < 4 || p.n > 64 || p.steps < 1 || p.steps > 10000 ||
        !std::isfinite(p.length) || p.length < 0.1 || p.length > 100 ||
        !std::isfinite(p.width) || p.width < p.length / 128 ||
        p.width > p.length / 4 || !std::isfinite(p.diffusivity) ||
        p.diffusivity <= 0 || !std::isfinite(p.dt) || p.dt <= 0 ||
        !std::isfinite(p.width * p.width + 2 * p.diffusivity * p.dt * p.steps) ||
        p.width * p.width + 2 * p.diffusivity * p.dt * p.steps > p.length * p.length)
      throw std::invalid_argument(
          "Use n in [4,64], steps in [1,10000], "
          "length in [0.1,100], width/length in [1/128,1/4], "
          "positive D,dt and width^2+2*D*dt*steps <= length^2");
    return p;
  }
};

// start snippet gaussian
double gaussian_line(double x, double time, const Parameters &p) {
  const double variance = p.width * p.width + 2 * p.diffusivity * time;
  double images = 0;
  for (int image = -12; image <= 12; ++image) {
    const double displacement = x + image * p.length;
    images += std::exp(-displacement * displacement / (2 * variance));
  }
  return p.width / std::sqrt(variance) * images;
}

double exact(double x, double y, double z, double time, const Parameters &p) {
  return gaussian_line(x, time, p) * gaussian_line(y, time, p) *
         gaussian_line(z, time, p);
}
// end snippet gaussian

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    const auto p = Parameters::read(argc, argv);
    const double h = p.length / p.n;
    const auto domain = pfc::domain::create(
        pfc::GridSize{{p.n, p.n, p.n}},
        pfc::PhysicalOrigin{{-p.length / 2, -p.length / 2, -p.length / 2}},
        pfc::GridSpacing{{h, h, h}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto &u = stack.u();
    auto &fft = stack.fft();
    // start snippet initialize
    u.apply([&](double x, double y, double z) { return exact(x, y, z, 0, p); });
    // end snippet initialize

    // start snippet evolve
    std::vector<std::complex<double>> modes(fft.size_outbox());
    std::vector<double> multiplier(modes.size());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          multiplier[i] =
              1 / (1 + p.dt * p.diffusivity * (kx * kx + ky * ky + kz * kz));
        });
    for (int step = 0; step < p.steps; ++step) {
      fft.forward(u.vec(), modes);
      for (std::size_t i = 0; i < modes.size(); ++i) modes[i] *= multiplier[i];
      fft.backward(modes, u.vec());
    }
    // end snippet evolve

    std::filesystem::create_directories("results");
    std::ofstream field("results/field_rank_" + std::to_string(comm.rank()) +
                        ".csv");
    if (!field) throw std::runtime_error("Cannot write Gaussian field");
    field << std::setprecision(17) << "x,y,z,initial,u,exact\n";
    std::array<double, 3> local{};
    double local_error = 0;
    const double time = p.dt * p.steps;
    u.for_each_owned([&](double x, double y, double z, double value) {
      const double initial = exact(x, y, z, 0, p);
      const double reference = exact(x, y, z, time, p);
      field << x << ',' << y << ',' << z << ',' << initial << ',' << value << ','
            << reference << '\n';
      local[0] += initial;
      local[1] += value;
      local[2] += (value - reference) * (value - reference);
      local_error = std::max(local_error, std::abs(value - reference));
    });
    if (!field) throw std::runtime_error("Cannot finish Gaussian field");
    std::array<double, 3> global{};
    double max_error = 0;
    MPI_Allreduce(local.data(), global.data(), 3, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (comm.rank() == 0) {
      std::ofstream summary("results/summary.json");
      if (!summary) throw std::runtime_error("Cannot write Gaussian summary");
      summary << std::setw(2)
              << nlohmann::json{{"time", time},
                                {"ranks", comm.size()},
                                {"initial_mass", h * h * h * global[0]},
                                {"mass", h * h * h * global[1]},
                                {"rms_error",
                                 std::sqrt(global[2] / (p.n * p.n * p.n))},
                                {"max_error", max_error}}
              << '\n';
      if (!summary) throw std::runtime_error("Cannot finish Gaussian summary");
      std::cout << "t=" << time << "  max error=" << max_error << '\n';
    }
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
