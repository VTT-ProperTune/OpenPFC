// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** \file
 * Periodic, one-dimensional Allen–Cahn dynamics embedded in an N x 4 x 4
 * distributed FFT grid. Keep only (kx,0,0), |kx| <= K, with 4K < N:
 * sampling the cubic then projecting is the alias-free Galerkin product.
 * The forward FFT is unscaled; the inverse divides by the global cell count.
 * ETD2 integrates the linear part exactly and approximates the cubic in time.
 * Run with input.json; owned fields, normalized coefficients and diagnostics
 * are written beneath results/. The companion verify.py supplies an
 * independent direct-convolution/RK4 reference and exercises this executable.
 */

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
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

namespace {
using Complex = std::complex<double>;
using Json = nlohmann::json;
constexpr double length = 2 * std::numbers::pi;

struct Settings {
  int n, cutoff, steps;
  double dt, diffusivity, mean, a1, a3;
};

Settings read_settings(const char *path) {
  std::ifstream stream(path);
  if (!stream) throw std::invalid_argument("cannot open input JSON");
  Json input;
  stream >> input;
  const auto integer = [&](const char *name, int lower, int upper) {
    const auto &value = input.at(name);
    if (!value.is_number_integer() || value.get<double>() < lower ||
        value.get<double>() > upper)
      throw std::invalid_argument(std::string(name) +
                                  " is outside its integer range");
    return value.get<int>();
  };
  Settings s{integer("n", 16, 128),
             integer("cutoff", 3, 31),
             integer("steps", 1, 20000),
             input.at("dt").get<double>(),
             input.at("diffusivity").get<double>(),
             input.at("mean").get<double>(),
             input.at("a1").get<double>(),
             input.at("a3").get<double>()};
  for (double value : {s.dt, s.diffusivity, s.mean, s.a1, s.a3})
    if (!std::isfinite(value)) throw std::invalid_argument("input must be finite");
  if (s.n < 16 || s.n > 128 || s.n % 2 || s.cutoff < 3 || s.cutoff > (s.n - 1) / 4 ||
      s.steps < 1 || s.steps > 20000 || s.dt <= 0 || s.dt > 0.05 ||
      s.steps * s.dt > 10 || s.diffusivity <= 0 || s.diffusivity > 2 ||
      std::abs(s.mean) + std::abs(s.a1) + std::abs(s.a3) > 1)
    throw std::invalid_argument(
        "require even 16<=n<=128, 3<=cutoff<n/4, 1<=steps<=20000, "
        "0<dt<=0.05, steps*dt<=10, 0<diffusivity<=2, "
        "abs(mean)+abs(a1)+abs(a3)<=1");
  return s;
}

// phi2(z) = (exp(z)-1-z)/z^2, including its removable singularity.
double phi2(double z) {
  if (std::abs(z) < 1e-4) return 0.5 + z * (1.0 / 6 + z * (1.0 / 24 + z / 120));
  return (std::expm1(z) - z) / (z * z);
}

struct AllenCahn {
  double diffusivity;
  double linear(double k_squared) const { return 1 - diffusivity * k_squared; }
  double nonlinear(double value) const { return -value * value * value; }
};

class Solver {
public:
  Solver(Settings settings, int rank, int ranks)
      : s(settings), physics{s.diffusivity},
        stack(pfc::domain::create(
                  pfc::GridSize{{s.n, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
                  pfc::GridSpacing{{length / s.n, length / 4, length / 4}}),
              rank, ranks),
        spectrum(stack.fft().size_outbox()), predictor(spectrum.size()),
        first(spectrum.size()), second(spectrum.size()),
        scratch(stack.fft().size_inbox()), keep(spectrum.size()),
        exponential(spectrum.size()), h_phi1(spectrum.size()),
        h_phi2(spectrum.size()), k_squared(spectrum.size()) {
    stack.u().apply([&](double x, double, double) {
      return s.mean + s.a1 * std::cos(x) + s.a3 * std::cos(3 * x);
    });
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), stack.u().domain(),
        [&](std::size_t index, double kx, double ky, double kz, int i, int j,
            int k) {
          // The r2c-compressed x index is nonnegative; y,z retain only DC.
          keep[index] = i <= s.cutoff && j == 0 && k == 0;
          k_squared[index] = kx * kx + ky * ky + kz * kz;
          const double linear = physics.linear(k_squared[index]);
          const auto coefficients =
              pfc::integrator::spectral_exp_coeffs(linear, s.dt);
          exponential[index] = coefficients.exp_Ldt;
          h_phi1[index] = coefficients.phi1_L;
          h_phi2[index] = s.dt * phi2(s.dt * linear);
        });
    stack.fft().forward(stack.u().vec(), spectrum);
    project(spectrum);
  }

  void step() {
    nonlinear(spectrum, first);
    for (std::size_t i = 0; i < spectrum.size(); ++i)
      predictor[i] = exponential[i] * spectrum[i] + h_phi1[i] * first[i];
    nonlinear(predictor, second);
    for (std::size_t i = 0; i < spectrum.size(); ++i)
      spectrum[i] = predictor[i] + h_phi2[i] * (second[i] - first[i]);
    project(spectrum);
  }

  // RMS and energy are global averages; the reaction does not conserve mean.
  std::array<double, 3> diagnostics() {
    stack.fft().backward(spectrum, stack.u().vec());
    double local[4]{};
    for (double value : stack.u().vec()) {
      if (!std::isfinite(value)) throw std::runtime_error("non-finite field");
      local[0] += value;
      local[1] += value * value;
      local[2] += std::pow(value * value - 1, 2) / 4;
    }
    const double count = static_cast<double>(s.n) * 4 * 4;
    const auto box = stack.fft().get_outbox_bounds();
    pfc::fft::kspace::for_each_kpoint(
        box, stack.u().domain(),
        [&](std::size_t index, double, double, double, int i, int, int) {
          const double weight = (i == 0 || i == s.n / 2) ? 1 : 2;
          local[3] += weight * k_squared[index] * std::norm(spectrum[index]) / count;
        });
    double global[4]{};
    MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return {global[0] / count, std::sqrt(global[1] / count),
            (global[2] + physics.diffusivity * global[3] / 2) / count};
  }

  void write_fields(int rank) {
    std::ofstream field("results/field_rank_" + std::to_string(rank) + ".csv");
    field.exceptions(std::ios::failbit | std::ios::badbit);
    field << "i,j,k,u\n" << std::setprecision(17);
    stack.u().for_each_owned([&](double x, double y, double z, double value) {
      field << std::llround(x * s.n / length) << ',' << std::llround(y * 4 / length)
            << ',' << std::llround(z * 4 / length) << ',' << value << '\n';
    });
    std::ofstream modes("results/modes_rank_" + std::to_string(rank) + ".csv");
    modes.exceptions(std::ios::failbit | std::ios::badbit);
    modes << "i,j,k,re,im\n" << std::setprecision(17);
    const double count = static_cast<double>(s.n) * 4 * 4;
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), stack.u().domain(),
        [&](std::size_t index, double, double, double, int i, int j, int k) {
          modes << i << ',' << j << ',' << k << ',' << spectrum[index].real() / count
                << ',' << spectrum[index].imag() / count << '\n';
        });
    const auto inbox = stack.fft().get_inbox_bounds();
    const auto outbox = stack.fft().get_outbox_bounds();
    std::ofstream layout("results/layout_rank_" + std::to_string(rank) + ".json");
    layout.exceptions(std::ios::failbit | std::ios::badbit);
    layout << Json{{"rank", rank},
                   {"inbox_low", inbox.low},
                   {"inbox_high", inbox.high},
                   {"outbox_low", outbox.low},
                   {"outbox_high", outbox.high}}
                  .dump(2)
           << '\n';
  }

private:
  void project(std::vector<Complex> &values) const {
    for (std::size_t i = 0; i < values.size(); ++i)
      if (!keep[i]) values[i] = 0;
  }

  void nonlinear(const std::vector<Complex> &state, std::vector<Complex> &rates) {
    stack.fft().backward(state, scratch);
    for (double &value : scratch) value = physics.nonlinear(value);
    stack.fft().forward(scratch, rates);
    project(rates);
  }

  Settings s;
  AllenCahn physics;
  pfc::sim::stacks::SpectralCPUStack stack;
  std::vector<Complex> spectrum, predictor, first, second;
  std::vector<double> scratch;
  std::vector<bool> keep;
  std::vector<double> exponential, h_phi1, h_phi2, k_squared;
};

void run(const Settings &s, int rank, int ranks) {
  Solver solver(s, rank, ranks);
  std::filesystem::create_directories("results");
  std::ofstream history;
  if (rank == 0) {
    history.exceptions(std::ios::failbit | std::ios::badbit);
    history.open("results/diagnostics.csv");
    history << "step,time,mean,rms,energy_density\n" << std::setprecision(17);
  }
  std::array<double, 3> values{};
  for (int step = 0; step <= s.steps; ++step) {
    values = solver.diagnostics();
    if (rank == 0)
      history << step << ',' << step * s.dt << ',' << values[0] << ',' << values[1]
              << ',' << values[2] << '\n';
    if (step != s.steps) solver.step();
  }
  solver.write_fields(rank);
  if (rank == 0) {
    std::ofstream summary("results/summary.json");
    summary.exceptions(std::ios::failbit | std::ios::badbit);
    summary << Json{{"n", s.n},
                    {"cutoff", s.cutoff},
                    {"ranks", ranks},
                    {"steps", s.steps},
                    {"time", s.steps * s.dt},
                    {"mean", values[0]},
                    {"rms", values[1]},
                    {"energy_density", values[2]},
                    {"normalization", "forward unscaled; inverse 1/(n*4*4)"}}
                   .dump(2)
            << '\n';
    std::cout << "Allen-Cahn ETD2 complete: " << s.steps << " steps on " << ranks
              << " rank(s); inspect results/ and run verify.py.\n";
  }
} // Solver and FFT objects are destroyed before MPI_Finalize.
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 2) throw std::invalid_argument("usage: allen_cahn input.json");
    run(read_settings(argv[1]), rank, ranks);
    MPI_Finalize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "nonlinear spectral example (rank " << rank << "): " << error.what()
              << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
