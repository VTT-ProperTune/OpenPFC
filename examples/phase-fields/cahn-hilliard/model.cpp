// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

// start snippet physics
struct CahnHilliard {
  pfc::Domain domain;
  pfc::Box3i box;
  double mobility, gamma, cubic;

  void declare_fields(pfc::SimulationState &state) const {
    pfc::sim::add_declared_field<double>(state, "psi", domain, box, 0);
  }
  double linear_symbol(double laplacian) const {
    return -mobility * laplacian * (1 + gamma * laplacian);
  }
  double nonlinear_symbol(double laplacian) const { return mobility * laplacian; }
  struct Local {
    double cubic;
    double nonlinearity(const pfc::sim::SpectralCell &cell) const {
      return cubic * cell.psi * cell.psi * cell.psi;
    }
  };
  Local pointwise() const { return {cubic}; }
};
// end snippet physics

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 64), steps = input.value("steps", 1000);
    const int save_every = input.value("save_every", 50);
    const double length = input.value("length", 4.0), dt = input.value("dt", .001);
    const double mobility = input.value("mobility", 1.0),
                 gamma = input.value("gamma", .04);
    const double mean = input.value("mean", -.1),
                 amplitude = input.value("amplitude", .05);
    const double cubic = input.value("cubic", 1.0);
    const auto initial = input.value("initial", std::string("mixture"));
    if (n < 8 || n > 128 || n % 2 || steps < 1 || steps > 2000 || save_every < 1 ||
        !std::isfinite(length) || length <= 0 || !std::isfinite(dt) || dt <= 0 ||
        dt > .01 || !std::isfinite(mobility) || mobility <= 0 || mobility > 10 ||
        !std::isfinite(gamma) || gamma <= 0 || gamma > 1 || !std::isfinite(mean) ||
        std::abs(mean) > .5 || !std::isfinite(amplitude) || amplitude < 0 ||
        amplitude > .2 || !std::isfinite(cubic) || cubic < 0 || cubic > 1 ||
        (initial != "mixture" && initial != "mode" && initial != "constant"))
      throw std::invalid_argument(
          "Use bounded grid/time/model parameters; see input.json");
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, n, 1}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{length / n, length / n, 1}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    CahnHilliard physics{domain, stack.u().box(), mobility, gamma, cubic};
    // start snippet state
    pfc::SimulationState state;
    physics.declare_fields(state);
    auto &u = state.get_field<double>("psi");
    u.apply([&](double x, double y, double) {
      const double a = 2 * std::numbers::pi * x / length;
      const double b = 2 * std::numbers::pi * y / length;
      return mean +
             (initial == "constant"
                  ? 0
                  : amplitude * (initial == "mode" ? std::cos(a) * std::cos(b)
                                                   : std::cos(a) + std::sin(2 * b) +
                                                         std::cos(2 * a + b)));
    });
    pfc::sim::SpectralETDSystem system(physics, stack.fft(), state, dt);
    // end snippet state
    auto &fft = stack.fft();
    std::vector<std::complex<double>> modes(fft.size_outbox());
    std::vector<double> k2(modes.size()), minus_laplacian(u.size());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          k2[i] = kx * kx + ky * ky + kz * kz;
        });
    std::filesystem::create_directories("results");
    std::ofstream fields("results/fields_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    fields.exceptions(std::ios::badbit | std::ios::failbit);
    fields << std::setprecision(17) << "step,time,x,y,u\n";
    std::ofstream diagnostics;
    if (comm.rank() == 0) {
      diagnostics.open("results/diagnostics.csv");
      diagnostics.exceptions(std::ios::badbit | std::ios::failbit);
      diagnostics << std::setprecision(17) << "step,time,mean,energy\n";
    }
    const auto observe = [&](int step) {
      fft.forward(u.vec(), modes);
      for (std::size_t i = 0; i < modes.size(); ++i) modes[i] *= k2[i];
      fft.backward(modes, minus_laplacian);
      double local[2]{}, global[2]{};
      for (std::size_t i = 0; i < u.size(); ++i) {
        const double v = u.vec()[i];
        if (!std::isfinite(v))
          throw std::runtime_error("Non-finite phase field; reduce dt");
        local[0] += v;
        local[1] += .25 * cubic * v * v * v * v - .5 * v * v + .25 +
                    .5 * gamma * v * minus_laplacian[i];
      }
      MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, comm);
      if (comm.rank() == 0)
        diagnostics << step << ',' << step * dt << ',' << global[0] / (n * n) << ','
                    << global[1] / (n * n) << '\n';
      u.for_each_owned([&](double x, double y, double, double value) {
        fields << step << ',' << step * dt << ',' << x << ',' << y << ',' << value
               << '\n';
      });
    };
    observe(0);
    // start snippet evolve
    for (int step = 1; step <= steps; ++step) {
      system.step((step - 1) * dt);
      if (step % save_every == 0 || step == steps) observe(step);
    }
    // end snippet evolve
    if (comm.rank() == 0) std::cout << "ETD1 phase field t=" << steps * dt << '\n';
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
