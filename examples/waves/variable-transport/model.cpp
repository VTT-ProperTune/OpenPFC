// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <string>
#include <vector>

constexpr double two_pi = 2 * std::numbers::pi;
struct Derivative {
  double x{};
};

struct Transport {
  bool variable = true;
  bool gaussian = true;

  double speed(double x) const {
    return variable ? 0.2 + std::pow(std::sin(x - 1), 2) : 1;
  }
  double speed_derivative(double x) const {
    return variable ? std::sin(2 * (x - 1)) : 0;
  }
  double initial(double x) const {
    return gaussian ? std::exp(-100 * (x - 1) * (x - 1)) : std::cos(x);
  }
  double first(double x) const {
    return gaussian ? -200 * (x - 1) * initial(x) : -std::sin(x);
  }
  double second(double x) const {
    return gaussian ? (40000 * (x - 1) * (x - 1) - 200) * initial(x) : -std::cos(x);
  }

  // start snippet characteristic
  double exact(double x, double t) const {
    if (!variable) {
      double foot = std::fmod(x - t, two_pi);
      if (foot < 0) foot += two_pi;
      return initial(foot);
    }
    constexpr double a = 0.2;
    const double phase = std::atan2(std::sqrt(a + 1) * std::sin(x - 1),
                                    std::sqrt(a) * std::cos(x - 1));
    const double foot_phase = phase - std::sqrt(a * (a + 1)) * t;
    double foot = 1 + std::atan2(std::sqrt(a) * std::sin(foot_phase),
                                 std::sqrt(a + 1) * std::cos(foot_phase));
    if (foot < 0) foot += two_pi;
    return initial(foot);
  }
  // end snippet characteristic
};

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
    const int n = input.value("n", 128), steps = input.value("steps", 200);
    const double dt = input.value("dt", 0.0025);
    const auto coefficient = input.value("coefficient", std::string("variable"));
    const auto profile = input.value("initial", std::string("gaussian"));
    const auto startup = input.value("startup", std::string("taylor"));
    if ((coefficient != "variable" && coefficient != "constant") ||
        (profile != "gaussian" && profile != "cosine") ||
        (startup != "taylor" && startup != "book"))
      throw std::invalid_argument("Unknown transport option");
    const Transport model{coefficient == "variable", profile == "gaussian"};
    const double maximum_speed = model.variable ? 1.2 : 1;
    if (n < 16 || n > 512 || n % 2 || steps < 1 || steps > 10000 ||
        !std::isfinite(dt) || dt <= 0 || dt * maximum_speed * n / 2 >= 1) {
      throw std::invalid_argument(
          "Use even n in [16,512], 1..10000 steps and 0 < dt*cmax*n/2 < 1");
    }
    if (startup == "book" && (!model.gaussian || !model.variable)) {
      throw std::invalid_argument(
          "The book startup belongs to the variable-speed Gaussian preset");
    }
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{two_pi / n, 1, 1}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto &u = stack.u();
    u.apply([&](double x, double, double) { return model.initial(x); });
    auto derivative = pfc::field::create<Derivative>(u, stack.fft());
    std::vector<double> previous(u.size()), next(u.size());
    // start snippet startup
    u.for_each_owned([&](int i, int j, int k) {
      const double x = u.coords(i, j, k)[0], c = model.speed(x);
      previous[u.idx(i, j, k)] =
          startup == "book" ? model.initial(x - 0.2 * two_pi / (4 * n))
                            : model.initial(x) + dt * c * model.first(x) +
                                  0.5 * dt * dt * c *
                                      (model.speed_derivative(x) * model.first(x) +
                                       c * model.second(x));
    });
    // end snippet startup
    std::filesystem::create_directories("results");
    std::ofstream output("results/transport_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    output << std::setprecision(17) << "step,time,x,u,exact\n";
    auto observe = [&](int step) {
      u.for_each_owned([&](int i, int j, int k) {
        if (j + u.box().low[1] != 0 || k + u.box().low[2] != 0) return;
        const double x = u.coords(i, j, k)[0];
        output << step << ',' << step * dt << ',' << x << ',' << u(i, j, k) << ','
               << model.exact(x, step * dt) << '\n';
      });
    };
    observe(0);
    // start snippet leapfrog
    for (int step = 1; step <= steps; ++step) {
      derivative.prepare();
      u.for_each_owned([&](int i, int j, int k) {
        const double x = u.coords(i, j, k)[0];
        next[u.idx(i, j, k)] = previous[u.idx(i, j, k)] -
                               2 * dt * model.speed(x) * derivative(i, j, k).x;
      });
      previous = u.vec();
      std::copy(next.begin(), next.end(), u.vec().begin());
      observe(step);
    }
    // end snippet leapfrog
    if (!output) throw std::runtime_error("Cannot write transport CSV");
    if (comm.rank() == 0)
      std::cout << "Transport t=" << steps * dt << " with " << startup
                << " startup\n";
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
