// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/field/sbp_diffusion.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

namespace {

struct Parameters {
  int n = 33;
  double diffusivity = 0.2;
  double amplitude = 0.5;
  double final_time = 0.1;
  std::string boundary = "flux";

  static Parameters read(int argc, char **argv) {
    if (argc > 2) throw std::invalid_argument("Use model [input.json]");
    nlohmann::json input = nlohmann::json::object();
    if (argc == 2) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    Parameters p{input.value("n", 33), input.value("diffusivity", 0.2),
                 input.value("amplitude", 0.5), input.value("final_time", 0.1),
                 input.value("boundary", std::string("flux"))};
    if (p.n < 17 || p.n > 65 || !std::isfinite(p.diffusivity) ||
        p.diffusivity <= 0 || p.diffusivity > 1 || !std::isfinite(p.amplitude) ||
        std::abs(p.amplitude) > 1 || !std::isfinite(p.final_time) ||
        p.final_time <= 0 || p.final_time > 0.2 ||
        (p.boundary != "flux" && p.boundary != "derivative"))
      throw std::invalid_argument(
          "Use 17<=n<=65, 0<D<=1, |A|<=1, 0<T<=0.2 and flux/derivative");
    return p;
  }
};

} // namespace

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    if (comm.size() != 1)
      throw std::invalid_argument("This axis example uses one MPI rank");
    const auto p = Parameters::read(argc, argv);
    const double h = 1.0 / (p.n - 1);
    const pfc::field::fd::DiffusionAxis axis(p.n, h);
    std::vector<double> u(p.n), next(p.n);
    auto initial = [&](double x) { return 1 + p.amplitude * x * (1 - x); };
    for (int i = 0; i < p.n; ++i) u[i] = initial(i * h);

    // start snippet boundary
    using pfc::field::fd::BoundaryQuantity;
    const bool prescribe_flux = p.boundary == "flux";
    const pfc::field::fd::FaceCondition face{
        prescribe_flux ? BoundaryQuantity::ConstitutiveFlux
                       : BoundaryQuantity::NormalDerivative,
        [=](double) {
          return prescribe_flux ? p.diffusivity * p.amplitude : -p.amplitude;
        }};
    // end snippet boundary

    // The reader callbacks intentionally expose the one-dimensional axis API.
    // This example owns all global nodes; no halo or hidden reduction is needed.
    auto mass = [&] {
      double sum = 0;
      for (int i = 0; i < p.n; ++i) sum += axis.weight(i) * u[i];
      return sum;
    };
    const double initial_mass = mass();
    const double max_dt = 0.02 * h * h / p.diffusivity;
    const int steps = static_cast<int>(std::ceil(p.final_time / max_dt));
    const double dt = p.final_time / steps;
    pfc::field::fd::FluxLedger ledger;

    std::filesystem::create_directories("results");
    std::ofstream diagnostics("results/budget.csv");
    if (!diagnostics) throw std::runtime_error("Cannot open budget CSV");
    diagnostics << std::setprecision(17)
                << "step,time,mass,outward_integral,balance_error\n";
    auto observe = [&](int step) {
      diagnostics << step << ',' << step * dt << ',' << mass() << ','
                  << ledger.accepted() << ','
                  << mass() + ledger.accepted() - initial_mass << '\n';
    };
    observe(0);

    // start snippet advance
    for (int step = 0; step < steps; ++step) {
      const double time = step * dt;
      for (int i = 0; i < p.n; ++i) {
        const double rhs = axis.apply(
            i, [&](int j) { return u[j]; }, [&](int) { return p.diffusivity; }, time,
            &face, &face);
        next[i] = u[i] + dt * rhs;
      }
      ledger.begin();
      ledger.stage(face.flux(time, p.diffusivity), 1.0, dt);
      ledger.stage(face.flux(time, p.diffusivity), 1.0, dt);
      u.swap(next);
      ledger.accept();
      observe(step + 1);
    }
    // end snippet advance

    std::ofstream field("results/field.csv"), metadata("results/run.json");
    if (!field || !metadata) throw std::runtime_error("Cannot open field/metadata");
    field << std::setprecision(17) << "i,x,weight,initial,u\n";
    for (int i = 0; i < p.n; ++i)
      field << i << ',' << i * h << ',' << axis.weight(i) << ',' << initial(i * h)
            << ',' << u[i] << '\n';
    metadata << std::setw(2)
             << nlohmann::json{{"n", p.n},
                               {"diffusivity", p.diffusivity},
                               {"amplitude", p.amplitude},
                               {"final_time", p.final_time},
                               {"boundary", p.boundary},
                               {"steps", steps},
                               {"dt", dt}}
             << '\n';
    if (!field || !metadata || !diagnostics)
      throw std::runtime_error("Cannot write example outputs");
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    return 1;
  }
}
