// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/solvers/microelasticity/homogenization.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    if (argc > 2) throw std::invalid_argument("Use model [input.json]");
    nlohmann::json input = nlohmann::json::object();
    if (argc == 2) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 8);
    const double amplitude = input.value("amplitude", 0.3);
    if (n < 8 || n > 32 || n % 2 || !std::isfinite(amplitude) || amplitude < 0 ||
        amplitude > 0.5)
      throw std::invalid_argument("Use even 8<=n<=32 and 0<=amplitude<=0.5");
    const double h = 2 * std::numbers::pi / n;
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, n, n}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{h, h, h}});
    pfc::sim::stacks::SpectralCPUStack stack(domain, comm.rank(), comm.size());
    auto &weight = stack.u();

    // start snippet layers
    const double soft = 1 / (1 + amplitude), hard = 1 / (1 - amplitude);
    weight.apply([&](double x, double, double) {
      const double mu = 1 / (1 + amplitude * std::cos(x));
      return amplitude == 0 ? 0.0 : (mu - soft) / (hard - soft);
    });
    pfc::solvers::MicroelasticityParams parameters;
    parameters.stiffness_at_zero = pfc::solvers::Stiffness::from_lame(soft, soft);
    parameters.stiffness_at_one = pfc::solvers::Stiffness::from_lame(hard, hard);
    parameters.relative_tolerance = 1e-11;
    parameters.max_iterations = 200;
    parameters.comm = comm;
    // end snippet layers

    // start snippet homogenize
    pfc::solvers::PeriodicHomogenizer homogenizer(domain, stack.fft(), parameters);
    const auto effective = homogenizer.compute(weight);
    if (!effective.all_converged())
      throw std::runtime_error("A unit-load elasticity solve did not converge");
    // end snippet homogenize

    std::filesystem::create_directories("results");
    if (comm.rank() == 0) {
      std::ofstream matrix("results/stiffness.csv"), metadata("results/run.json");
      if (!matrix || !metadata)
        throw std::runtime_error("Cannot open tensor output");
      matrix << std::setprecision(17)
             << "component,load,stiffness,mean_stress,mean_strain,iterations,"
                "residual\n";
      for (int load = 0; load < 6; ++load)
        for (int component = 0; component < 6; ++component) {
          const auto &report = effective.reports[load];
          matrix << component << ',' << load << ','
                 << effective.stiffness(component, load) << ','
                 << effective.mean_stress[load][component] << ','
                 << effective.mean_strain[load][component] << ','
                 << report.iterations << ',' << report.residual << '\n';
        }
      metadata << std::setw(2)
               << nlohmann::json{{"n", n},
                                 {"amplitude", amplitude},
                                 {"ranks", comm.size()},
                                 {"tolerance", 1e-11},
                                 {"voigt_order",
                                  {"11", "22", "33", "23", "13", "12"}}}
               << '\n';
      if (!matrix || !metadata)
        throw std::runtime_error("Cannot write tensor output");
    }

    // start snippet shear
    const auto &strain =
        homogenizer.strain(pfc::solvers::SYM_XY)[pfc::solvers::SYM_XY];
    const auto &stress = homogenizer.solver().stress()[pfc::solvers::SYM_XY];
    std::ofstream field("results/field_rank_" + std::to_string(comm.rank()) +
                        ".csv");
    if (!field) throw std::runtime_error("Cannot open local shear field");
    field << std::setprecision(17) << "x,y,z,weight,epsilon_xy,sigma_xy\n";
    strain.for_each_owned([&](int i, int j, int k) {
      const auto x = strain.coords(i, j, k);
      field << x[0] << ',' << x[1] << ',' << x[2] << ',' << weight(i, j, k) << ','
            << strain(i, j, k) << ',' << stress(i, j, k) << '\n';
    });
    // end snippet shear
    if (!field) throw std::runtime_error("Cannot write local shear field");
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
