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
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/nodal_dirichlet_multigrid.hpp>

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    if (comm.size() > 2)
      throw std::invalid_argument("This bounded example supports one or two ranks");
    if (argc > 2) throw std::invalid_argument("Use model [input.json]");
    nlohmann::json input = nlohmann::json::object();
    if (argc == 2) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 15);
    const double diffusion = input.value("diffusion", 1.0);
    const double reaction = input.value("reaction", 0.0);
    if ((n != 15 && n != 31) || !std::isfinite(diffusion) || diffusion <= 0 ||
        diffusion > 2 || !std::isfinite(reaction) || reaction < 0 || reaction > 5)
      throw std::invalid_argument("Use n=15/31, 0<diffusion<=2 and 0<=reaction<=5");
    const double h = 1.0 / (n + 1), pi = std::numbers::pi;

    // start snippet layout
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, n, n}}, pfc::PhysicalOrigin{{h, h, h}},
                            pfc::GridSpacing{{h, h, h}}, {false, false, false});
    const auto decomposition =
        pfc::decomposition::create(domain, pfc::Int3{comm.size(), 1, 1});
    const auto box = pfc::decomposition::local_box(decomposition, comm.rank());
    using Operations = pfc::solvers::FieldOperations<pfc::HostSpace>;
    Operations operations(domain, box, 1, 1, comm);
    auto rhs = operations.make(operations.size());
    auto solution = operations.make(operations.size());
    // end snippet layout

    // start snippet load
    auto sine = [=](double x, double y, double z) {
      return std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
    };
    const double beta = 2 * (1 - std::cos(pi * h)) / (h * pi * pi);
    const double load = (3 * diffusion * pi * pi + reaction) * beta * beta * beta;
    rhs[0].apply([&](double x, double y, double z) { return load * sine(x, y, z); });
    // end snippet load

    // start snippet solve
    using Multigrid = pfc::solvers::NodalDirichletMultigrid;
    Multigrid::Controls settings;
    settings.diffusion = diffusion;
    settings.reaction = reaction;
    Multigrid hierarchy(decomposition, comm.rank(), comm, settings);
    pfc::solvers::KrylovControls controls;
    controls.relative_tolerance = 1e-12;
    controls.maximum_iterations = 120;
    const auto status = pfc::solvers::cg(
        operations,
        [&](const auto &x, auto &out) { hierarchy.apply_operator(x[0], out[0]); },
        rhs, solution, controls,
        [&](const auto &r, auto &out, int) { hierarchy.apply(r[0], out[0]); });
    if (!status.converged())
      throw std::runtime_error("Q1 multigrid-CG did not converge");
    // end snippet solve

    std::filesystem::create_directories("results");
    if (comm.rank() == 0) {
      std::ofstream metadata("results/run.json");
      if (!metadata) throw std::runtime_error("Cannot open solver metadata");
      metadata << std::setw(2)
               << nlohmann::json{{"n", n},
                                 {"h", h},
                                 {"diffusion", diffusion},
                                 {"reaction", reaction},
                                 {"ranks", comm.size()},
                                 {"levels", hierarchy.levels()},
                                 {"coarse_unknowns", hierarchy.coarse_unknowns()},
                                 {"iterations", status.iterations},
                                 {"residual", status.residual}}
               << '\n';
      if (!metadata) throw std::runtime_error("Cannot write solver metadata");
    }
    std::ofstream field("results/field_rank_" + std::to_string(comm.rank()) +
                        ".csv");
    if (!field) throw std::runtime_error("Cannot open local nodal output");
    field << std::setprecision(17) << "x,y,z,u,continuum\n";
    solution[0].for_each_owned([&](double x, double y, double z, double value) {
      field << x << ',' << y << ',' << z << ',' << value << ',' << sine(x, y, z)
            << '\n';
    });
    if (!field) throw std::runtime_error("Cannot write local nodal output");
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
