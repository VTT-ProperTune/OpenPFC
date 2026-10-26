// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../equations.hpp"
#include "generated_rhs.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>

// This lesson supplies exact derivatives to isolate the pointwise generated law.
// Numerical derivatives and their errors are taught in the preceding chapter.
struct Located {
  coupled::Local derivatives;
  std::array<double, 3> position;
};

// start snippet generated_model
struct GeneratedModel {
  coupled::Parameters parameters;
  double source_rate;

  coupled::Increments rhs(double time, const Located &local) const noexcept {
    return coupled::generated::rhs<coupled::Increments>(
        time, local.position, local.derivatives, parameters, source_rate);
  }
};
// end snippet generated_model

// A model-owned evaluator adapts coordinates to the existing sweep contract.
struct ExactEvaluator {
  const pfc::data::Field<double> &grid;
  double time;
  void prepare() {} // analytic derivatives need no halos or transforms
  int imin() const { return 0; }
  int imax() const { return grid.local_size()[0]; }
  int jmin() const { return 0; }
  int jmax() const { return grid.local_size()[1]; }
  int kmin() const { return 0; }
  int kmax() const { return grid.local_size()[2]; }
  std::size_t idx(int i, int j, int k) const { return grid.idx(i, j, k); }
  Located operator()(int i, int j, int k) const {
    const auto x = grid.coords(i, j, k);
    return {coupled::exact_local(time, {x[0], x[1], x[2]}), {x[0], x[1], x[2]}};
  }
};

int main(int argc, char **argv) {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    if (comm.size() != 1)
      throw std::invalid_argument("This lesson uses one MPI process");
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 16);
    const double time = input.value("time", 0.75);
    const double rate = input.value("source_rate", 0.1);
    if (n < 8 || n > 64 || !std::isfinite(time) || std::abs(time) > 10 ||
        !std::isfinite(rate) || std::abs(rate) > 10)
      throw std::invalid_argument(
          "Use n in [8,64], finite |time|,|source_rate| <= 10");
    const double h = 2 * std::numbers::pi / n;
    const auto domain =
        pfc::domain::create(pfc::GridSize{{n, n, n}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{h, h, h}});
    const auto decomposition = pfc::decomposition::create(domain, 1);
    const auto grid = pfc::data::field_from_subdomain<double>(decomposition, 0, 0);
    // start snippet evaluate
    ExactEvaluator evaluator{grid, time};
    std::vector<double> du(grid.size()), dv(grid.size());
    pfc::sim::for_each_interior(GeneratedModel{{}, rate}, evaluator,
                                std::make_tuple(du.data(), dv.data()), time);
    // end snippet evaluate

    std::filesystem::create_directories("results");
    std::ofstream file("results/rates.csv");
    if (!file) throw std::runtime_error("Cannot write generated rates");
    file << std::setprecision(17) << "i,j,k,x,y,z,du,dv\n";
    grid.for_each_owned([&](int i, int j, int k) {
      const auto x = grid.coords(i, j, k);
      const auto at = grid.idx(i, j, k);
      if (!std::isfinite(du[at]) || !std::isfinite(dv[at]))
        throw std::runtime_error("Nonfinite generated rate");
      file << i << ',' << j << ',' << k << ',' << x[0] << ',' << x[1] << ',' << x[2]
           << ',' << du[at] << ',' << dv[at] << '\n';
    });
    if (!file) throw std::runtime_error("Cannot finish generated rates");
    std::cout << "Generated pointwise rates written at t=" << time << '\n';
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    return 1;
  }
}
