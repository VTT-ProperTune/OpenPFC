// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "source.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/coupling.hpp>
#include <openpfc/kernel/simulation/time.hpp>

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator world;
  try {
    // start snippet session
    auto isolated = world.duplicate();
    const auto domain =
        pfc::domain::create(pfc::GridSize{{8, 8, 2}}, pfc::PhysicalOrigin{{0, 0, 0}},
                            pfc::GridSpacing{{.25, .5, 1}});
    const auto decomposition = pfc::decomposition::create(domain, isolated.size());
    pfc::SimulationState state;
    state.add_field("u", pfc::data::field_from_subdomain<double>(
                             decomposition, isolated.rank(), 0));
    state.get_field<double>("u").apply([](double, double, double) { return 0.; });
    pfc::Time time({0, .4, .1}, .15);
    openpfc_examples::HostSourceModifier source;
    // end snippet session
    std::filesystem::create_directories("results");
    std::ofstream output("results/coupling_rank_" + std::to_string(world.rank()) +
                         ".csv");
    if (!output) throw std::runtime_error("Cannot open coupling CSV");
    output << std::setprecision(17) << "step,t0,dt,time,increment,read_sum\n";
    // start snippet orchestrate
    while (!time.done()) {
      const double t0 = time.get_current();
      time.begin_attempt(std::min(.25, time.get_dt())); // Host and OpenPFC limits.
      const double dt = time.get_attempted_dt();
      const auto exported = pfc::coupling::export_host_field(state, "u");
      double local = 0, global = 0;
      for (std::size_t i = 0; i < exported.view.size(); ++i)
        local += exported.view.data()[i];
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, isolated);
      source.apply(state, t0 + dt);
      time.commit_attempt();
      time.increment_step_success();
      output << time.get_increment() << ',' << t0 << ',' << dt << ','
             << time.get_current() << ',' << time.get_increment() << ',' << global
             << '\n';
    }
    // end snippet orchestrate
    auto &u = state.get_field<double>("u");
    std::ofstream field("results/field_rank_" + std::to_string(world.rank()) +
                        ".csv");
    if (!field) throw std::runtime_error("Cannot open field CSV");
    field << std::setprecision(17) << "x,y,z,u\n";
    u.for_each_owned([&](double x, double y, double z, double value) {
      field << x << ',' << y << ',' << z << ',' << value << '\n';
    });
    if (!field || !output) throw std::runtime_error("Cannot write coupling CSV");
  } catch (const std::exception &error) {
    if (world.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(world, 1);
    return 1;
  }
}
