// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    constexpr double dt = .05;
    constexpr int split = 3, steps = 8;
    const auto domain = pfc::domain::create(
        pfc::GridSize{{16, 4, 4}}, pfc::PhysicalOrigin{{0, 0, 0}},
        pfc::GridSpacing{{2 * std::numbers::pi / 16, 1, 1}});
    const auto decomposition = pfc::decomposition::create(domain, comm.size());
    auto make_state = [&](bool initialize) {
      pfc::SimulationState state;
      auto field =
          pfc::data::field_from_subdomain<double>(decomposition, comm.rank(), 0);
      field.apply([&](double x, double, double) {
        return initialize ? 1 + .2 * std::cos(x) : 0.;
      });
      state.add_field("u", std::move(field));
      return state;
    };
    // The evolution is a visible Euler update for u_t=-u, independent of I/O.
    auto advance = [&](auto &state, pfc::Time &time) {
      auto &u = state.template get_field<double>("u");
      u.for_each_owned([&](int i, int j, int k) { u(i, j, k) *= 1 - dt; });
      u.note_host_write();
      time.next();
    };
    auto uninterrupted = make_state(true);
    pfc::Time full_time({0, steps * dt, dt}, 0);
    for (int step = 0; step < steps; ++step) advance(uninterrupted, full_time);
    // start snippet checkpoint
    auto state = make_state(true);
    pfc::Time time({0, steps * dt, dt}, 0);
    for (int step = 0; step < split; ++step) advance(state, time);
    pfc::sim::CheckpointService checkpoints(
        {.every = 0, .directory = "results/checkpoints"}, comm);
    checkpoints.set_result_counter(7);
    checkpoints.save(state, time);

    auto restarted =
        make_state(false); // New fields and clock; no accepted values reused.
    pfc::Time restart_time({0, steps * dt, dt}, 0);
    pfc::sim::CheckpointService loader({}, comm);
    loader.load(restarted, restart_time, checkpoints.step_dir(split));
    for (int step = restart_time.get_increment(); step < steps; ++step)
      advance(restarted, restart_time);
    // end snippet checkpoint
    std::filesystem::create_directories("results");
    std::ofstream output("results/restart_rank_" + std::to_string(comm.rank()) +
                         ".csv");
    if (!output) throw std::runtime_error("Cannot open restart CSV");
    output << std::setprecision(17) << "i,j,k,x,uninterrupted,restarted\n";
    auto &u = uninterrupted.get_field<double>("u");
    auto &r = restarted.get_field<double>("u");
    double mismatch = 0;
    u.for_each_owned([&](int i, int j, int k) {
      const auto low = u.box().low;
      mismatch = std::max(mismatch, std::abs(u(i, j, k) - r(i, j, k)));
      output << i + low[0] << ',' << j + low[1] << ',' << k + low[2] << ','
             << (i + low[0]) * domain.spacing[0] << ',' << u(i, j, k) << ','
             << r(i, j, k) << '\n';
    });
    if (!output) throw std::runtime_error("Cannot write restart CSV");
    if (mismatch != 0 || loader.result_counter() != 7 ||
        restart_time.get_increment() != steps)
      throw std::runtime_error("Restart changed the discrete trajectory");
    if (comm.rank() == 0)
      std::cout << "checkpoint_step=3 final_step=8 result_counter="
                << loader.result_counter() << '\n';
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
