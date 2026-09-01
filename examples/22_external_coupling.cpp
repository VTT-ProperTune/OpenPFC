// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <mpi.h>

#include "22_external_coupling.hpp"

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/simulation/coupling.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/time.hpp>

/** \example 22_external_coupling.cpp
 *
 * Mock FEM orchestrator that owns the time loop. It duplicates
 * `MPI_COMM_WORLD` so OpenPFC halo tags cannot collide with the host
 * application, pulls a `pfc::coupling::FieldHandle` (read export), negotiates
 * `dt` with `Time::clip_attempt_dt`, and imposes a time-varying source through
 * a FieldModifier-shaped adapter on `SimulationState`.
 *
 * Run: `mpirun -np 2 ./22_external_coupling`
 */

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  {
    pfc::mpi::communicator world;
    auto isolated = world.duplicate();
    const int rank = isolated.rank();
    const int nproc = isolated.size();

    auto domain = pfc::domain::create(pfc::GridSize({8, 8, 1}),
                                      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                      pfc::GridSpacing({1.0, 1.0, 1.0}));
    auto decomp = pfc::decomposition::create(domain, nproc);
    auto field = pfc::data::field_from_subdomain<double>(decomp, rank, 0);
    field.apply([](double, double, double) { return 0.0; });

    pfc::SimulationState state;
    state.add_field("u", std::move(field));

    pfc::Time time({0.0, 0.4, 0.1}, 0.0);
    openpfc_examples::HostSourceModifier source;

    int steps = 0;
    while (!time.done() && steps < 20) {
      const double fem_dt = 0.25;
      const double dt = time.clip_attempt_dt(fem_dt);
      time.begin_attempt(dt);

      auto handle = pfc::coupling::export_host_field(state, "u");
      source.apply(state, time.get_accepted_time());

      time.commit_attempt();
      ++steps;

      if (rank == 0 && steps == 1) {
        std::cout << "coupling: field=" << handle.name
                  << " memory=" << handle.memory_space
                  << " owned=" << handle.owned_box.size[0] << "x"
                  << handle.owned_box.size[1] << "x" << handle.owned_box.size[2]
                  << " clipped_dt=" << dt << "\n";
      }
    }

    const auto &u = state.get_field<double>("u");
    if (rank == 0) {
      std::cout << "coupling done: t=" << time.get_current()
                << " increment=" << time.get_increment()
                << " u(0,0,0)=" << u(0, 0, 0) << "\n";
    }

  } // communicators freed before MPI_Finalize
  MPI_Finalize();
  return 0;
}
