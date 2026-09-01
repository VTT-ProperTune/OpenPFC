// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file coupling.hpp
 * @brief Stable field-handle export for external solvers (M11).
 *
 * `pfc::coupling::FieldHandle` is a non-owning snapshot of one host field
 * (name, view, owned box, spacing, origin, memory space). The driver loop
 * `pfc::sim::run` is already a free function an external orchestrator can
 * own. Inject a duplicated communicator (`mpi::communicator::duplicate`)
 * when the host application shares `MPI_COMM_WORLD` and must not collide
 * on OpenPFC halo tags.
 */

#include <string>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

namespace pfc::coupling {

struct FieldHandle {
  std::string name;
  pfc::field::FieldView<double> view;
  pfc::Box3i owned_box{};
  pfc::Real3 spacing{};
  pfc::Real3 origin{};
  const char *memory_space = "host";
};

[[nodiscard]] inline FieldHandle export_host_field(SimulationState &state,
                                                   const std::string &name) {
  auto &f = state.get_field<double>(name);
  const auto sz = f.local_size();
  const pfc::types::Int3 extents{sz[0], sz[1], sz[2]};
  FieldHandle h;
  h.name = name;
  h.view = pfc::field::FieldView<double>(f.data(), f.size(), extents, f.spacing(),
                                         f.origin());
  h.owned_box = f.box();
  h.spacing = f.spacing();
  h.origin = f.origin();
  h.memory_space = "host";
  return h;
}

} // namespace pfc::coupling
