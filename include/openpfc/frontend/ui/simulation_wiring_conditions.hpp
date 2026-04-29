// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_conditions.hpp
 * @brief JSON-driven initial and boundary conditions
 */

#ifndef PFC_UI_SIMULATION_WIRING_CONDITIONS_HPP
#define PFC_UI_SIMULATION_WIRING_CONDITIONS_HPP

#include <memory>
#include <sstream>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/simulation_wiring_detail.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {

inline void
add_initial_conditions_from_json(Simulator &sim, const nlohmann::json &settings,
                                 MPI_Comm comm, int mpi_rank, bool rank0,
                                 const FieldModifierCatalog &modifier_catalog =
                                     default_field_modifier_catalog()) {
  const pfc::Logger lg{pfc::LogLevel::Info, mpi_rank};
  if (!settings.contains("initial_conditions")) {
    if (rank0) {
      pfc::log_warning(lg, "no initial conditions are set!");
    }
    return;
  }
  if (rank0) {
    pfc::log_info(lg, "Adding initial conditions");
  }
  for (const nlohmann::json &params : settings["initial_conditions"]) {
    if (rank0) {
      std::ostringstream ps;
      ps << params;
      pfc::log_info(lg,
                    std::string("Creating initial condition from data ") + ps.str());
    }
    if (!params.contains("type")) {
      if (rank0) {
        pfc::log_warning(lg, "no type is set for initial condition!");
      }
      continue;
    }
    std::string type = params["type"];
    auto field_modifier = create_field_modifier(type, params, modifier_catalog);
    detail::configure_field_modifier_targets_from_json(*field_modifier, params, lg,
                                                       rank0, "initial condition");
    field_modifier->set_mpi_comm(comm);
    sim.add_initial_conditions(std::move(field_modifier));
  }
}

inline void
add_boundary_conditions_from_json(Simulator &sim, const nlohmann::json &settings,
                                  MPI_Comm comm, int mpi_rank, bool rank0,
                                  const FieldModifierCatalog &modifier_catalog =
                                      default_field_modifier_catalog()) {
  const pfc::Logger lg{pfc::LogLevel::Info, mpi_rank};
  if (!settings.contains("boundary_conditions")) {
    if (rank0) {
      pfc::log_warning(lg, "no boundary conditions are set!");
    }
    return;
  }
  if (rank0) {
    pfc::log_info(lg, "Adding boundary conditions");
  }
  for (const nlohmann::json &params : settings["boundary_conditions"]) {
    if (rank0) {
      std::ostringstream ps;
      ps << params;
      pfc::log_info(lg, std::string("Creating boundary condition from data ") +
                            ps.str());
    }
    if (!params.contains("type")) {
      if (rank0) {
        pfc::log_warning(lg, "no type is set for boundary condition!");
      }
      continue;
    }
    std::string type = params["type"];
    auto field_modifier = create_field_modifier(type, params, modifier_catalog);
    detail::configure_field_modifier_targets_from_json(*field_modifier, params, lg,
                                                       rank0, "boundary condition");
    field_modifier->set_mpi_comm(comm);
    sim.add_boundary_conditions(std::move(field_modifier));
  }
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_CONDITIONS_HPP
