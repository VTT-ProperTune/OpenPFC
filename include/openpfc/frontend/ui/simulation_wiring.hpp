// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring.hpp
 * @brief Connect JSON settings to Simulator (writers, ICs, BCs)
 *
 * @details
 * Shared helpers used by `App::main()` (and available for other drivers) to
 * register binary result writers and field modifiers from parsed settings.
 */

#ifndef PFC_UI_SIMULATION_WIRING_HPP
#define PFC_UI_SIMULATION_WIRING_HPP

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/utils/logging.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>

namespace pfc::ui {

/**
 * @brief Ensure parent directory of a writer path exists (rank 0 only)
 *
 * @param output File or URI path used by a results writer
 * @param mpi_rank Rank for log attribution
 * @return true if a new directory was created, false if it already existed
 */
inline bool ensure_results_parent_dir_for_writer(const std::string &output,
                                                 int mpi_rank) {
  const pfc::Logger lg{pfc::LogLevel::Info, mpi_rank};
  std::filesystem::path results_dir(output);
  if (results_dir.has_filename()) {
    results_dir = results_dir.parent_path();
  }
  if (!std::filesystem::exists(results_dir)) {
    pfc::log_info(lg, std::string("Results dir ") + results_dir.string() +
                          " does not exist, creating");
    std::filesystem::create_directories(results_dir);
    return true;
  }
  pfc::log_warning(lg, std::string("results dir ") + results_dir.string() +
                           " already exists");
  return false;
}

inline void add_result_writers_from_json(Simulator &sim,
                                         const nlohmann::json &settings,
                                         MPI_Comm comm, int mpi_rank, bool rank0) {
  const pfc::Logger lg{pfc::LogLevel::Info, mpi_rank};
  if (rank0) {
    pfc::log_info(lg, "Adding results writers");
  }
  if (settings.contains("saveat") && settings.contains("fields") &&
      settings["saveat"] > 0) {
    for (const auto &field : settings["fields"]) {
      std::string name = field["name"];
      std::string data = field["data"];
      if (rank0) {
        (void)ensure_results_parent_dir_for_writer(data, mpi_rank);
      }
      if (rank0) {
        pfc::log_info(lg, "Writing field " + name + " to " + data);
      }
      sim.add_results_writer(name, std::make_unique<BinaryWriter>(data, comm));
    }
  } else {
    if (rank0) {
      pfc::log_warning(lg, "not writing results to anywhere.");
      pfc::log_info(lg, "To write results, add ResultsWriter to model.");
    }
  }
}

inline void add_initial_conditions_from_json(Simulator &sim,
                                             const nlohmann::json &settings,
                                             MPI_Comm comm, int mpi_rank,
                                             bool rank0) {
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
    auto field_modifier = create_field_modifier(type, params);
    if (!params.contains("target")) {
      if (rank0) {
        pfc::log_warning(
            lg, "no target is set for initial condition! Using target 'default'");
      }
    } else {
      const auto &target = params["target"];
      if (target.is_array()) {
        std::vector<std::string> names;
        names.reserve(target.size());
        for (const auto &el : target) {
          names.push_back(el.get<std::string>());
        }
        field_modifier->set_field_names(std::move(names));
        if (rank0) {
          pfc::log_info(lg, "Setting initial condition targets (multi-field)");
        }
      } else {
        auto t = target.get<std::string>();
        if (rank0) {
          pfc::log_info(lg, std::string("Setting initial condition target to ") + t);
        }
        field_modifier->set_field_name(t);
      }
    }
    field_modifier->set_mpi_comm(comm);
    sim.add_initial_conditions(std::move(field_modifier));
  }
}

inline void add_boundary_conditions_from_json(Simulator &sim,
                                              const nlohmann::json &settings,
                                              MPI_Comm comm, int mpi_rank,
                                              bool rank0) {
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
    auto field_modifier = create_field_modifier(type, params);
    if (!params.contains("target")) {
      if (rank0) {
        pfc::log_warning(
            lg, "no target is set for boundary condition! Using target 'default'");
      }
    } else {
      const auto &target = params["target"];
      if (target.is_array()) {
        std::vector<std::string> names;
        names.reserve(target.size());
        for (const auto &el : target) {
          names.push_back(el.get<std::string>());
        }
        field_modifier->set_field_names(std::move(names));
        if (rank0) {
          pfc::log_info(lg, "Setting boundary condition targets (multi-field)");
        }
      } else {
        auto t = target.get<std::string>();
        if (rank0) {
          pfc::log_info(lg,
                        std::string("Setting boundary condition target to ") + t);
        }
        field_modifier->set_field_name(t);
      }
    }
    field_modifier->set_mpi_comm(comm);
    sim.add_boundary_conditions(std::move(field_modifier));
  }
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_HPP
