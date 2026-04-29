// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_writers.hpp
 * @brief JSON-driven binary result writers (`saveat`, `fields`)
 */

#ifndef PFC_UI_SIMULATION_WIRING_WRITERS_HPP
#define PFC_UI_SIMULATION_WIRING_WRITERS_HPP

#include <filesystem>
#include <memory>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/utils/logging.hpp>

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

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_WRITERS_HPP
