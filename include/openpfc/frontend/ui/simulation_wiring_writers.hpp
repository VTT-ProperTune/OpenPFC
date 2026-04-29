// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_writers.hpp
 * @brief JSON-driven result writers (`saveat`, `fields`)
 *
 * @details
 * Each `fields[]` object uses `ResultsWriterCatalog` (default: `binary` →
 * `pfc::BinaryWriter`). Optional `"writer"` string selects the catalog key.
 */

#ifndef PFC_UI_SIMULATION_WIRING_WRITERS_HPP
#define PFC_UI_SIMULATION_WIRING_WRITERS_HPP

#include <filesystem>
#include <memory>
#include <string>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
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

inline void add_result_writers_from_json(
    Simulator &sim, const nlohmann::json &settings, const JsonWiringContext &ctx,
    const ResultsWriterCatalog &writer_catalog = default_results_writer_catalog()) {
  const pfc::Logger lg{pfc::LogLevel::Info, ctx.mpi_rank};
  if (ctx.rank0) {
    pfc::log_info(lg, "Adding results writers");
  }
  if (settings.contains("saveat") && settings.contains("fields") &&
      settings["saveat"] > 0) {
    for (const auto &field : settings["fields"]) {
      std::string name = field["name"];
      std::string data = field["data"];
      std::string writer_type = "binary";
      if (field.contains("writer") && field["writer"].is_string()) {
        writer_type = field["writer"].get<std::string>();
      }
      if (ctx.rank0) {
        (void)ensure_results_parent_dir_for_writer(data, ctx.mpi_rank);
      }
      if (ctx.rank0) {
        pfc::log_info(lg, "Writing field " + name + " to " + data +
                              " (writer: " + writer_type + ")");
      }
      auto writer_opt = writer_catalog.try_create(writer_type, data, ctx.comm);
      if (!writer_opt) {
        if (ctx.rank0) {
          pfc::log_warning(lg, "Unknown results writer type '" + writer_type +
                                   "' for field '" + name + "' — skipping");
        }
        continue;
      }
      sim.add_results_writer(name, std::move(*writer_opt));
    }
  } else {
    if (ctx.rank0) {
      pfc::log_warning(lg, "not writing results to anywhere.");
      pfc::log_info(lg, "To write results, add ResultsWriter to model.");
    }
  }
}

inline void add_result_writers_from_json(
    Simulator &sim, const nlohmann::json &settings, MPI_Comm comm, int mpi_rank,
    bool rank0,
    const ResultsWriterCatalog &writer_catalog = default_results_writer_catalog()) {
  add_result_writers_from_json(
      sim, settings, JsonWiringContext{comm, mpi_rank, rank0}, writer_catalog);
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_WRITERS_HPP
