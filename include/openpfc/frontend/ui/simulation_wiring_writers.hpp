// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_wiring_writers.hpp
 * @brief JSON-driven result writers (`saveat`, `fields`)
 *
 * @details
 * Each `fields[]` object uses `ResultsWriterCatalog` (default: `binary` →
 * `pfc::BinaryWriter`). Optional `"writer"` string selects the catalog key.
 * Returns named writers; sessions attach them on `on_save`.
 */

#ifndef PFC_UI_SIMULATION_WIRING_WRITERS_HPP
#define PFC_UI_SIMULATION_WIRING_WRITERS_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::ui {

struct NamedResultsWriter {
  std::string field_name;
  std::unique_ptr<pfc::ResultsWriter> writer;
};

/**
 * @brief Ensure parent directory of a writer path exists (rank 0 only)
 *
 * @param output File or URI path used by a results writer
 * @param mpi_rank Rank for log attribution
 * @return true if a new directory was created, false if it already existed
 */
inline bool ensure_results_parent_dir_for_writer(const std::string &output,
                                                 int mpi_rank) {
  if (mpi_rank != 0) {
    return false;
  }

  const pfc::Logger lg{pfc::LogLevel::Info, mpi_rank};
  std::filesystem::path results_dir(output);
  if (results_dir.has_filename()) {
    results_dir = results_dir.parent_path();
  }
  if (results_dir.empty()) {
    return false;
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

[[nodiscard]] inline std::vector<NamedResultsWriter>
parse_result_writers_from_json(const nlohmann::json &settings,
                               const JsonWiringContext &ctx,
                               const ResultsWriterCatalog &writer_catalog) {
  const pfc::Logger lg{pfc::LogLevel::Info, ctx.mpi_rank};
  if (ctx.rank0) {
    pfc::log_info(lg, "Adding results writers");
  }
  std::vector<NamedResultsWriter> writers;
  if (settings.contains("saveat") && settings.contains("fields") &&
      settings["saveat"] > 0) {
    for (const auto &field : settings["fields"]) {
      std::string name = field["name"];
      std::string data = field["data"];
      std::string writer_type = "binary";
      if (field.contains("writer") && field["writer"].is_string()) {
        writer_type = field["writer"].get<std::string>();
      }
      auto writer = writer_catalog.create_writer(writer_type, data, ctx.comm, name);
      if (ctx.rank0) {
        (void)ensure_results_parent_dir_for_writer(data, ctx.mpi_rank);
        pfc::log_info(lg, "Writing field " + name + " to " + data +
                              " (writer: " + writer_type + ")");
      }
      writers.push_back(NamedResultsWriter{std::move(name), std::move(writer)});
    }
  } else if (ctx.rank0) {
    pfc::log_warning(lg, "not writing results to anywhere.");
    pfc::log_info(lg, "To write results, add a ResultsWriter via JSON fields[].");
  }
  return writers;
}

} // namespace pfc::ui

#endif // PFC_UI_SIMULATION_WIRING_WRITERS_HPP
