// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file results_writer_catalog.hpp
 * @brief Type string → `ResultsWriter` factory for JSON wiring (OCP / extension)
 *
 * @details
 * JSON `fields[]` entries can specify `"writer": "<type>"` (default `"binary"`).
 * Applications and tests inject a custom `ResultsWriterCatalog` to register
 * additional writer types without editing `simulation_wiring_writers.hpp`.
 */

#ifndef PFC_UI_RESULTS_WRITER_CATALOG_HPP
#define PFC_UI_RESULTS_WRITER_CATALOG_HPP

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <mpi.h>
#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>

namespace pfc::ui {

using ResultsWriterCreateFn = std::function<std::unique_ptr<pfc::ResultsWriter>(
    std::string path, MPI_Comm comm)>;

/**
 * @brief Maps JSON `writer` type strings to `ResultsWriter` factories
 */
class ResultsWriterCatalog {
public:
  /** @brief Register or replace a writer factory for @p type (case-sensitive). */
  void register_writer_type(std::string type, ResultsWriterCreateFn fn) {
    m_factories[std::move(type)] = std::move(fn);
  }

  /**
   * @brief Instantiate a writer for @p type and output @p path
   * @return Writer on success, `std::nullopt` if @p type is unknown
   */
  [[nodiscard]] std::optional<std::unique_ptr<pfc::ResultsWriter>>
  try_create(const std::string &type, const std::string &path, MPI_Comm comm) const {
    const auto it = m_factories.find(type);
    if (it == m_factories.end()) {
      return std::nullopt;
    }
    return it->second(path, comm);
  }

  [[nodiscard]] bool has_type(const std::string &type) const {
    return m_factories.find(type) != m_factories.end();
  }

private:
  std::unordered_map<std::string, ResultsWriterCreateFn> m_factories;
};

/** @brief Built-in catalog: `binary` → `pfc::BinaryWriter` */
[[nodiscard]] inline ResultsWriterCatalog make_builtin_results_writer_catalog() {
  ResultsWriterCatalog c;
  c.register_writer_type(
      "binary",
      [](std::string path, MPI_Comm comm) -> std::unique_ptr<pfc::ResultsWriter> {
        return std::make_unique<pfc::BinaryWriter>(std::move(path), comm);
      });
  return c;
}

/** @brief Process-wide default catalog (built-ins only unless extended in tests). */
[[nodiscard]] inline ResultsWriterCatalog &default_results_writer_catalog() {
  static ResultsWriterCatalog instance = make_builtin_results_writer_catalog();
  return instance;
}

} // namespace pfc::ui

#endif // PFC_UI_RESULTS_WRITER_CATALOG_HPP
