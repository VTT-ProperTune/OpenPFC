// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file results_writer_catalog.hpp
 * @brief Type string → `ResultsWriter` factory for JSON wiring (OCP / extension)
 *
 * @details
 * JSON `fields[]` entries can specify `"writer": "<type>"` (default `"binary"`).
 * `add_result_writers_from_json` requires a **`ResultsWriterCatalog`** argument
 * (no default); pass `default_results_writer_catalog()` for built-in `binary`,
 * `vtk`, and `hdf5` when `OPENPFC_HAS_HDF5`. Unknown types are a hard error at
 * `create_writer` (same
 * `format_config_error` shape as `FieldModifierCatalog::create_modifier`).
 * Applications and tests inject a custom catalog to register additional writer
 * types without editing `simulation_wiring_writers.hpp`.
 */

#ifndef PFC_UI_RESULTS_WRITER_CATALOG_HPP
#define PFC_UI_RESULTS_WRITER_CATALOG_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <openpfc/frontend/io/binary_writer.hpp>
#ifdef OPENPFC_HAS_HDF5
#include <openpfc/frontend/io/hdf5_writer.hpp>
#endif
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/ui/errors_config_format.hpp>
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
  void register_writer(std::string type, ResultsWriterCreateFn fn) {
    m_factories[std::move(type)] = std::move(fn);
  }

  /**
   * @brief Instantiate a writer for @p type and output @p path
   * @throw std::invalid_argument if @p type is not registered
   */
  [[nodiscard]] std::unique_ptr<pfc::ResultsWriter>
  create_writer(const std::string &type, const std::string &path, MPI_Comm comm,
                std::string_view field_name = {}) const {
    const auto it = m_factories.find(type);
    if (it != m_factories.end()) {
      return it->second(path, comm);
    }
    const std::string what = field_name.empty()
                                 ? std::string("results writer type")
                                 : std::string("results writer type for field '") +
                                       std::string(field_name) + "'";
    throw std::invalid_argument(
        format_config_error("writer", what, "a registered catalog key", type,
                            registered_writer_types(), "\"writer\": \"binary\""));
  }

  [[nodiscard]] bool has_type(const std::string &type) const {
    return m_factories.contains(type);
  }

  [[nodiscard]] std::vector<std::string> registered_writer_types() const {
    std::vector<std::string> names;
    names.reserve(m_factories.size());
    for (const auto &[k, _] : m_factories) {
      names.push_back(k);
    }
    std::sort(names.begin(), names.end());
    return names;
  }

private:
  std::unordered_map<std::string, ResultsWriterCreateFn> m_factories;
};

/** @brief Built-in catalog: `binary` → `BinaryWriter`, `vtk` → `VTKWriter`,
 *         `hdf5` → `HDF5Writer` when HDF5 is enabled. */
[[nodiscard]] inline ResultsWriterCatalog make_builtin_results_writer_catalog() {
  ResultsWriterCatalog c;
  c.register_writer(
      "binary",
      [](std::string path, MPI_Comm comm) -> std::unique_ptr<pfc::ResultsWriter> {
        return std::make_unique<pfc::BinaryWriter>(std::move(path), comm);
      });
  c.register_writer(
      "vtk",
      [](std::string path, MPI_Comm comm) -> std::unique_ptr<pfc::ResultsWriter> {
        return std::make_unique<pfc::VTKWriter>(std::move(path), comm);
      });
#ifdef OPENPFC_HAS_HDF5
  c.register_writer(
      "hdf5",
      [](std::string path, MPI_Comm comm) -> std::unique_ptr<pfc::ResultsWriter> {
        return std::make_unique<pfc::HDF5Writer>(std::move(path), comm);
      });
#endif
  return c;
}

/** @brief Process-wide default catalog (built-ins only unless extended in tests). */
[[nodiscard]] inline ResultsWriterCatalog &default_results_writer_catalog() {
  static ResultsWriterCatalog instance = make_builtin_results_writer_catalog();
  return instance;
}

} // namespace pfc::ui

#endif // PFC_UI_RESULTS_WRITER_CATALOG_HPP
