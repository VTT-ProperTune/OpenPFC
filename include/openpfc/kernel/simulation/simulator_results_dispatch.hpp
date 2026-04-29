// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulator_results_dispatch.hpp
 * @brief Dispatch `ResultsWriter::write` for all fields registered on a model
 *
 * @details
 * Shared by `Simulator::write_results` so the per-field write loop stays in one
 * place next to the `ResultsWriter` / `Model` field accessors.
 */

#ifndef PFC_KERNEL_SIMULATION_SIMULATOR_RESULTS_DISPATCH_HPP
#define PFC_KERNEL_SIMULATION_SIMULATOR_RESULTS_DISPATCH_HPP

#include <memory>
#include <string>
#include <unordered_map>

#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>

namespace pfc {

/**
 * @brief Write every registered results writer at the given file index
 *
 * For each `(field_name, writer)` entry, dispatches to real or complex field
 * accessors on `model` when that field exists.
 */
inline void write_results_for_registered_fields(
    Model &model,
    const std::unordered_map<std::string, std::unique_ptr<ResultsWriter>> &writers,
    int file_num) {
  for (const auto &[field_name, writer] : writers) {
    if (pfc::has_real_field(model, field_name)) {
      writer->write(file_num, pfc::get_real_field(model, field_name));
    }
    if (pfc::has_complex_field(model, field_name)) {
      writer->write(file_num, pfc::get_complex_field(model, field_name));
    }
  }
}

} // namespace pfc

#endif // PFC_KERNEL_SIMULATION_SIMULATOR_RESULTS_DISPATCH_HPP
