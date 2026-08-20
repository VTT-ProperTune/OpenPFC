// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file file_results_writer.hpp
 * @brief `ResultsWriter` intermediate for file sinks with increment templating
 *
 * @details
 * Kernel `ResultsWriter` is format-agnostic (file, stdout, in-memory). Filename
 * patterns such as `output_%04d.bin` belong here so a non-file sink does not
 * need a dummy path. `BinaryWriter`, `VTKWriter`, and `HDF5Writer` derive from
 * this type.
 */

#include <string>
#include <utility>

#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>

namespace pfc {

class FileResultsWriter : public ResultsWriter {
public:
  explicit FileResultsWriter(std::string filename_pattern)
      : m_filename(std::move(filename_pattern)) {}

  [[nodiscard]] const std::string &filename_pattern() const noexcept {
    return m_filename;
  }

  [[nodiscard]] std::string formatted_path(int increment) const {
    return utils::format_with_number(m_filename, increment);
  }

protected:
  std::string m_filename;
};

} // namespace pfc
