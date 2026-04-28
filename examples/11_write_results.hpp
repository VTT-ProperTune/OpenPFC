// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_EXAMPLES_11_WRITE_RESULTS_HPP
#define PFC_EXAMPLES_11_WRITE_RESULTS_HPP

/**
 * @file 11_write_results.hpp
 * @brief Thin example wrapper around `pfc::VTKWriter` (legacy tutorial API)
 *
 * Older examples used a standalone MPI-VTK implementation here. That logic now
 * lives in `openpfc/frontend/io/vtk_writer.hpp` (`VTKWriter`). This header keeps the
 * small `VtkWriter<T>` surface (`set_uri`, `set_domain`, `initialize`, `write`) so
 * tutorial sources keep compiling while delegating all I/O to the library writer.
 */

#include <array>
#include <mpi.h>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace pfc {

/**
 * @brief Minimal VTK writer used by examples; delegates to VTKWriter.
 *
 * Only `T == double` is supported. For other scalar types or production code, use
 * `pfc::VTKWriter` directly.
 */
template <typename T> class VtkWriter {
private:
  std::string m_uri;
  std::array<int, 3> m_global{};
  std::array<int, 3> m_local{};
  std::array<int, 3> m_offset{};
  std::array<double, 3> m_origin{};
  std::array<double, 3> m_spacing{};
  std::string m_field_name = "default";
  bool m_have_domain = false;

public:
  explicit VtkWriter(MPI_Comm /* comm */ = MPI_COMM_WORLD) {}

  void set_uri(const std::string &uri) { m_uri = uri; }
  [[nodiscard]] const std::string &get_uri() const { return m_uri; }

  void set_domain(const std::array<int, 3> &global_dimensions,
                  const std::array<int, 3> &local_dimensions,
                  const std::array<int, 3> &offset) {
    m_global = global_dimensions;
    m_local = local_dimensions;
    m_offset = offset;
    m_have_domain = true;
  }

  void set_origin(const std::array<double, 3> &origin) { m_origin = origin; }
  void set_spacing(const std::array<double, 3> &spacing) { m_spacing = spacing; }
  void set_field_name(const std::string &field_name) { m_field_name = field_name; }

  /** @brief Validates configuration; VTKWriter builds MPI types on write. */
  void initialize() {
    if (m_uri.empty()) {
      throw std::runtime_error("VtkWriter: set_uri before initialize");
    }
    if (!m_have_domain) {
      throw std::runtime_error("VtkWriter: set_domain before initialize");
    }
  }

  void write(const std::vector<T> &data) {
    if constexpr (!std::is_same_v<T, double>) {
      throw std::runtime_error(
          "examples::VtkWriter only wraps VTKWriter for double; use pfc::VTKWriter "
          "otherwise");
    }
    if (m_uri.empty()) {
      throw std::runtime_error("VtkWriter: set_uri before write");
    }
    if (!m_have_domain) {
      throw std::runtime_error("VtkWriter: set_domain before write");
    }
    VTKWriter vtk(m_uri);
    vtk.set_domain(m_global, m_local, m_offset);
    vtk.set_origin(m_origin);
    vtk.set_spacing(m_spacing);
    vtk.set_field_name(m_field_name);
    (void)vtk.write(0, data);
  }
};

} // namespace pfc

#endif // PFC_EXAMPLES_11_WRITE_RESULTS_HPP
