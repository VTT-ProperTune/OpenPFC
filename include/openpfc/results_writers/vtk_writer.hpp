// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file vtk_writer.hpp
 * @brief VTK ImageData format writer for visualization
 *
 * @details
 * VTKWriter outputs simulation fields in VTK ImageData format (.vti files),
 * which can be directly opened in ParaView, VisIt, or other VTK-compatible
 * visualization tools.
 *
 * Supports:
 * - Serial output (single .vti file)
 * - Parallel output (.pvti master + .vti pieces per rank)
 * - Real and complex field data
 * - Binary format (base64-encoded) for compact output
 *
 * @example
 * @code
 * using namespace pfc;
 *
 * // Create writer
 * auto writer = std::make_unique<VTKWriter>("output_%04d.vti");
 * writer->set_domain(global_size, local_size, local_offset);
 *
 * // Write field at each time step
 * for (int step = 0; step < 1000; ++step) {
 *     if (step % 100 == 0) {
 *         writer->write(step, density_field);
 *     }
 * }
 * @endcode
 *
 * @see ResultsWriter base class
 * @see examples/11_write_results.hpp for original implementation
 */

#ifndef PFC_VTK_WRITER_HPP
#define PFC_VTK_WRITER_HPP

#include "openpfc/results_writer.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace pfc {

/**
 * @brief VTK ImageData format writer
 *
 * Writes simulation fields to VTK ImageData (.vti) format for visualization.
 * In parallel runs, creates a .pvti master file and .vti piece files.
 */
class VTKWriter : public ResultsWriter {
private:
  std::array<int, 3> m_global_size;
  std::array<int, 3> m_local_size;
  std::array<int, 3> m_offset;
  std::array<double, 3> m_origin = {0.0, 0.0, 0.0};
  std::array<double, 3> m_spacing = {1.0, 1.0, 1.0};
  std::string m_field_name = "Field";
  int m_rank = 0;
  int m_num_ranks = 1;
  MPI_Comm m_comm = MPI_COMM_WORLD;

  /**
   * @brief Generate filename for a given increment and rank
   */
  std::string generate_filename(int increment, int rank = -1) const;

  /**
   * @brief Write VTK header XML
   */
  void write_vti_header(std::ofstream &file, int increment) const;

  /**
   * @brief Write VTK data section
   */
  void write_vti_data(std::ofstream &file, const RealField &data) const;

  /**
   * @brief Write parallel master file (.pvti)
   */
  void write_pvti_file(int increment) const;

public:
  /**
   * @brief Construct VTKWriter with filename pattern
   *
   * @param filename Filename pattern (e.g., "output_%04d.vti")
   */
  explicit VTKWriter(const std::string &filename) : ResultsWriter(filename) {
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_num_ranks);
  }

  /**
   * @brief Set domain decomposition
   */
  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) override;

  /**
   * @brief Set origin of the domain
   */
  void set_origin(const std::array<double, 3> &origin) { m_origin = origin; }

  /**
   * @brief Set spacing of the domain
   */
  void set_spacing(const std::array<double, 3> &spacing) { m_spacing = spacing; }

  /**
   * @brief Set field name for VTK output
   */
  void set_field_name(const std::string &name) { m_field_name = name; }

  /**
   * @brief Write real field to VTK file
   */
  MPI_Status write(int increment, const RealField &data) override;

  /**
   * @brief Write complex field to VTK file (writes magnitude)
   */
  MPI_Status write(int increment, const ComplexField &data) override;
};

} // namespace pfc

#endif // PFC_VTK_WRITER_HPP
