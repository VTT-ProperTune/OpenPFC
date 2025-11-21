// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file results_writer.hpp
 * @brief Base interface for simulation output and I/O operations
 *
 * @details
 * This file defines the ResultsWriter abstract base class, which provides
 * a unified interface for writing simulation results to various output formats.
 * OpenPFC currently supports:
 * - Binary format (BinaryWriter) for checkpointing and restart
 * - Future: VTK format for visualization (User Story #0025)
 *
 * The ResultsWriter interface supports:
 * - Parallel I/O via MPI (distributed data from decomposed domains)
 * - Writing real and complex fields
 * - Timestep-indexed output (increment parameter)
 * - Domain configuration (global/local sizes, offsets)
 *
 * Typical usage:
 * @code
 * // Create writer
 * auto writer = std::make_unique<pfc::BinaryWriter>("output.bin");
 *
 * // Configure domain (once)
 * writer->set_domain(global_size, local_size, local_offset);
 *
 * // Write data at each save step
 * for (int step = 0; step < num_steps; ++step) {
 *     writer->write(step, field_data);
 * }
 * @endcode
 *
 * This file is part of the I/O module, providing output capabilities for
 * simulation results and checkpointing.
 *
 * @see simulator.hpp for I/O orchestration
 * @see types.hpp for RealField and ComplexField definitions
 * @see binary_reader.hpp for input counterpart
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_RESULTS_WRITER_HPP
#define PFC_RESULTS_WRITER_HPP

#include "types.hpp"
#include "utils.hpp"
#include <array>
#include <iostream>
#include <mpi.h>
#include <vector>

namespace pfc {

class ResultsWriter {
public:
  ResultsWriter(const std::string &filename) { m_filename = filename; }
  virtual ~ResultsWriter() = default;

  virtual void set_domain(const std::array<int, 3> &arr_global,
                          const std::array<int, 3> &arr_local,
                          const std::array<int, 3> &arr_offset) = 0;

  virtual MPI_Status write(int increment, const RealField &data) = 0;

  virtual MPI_Status write(int increment, const ComplexField &data) = 0;

  template <typename T> MPI_Status write(const std::vector<T> &data) {
    return write(0, data);
  }

protected:
  std::string m_filename;
};

class BinaryWriter : public ResultsWriter {
  using ResultsWriter::ResultsWriter;

private:
  MPI_Datatype m_filetype;

  static MPI_Datatype get_type(RealField) { return MPI_DOUBLE; }
  static MPI_Datatype get_type(ComplexField) { return MPI_DOUBLE_COMPLEX; }

public:
  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) {
    MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                             arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                             &m_filetype);
    MPI_Type_commit(&m_filetype);
  };

  MPI_Status write(int increment, const RealField &data) {
    return write_(increment, data);
  }

  MPI_Status write(int increment, const ComplexField &data) {
    return write_(increment, data);
  }

  template <typename T>
  MPI_Status write_(int increment, const std::vector<T> &data) {
    MPI_File fh;
    std::string filename2 = utils::format_with_number(m_filename, increment);
    MPI_File_open(MPI_COMM_WORLD, filename2.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    MPI_Status status;
    const unsigned int disp = 0;
    MPI_Datatype type = get_type(data);
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, type, m_filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, data.data(), data.size(), type, &status);
    MPI_File_close(&fh);
    return status;
  }
};
} // namespace pfc

#endif // PFC_RESULTS_WRITER_HPP
