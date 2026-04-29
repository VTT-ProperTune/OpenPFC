// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file binary_reader.hpp
 * @brief Read field data from binary files
 *
 * @details
 * This file defines the BinaryReader class for reading field data from binary
 * files using MPI-IO. Used for:
 * - Loading checkpoints
 * - Reading initial conditions from files
 * - Restarting simulations
 *
 * @note MPI-IO collectives: `read()` uses `MPI_File_open`, `MPI_File_set_view`,
 * `MPI_File_read_all`, and `MPI_File_close`, which are collective over the
 * communicator passed to the constructor (default `MPI_COMM_WORLD`). All ranks
 * must call `read()` together with the same
 * filename and matching `set_domain()` layout; otherwise the program may hang.
 *
 * The reader handles parallel I/O with proper domain decomposition, allowing
 * each MPI rank to read only its local portion of the data.
 *
 * Usage:
 * @code
 * pfc::BinaryReader reader;
 * reader.set_domain(global_size, local_size, local_offset);
 * reader.read("checkpoint.bin", field_data);
 * @endcode
 *
 * @see results_writer.hpp for writing binary files
 * @see initial_conditions/file_reader.hpp for using this in initial conditions
 *
 * @author OpenPFC Contributors
 * @date 2026
 *
 * @see
 * https://github.com/VTT-ProperTune/OpenPFC/blob/master/docs/postprocess_binary_fields.md
 * (post-processing raw binary fields outside OpenPFC)
 */

#ifndef PFC_BINARY_READER_HPP
#define PFC_BINARY_READER_HPP

#include <mpi.h>
#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

namespace pfc {

class BinaryReader {

private:
  MPI_Comm m_comm = MPI_COMM_WORLD;
  MPI_Datatype m_filetype{};
  bool m_type_valid = false;

public:
  BinaryReader() = default;

  explicit BinaryReader(MPI_Comm comm) : m_comm(comm) {}

  BinaryReader(const BinaryReader &) = delete;
  BinaryReader &operator=(const BinaryReader &) = delete;
  BinaryReader(BinaryReader &&) = delete;
  BinaryReader &operator=(BinaryReader &&) = delete;

  ~BinaryReader() {
    if (m_type_valid) {
      (void)MPI_Type_free(&m_filetype);
    }
  }

  void set_domain(const Vec3<int> &arr_global, const Vec3<int> &arr_local,
                  const Vec3<int> &arr_offset) {
    if (m_type_valid) {
      pfc::mpi::throw_on_mpi_error(MPI_Type_free(&m_filetype), "MPI_Type_free");
      m_type_valid = false;
    }
    pfc::mpi::throw_on_mpi_error(
        MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                                 arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                                 &m_filetype),
        "MPI_Type_create_subarray");
    pfc::mpi::throw_on_mpi_error(MPI_Type_commit(&m_filetype), "MPI_Type_commit");
    m_type_valid = true;
  }

  MPI_Status read(const std::string &filename, Field &data) {
    if (!m_type_valid) {
      throw std::runtime_error("BinaryReader::read: set_domain() was not called");
    }
    MPI_File fh{};
    pfc::mpi::throw_on_mpi_error(MPI_File_open(m_comm,
                                               const_cast<char *>(filename.c_str()),
                                               MPI_MODE_RDONLY, MPI_INFO_NULL, &fh),
                                 "MPI_File_open");

    pfc::mpi::throw_on_mpi_error(
        MPI_File_set_view(fh, 0, MPI_DOUBLE, m_filetype, "native", MPI_INFO_NULL),
        "MPI_File_set_view");

    MPI_Status status{};
    pfc::mpi::throw_on_mpi_error(MPI_File_read_all(fh, data.data(),
                                                   static_cast<int>(data.size()),
                                                   MPI_DOUBLE, &status),
                                 "MPI_File_read_all");
    int received = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Get_count(&status, MPI_DOUBLE, &received),
                                 "MPI_Get_count");
    if (received != static_cast<int>(data.size())) {
      (void)MPI_File_close(&fh);
      std::ostringstream oss;
      oss << "Short read from \"" << filename << "\": got " << received
          << " doubles, expected " << data.size();
      throw std::runtime_error(oss.str());
    }

    pfc::mpi::throw_on_mpi_error(MPI_File_close(&fh), "MPI_File_close");
    return status;
  }
};

} // namespace pfc

#endif // PFC_BINARY_READER_HPP
