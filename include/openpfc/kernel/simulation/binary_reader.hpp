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
 * @date 2025
 */

#ifndef PFC_BINARY_READER_HPP
#define PFC_BINARY_READER_HPP

#include <mpi.h>
#include <openpfc/kernel/data/model_types.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

namespace pfc {

namespace detail {

inline std::string mpi_error_string(int errcode) {
  char err[MPI_MAX_ERROR_STRING] = {};
  int errlen = 0;
  if (MPI_Error_string(errcode, err, &errlen) == MPI_SUCCESS && errlen > 0) {
    return std::string(err, err + errlen);
  }
  return "MPI error " + std::to_string(errcode);
}

} // namespace detail

class BinaryReader {

private:
  MPI_Datatype m_filetype{};
  bool m_type_valid = false;

  static void check_mpi(int err, const char *what) {
    if (err == MPI_SUCCESS) {
      return;
    }
    std::ostringstream oss;
    oss << what << " failed: " << detail::mpi_error_string(err);
    throw std::runtime_error(oss.str());
  }

public:
  BinaryReader() = default;

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
      check_mpi(MPI_Type_free(&m_filetype), "MPI_Type_free");
      m_type_valid = false;
    }
    check_mpi(MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                                       arr_offset.data(), MPI_ORDER_FORTRAN,
                                       MPI_DOUBLE, &m_filetype),
              "MPI_Type_create_subarray");
    check_mpi(MPI_Type_commit(&m_filetype), "MPI_Type_commit");
    m_type_valid = true;
  }

  MPI_Status read(const std::string &filename, Field &data) {
    if (!m_type_valid) {
      throw std::runtime_error("BinaryReader::read: set_domain() was not called");
    }
    MPI_File fh{};
    int ierr = MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filename.c_str()),
                             MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ierr != MPI_SUCCESS) {
      std::ostringstream oss;
      oss << "Unable to open \"" << filename
          << "\": " << detail::mpi_error_string(ierr);
      throw std::runtime_error(oss.str());
    }

    check_mpi(
        MPI_File_set_view(fh, 0, MPI_DOUBLE, m_filetype, "native", MPI_INFO_NULL),
        "MPI_File_set_view");

    MPI_Status status{};
    check_mpi(MPI_File_read_all(fh, data.data(), static_cast<int>(data.size()),
                                MPI_DOUBLE, &status),
              "MPI_File_read_all");
    int received = 0;
    check_mpi(MPI_Get_count(&status, MPI_DOUBLE, &received), "MPI_Get_count");
    if (received != static_cast<int>(data.size())) {
      (void)MPI_File_close(&fh);
      std::ostringstream oss;
      oss << "Short read from \"" << filename << "\": got " << received
          << " doubles, expected " << data.size();
      throw std::runtime_error(oss.str());
    }

    check_mpi(MPI_File_close(&fh), "MPI_File_close");
    return status;
  }
};

} // namespace pfc

#endif // PFC_BINARY_READER_HPP
