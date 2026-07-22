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
 * @note Buffer vs domain: each rank's buffer length must equal the local brick
 * product from `set_domain`. A mismatch is fail-closed (communicator-wide
 * agreement via `MPI_Allreduce`) before `MPI_File_open` / `MPI_File_read_all`.
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
#include <openpfc/kernel/mpi/domain_geometry.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

#include <sstream>
#include <stdexcept>
#include <string>

namespace pfc {

/**
 * @brief Read field data from binary files using MPI-IO
 *
 * BinaryReader reads field data from binary files with proper domain decomposition.
 * Each MPI rank reads only its local portion of the data.
 *
 * @note MPI-IO collectives: `read()` uses `MPI_File_open`, `MPI_File_set_view`,
 * `MPI_File_read_all`, and `MPI_File_close`, which are collective over the
 * communicator passed to the constructor (default `MPI_COMM_WORLD`). All ranks
 * must call `read()` together with matching `set_domain()` layout.
 *
 * @note Buffer vs domain: each rank's buffer length must equal the local brick
 * product from `set_domain`. A mismatch is fail-closed (communicator-wide
 * agreement via `MPI_Allreduce`) before `MPI_File_open` / `MPI_File_read_all`.
 *
 * @note Destructor: The destructor is noexcept(false) and fails closed on
 * MPI_Type_free errors via pfc::mpi::throw_on_mpi_error (same policy as
 * environment::~environment).
 */
class BinaryReader {

private:
  MPI_Comm m_comm = MPI_COMM_WORLD;
  Vec3<int> m_global{};
  Vec3<int> m_local{};
  Vec3<int> m_offset{};
  bool m_domain_valid = false;
  MPI_Datatype m_filetype{};
  MPI_Datatype m_etype = MPI_DATATYPE_NULL;
  bool m_type_valid = false;

  void ensure_filetype(MPI_Datatype etype) {
    if (!m_domain_valid) {
      throw std::runtime_error("BinaryReader::read: set_domain() was not called");
    }
    if (m_type_valid && m_etype == etype) {
      return;
    }
    if (m_type_valid) {
      pfc::mpi::throw_on_mpi_error(MPI_Type_free(&m_filetype), "MPI_Type_free");
      m_type_valid = false;
    }
    pfc::mpi::throw_on_mpi_error(
        MPI_Type_create_subarray(3, m_global.data(), m_local.data(), m_offset.data(),
                                 MPI_ORDER_FORTRAN, etype, &m_filetype),
        "MPI_Type_create_subarray");
    pfc::mpi::throw_on_mpi_error(MPI_Type_commit(&m_filetype), "MPI_Type_commit");
    m_etype = etype;
    m_type_valid = true;
  }

  template <typename T>
  MPI_Status read_mpi_binary(const std::string &filename, std::vector<T> &data,
                             MPI_Datatype etype, const char *element_name) {
    const std::size_t expected =
        pfc::mpi::checked_local_extent_product(m_local, "BinaryReader::read");

    int local_ok = 1;
    std::string error_msg;
    if (data.size() != expected) {
      std::ostringstream oss;
      oss << "BinaryReader::read: buffer size mismatch (expected " << expected
          << " elements from set_domain, got " << data.size() << ")";
      error_msg = oss.str();
      local_ok = 0;
    }

    int global_ok = 0;
    pfc::mpi::throw_on_mpi_error(
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_comm),
        "MPI_Allreduce on buffer size check");
    if (global_ok == 0) {
      if (!error_msg.empty()) {
        throw std::runtime_error(error_msg);
      }
      throw std::runtime_error(
          "BinaryReader::read: collective buffer size mismatch on peer rank");
    }

    ensure_filetype(etype);

    const int count =
        pfc::mpi::expect_mpi_io_count(data.size(), "BinaryReader::read");

    MPI_File fh{};
    pfc::mpi::throw_on_mpi_error(MPI_File_open(m_comm,
                                               const_cast<char *>(filename.c_str()),
                                               MPI_MODE_RDONLY, MPI_INFO_NULL, &fh),
                                 "MPI_File_open");
    pfc::mpi::MPI_File_guard file_guard(fh);

    pfc::mpi::throw_on_mpi_error(
        MPI_File_set_view(fh, 0, etype, m_filetype, "native", MPI_INFO_NULL),
        "MPI_File_set_view");

    MPI_Status status{};
    pfc::mpi::throw_on_mpi_error(
        MPI_File_read_all(fh, data.data(), count, etype, &status),
        "MPI_File_read_all");
    int received = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Get_count(&status, etype, &received),
                                 "MPI_Get_count");
    if (received != count) {
      std::ostringstream oss;
      oss << "Short read from \"" << filename << "\": got " << received << " "
          << element_name << ", expected " << data.size();
      throw std::runtime_error(oss.str());
    }

    pfc::mpi::throw_on_mpi_error(MPI_File_close(&file_guard.file), "MPI_File_close");
    return status;
  }

public:
  BinaryReader() = default;

  explicit BinaryReader(MPI_Comm comm) : m_comm(comm) {}

  BinaryReader(const BinaryReader &) = delete;
  BinaryReader &operator=(const BinaryReader &) = delete;
  BinaryReader(BinaryReader &&) = delete;
  BinaryReader &operator=(BinaryReader &&) = delete;

  ~BinaryReader() noexcept(false) {
    if (m_type_valid) {
      // Fail closed: silent MPI_Type_free errors mask corrupted MPI state.
      pfc::mpi::throw_on_mpi_error(MPI_Type_free(&m_filetype),
                                   "MPI_Type_free in ~BinaryReader");
    }
  }

  void set_domain(const Vec3<int> &arr_global, const Vec3<int> &arr_local,
                  const Vec3<int> &arr_offset) {
    pfc::mpi::validate_subarray_domain(arr_global, arr_local, arr_offset,
                                       "BinaryReader::set_domain");
    if (m_type_valid) {
      pfc::mpi::throw_on_mpi_error(MPI_Type_free(&m_filetype), "MPI_Type_free");
      m_type_valid = false;
      m_etype = MPI_DATATYPE_NULL;
    }
    m_global = arr_global;
    m_local = arr_local;
    m_offset = arr_offset;
    m_domain_valid = true;
  }

  MPI_Status read(const std::string &filename, Field &data) {
    return read_mpi_binary(filename, data, MPI_DOUBLE, "doubles");
  }

  MPI_Status read(const std::string &filename, ComplexField &data) {
    return read_mpi_binary(filename, data, MPI_DOUBLE_COMPLEX, "complex elements");
  }
};

} // namespace pfc

#endif // PFC_BINARY_READER_HPP
