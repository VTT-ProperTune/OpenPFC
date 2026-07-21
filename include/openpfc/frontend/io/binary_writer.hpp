// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file binary_writer.hpp
 * @brief Binary format writer for raw field data output (frontend I/O
 * implementation)
 *
 * @details
 * BinaryWriter implements the kernel ResultsWriter interface for raw binary
 * output. It lives in frontend/io because it uses frontend utils (e.g. filename
 * formatting). The abstract ResultsWriter interface remains in kernel.
 *
 * @see kernel/simulation/results_writer.hpp for the ResultsWriter interface
 * @see frontend/utils/utils.hpp for format_with_number
 */

#ifndef PFC_BINARY_WRITER_HPP
#define PFC_BINARY_WRITER_HPP

#include <mpi.h>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pfc {

/**
 * @brief Binary format writer for raw field data output
 *
 * BinaryWriter implements ResultsWriter for raw binary output format. This
 * format is optimal for:
 * - Checkpointing and restart (exact data preservation)
 * - Large-scale simulations (minimal storage overhead)
 * - Fast I/O performance (no parsing or conversion)
 *
 * The binary format stores double or complex<double> values directly in native
 * byte order (platform-dependent). Files can be read back using BinaryReader
 * for simulation restart.
 *
 * @note MPI-IO collectives: `write()` uses `MPI_File_open`, `MPI_File_set_size`,
 * `MPI_File_set_view`, `MPI_File_write_all`, and `MPI_File_close`, which are
 * collective over the communicator passed to the constructor (default
 * `MPI_COMM_WORLD`). Every process in that communicator must call `write()` with
 * consistent `set_domain()` and must reach `MPI_File_close`;
 * skipping the call on some ranks will deadlock the job.
 *
 * @note Buffer vs domain: each rank's buffer length must equal the local brick
 * product from `set_domain`. A mismatch is fail-closed (communicator-wide
 * agreement via `MPI_Allreduce`) before `MPI_File_open` / `MPI_File_write_all`.
 *
 * @note Filetype/etype: `set_domain` only stores the decomposition. Before each
 * write, the MPI filetype is rebuilt from the same element type used as the
 * `MPI_File_set_view` etype (`MPI_DOUBLE` or `MPI_DOUBLE_COMPLEX`), so filetype
 * oldtype and etype always match.
 *
 * @see ResultsWriter - base class interface
 * @see BinaryReader - read binary files for restart
 *
 * Repository prose describing the on-disk layout and filename templates:
 * `docs/reference/binary_field_io_spec.md`.
 */
class BinaryWriter : public ResultsWriter {
private:
  MPI_Comm m_comm = MPI_COMM_WORLD;
  std::array<int, 3> m_global{};
  std::array<int, 3> m_local{};
  std::array<int, 3> m_offset{};
  bool m_domain_valid = false;
  MPI_Datatype m_filetype{};
  MPI_Datatype m_etype = MPI_DATATYPE_NULL;
  bool m_type_valid = false;

  static MPI_Datatype get_type([[maybe_unused]] const RealField &field) {
    return MPI_DOUBLE;
  }
  static MPI_Datatype get_type([[maybe_unused]] const ComplexField &field) {
    return MPI_DOUBLE_COMPLEX;
  }

  void ensure_filetype(MPI_Datatype etype) {
    if (!m_domain_valid) {
      throw std::runtime_error("BinaryWriter::write: set_domain() was not called");
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

public:
  ~BinaryWriter() override {
    if (m_type_valid) {
      (void)MPI_Type_free(&m_filetype);
    }
  }

  BinaryWriter(const BinaryWriter &) = delete;
  BinaryWriter &operator=(const BinaryWriter &) = delete;
  BinaryWriter(BinaryWriter &&) = delete;
  BinaryWriter &operator=(BinaryWriter &&) = delete;

  explicit BinaryWriter(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD)
      : ResultsWriter(filename), m_comm(comm) {}

  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) override {
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

  MPI_Status write(int increment, const RealField &data) override {
    return write_mpi_binary(increment, data);
  }

  MPI_Status write(int increment, const ComplexField &data) override {
    return write_mpi_binary(increment, data);
  }

  template <typename T>
  MPI_Status write_mpi_binary(int increment, const std::vector<T> &data) {
    const std::size_t expected = pfc::mpi::checked_local_extent_product(
        m_local, "BinaryWriter::write");

    int local_ok = 1;
    std::string error_msg;
    if (data.size() != expected) {
      std::ostringstream oss;
      oss << "BinaryWriter::write: buffer size mismatch (expected " << expected
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
          "BinaryWriter::write: collective buffer size mismatch on peer rank");
    }

    MPI_Datatype type = get_type(data);
    ensure_filetype(type);

    const int count =
        pfc::mpi::expect_mpi_io_count(data.size(), "BinaryWriter::write");

    MPI_File fh{};
    std::string filename2 = utils::format_with_number(m_filename, increment);
    pfc::mpi::throw_on_mpi_error(
        MPI_File_open(m_comm, const_cast<char *>(filename2.c_str()),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh),
        "MPI_File_open");
    pfc::mpi::MPI_File_guard file_guard(fh);

    MPI_Offset filesize = 0;
    MPI_Status status{};
    const unsigned int disp = 0;
    pfc::mpi::throw_on_mpi_error(MPI_File_set_size(fh, filesize),
                                 "MPI_File_set_size"); // truncate at offset 0
    pfc::mpi::throw_on_mpi_error(
        MPI_File_set_view(fh, disp, type, m_filetype, "native", MPI_INFO_NULL),
        "MPI_File_set_view");
    pfc::mpi::throw_on_mpi_error(
        MPI_File_write_all(fh, data.data(), count, type, &status),
        "MPI_File_write_all");

    int written = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Get_count(&status, type, &written),
                                 "MPI_Get_count");
    if (written != MPI_UNDEFINED && written != count) {
      std::ostringstream oss;
      oss << "Short write to \"" << filename2 << "\": wrote " << written
          << " elements, expected " << data.size();
      throw std::runtime_error(oss.str());
    }

    pfc::mpi::throw_on_mpi_error(MPI_File_close(&file_guard.file), "MPI_File_close");
    return status;
  }
};

} // namespace pfc

#endif // PFC_BINARY_WRITER_HPP
