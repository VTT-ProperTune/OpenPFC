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
 * @see ResultsWriter - base class interface
 * @see BinaryReader - read binary files for restart
 */
class BinaryWriter : public ResultsWriter {
  using ResultsWriter::ResultsWriter;

private:
  MPI_Datatype m_filetype{};
  bool m_type_valid = false;

  static MPI_Datatype get_type([[maybe_unused]] const RealField &field) {
    return MPI_DOUBLE;
  }
  static MPI_Datatype get_type([[maybe_unused]] const ComplexField &field) {
    return MPI_DOUBLE_COMPLEX;
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

  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) override {
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

  MPI_Status write(int increment, const RealField &data) override {
    return write_mpi_binary(increment, data);
  }

  MPI_Status write(int increment, const ComplexField &data) override {
    return write_mpi_binary(increment, data);
  }

  template <typename T>
  MPI_Status write_mpi_binary(int increment, const std::vector<T> &data) {
    if (!m_type_valid) {
      throw std::runtime_error("BinaryWriter::write: set_domain() was not called");
    }
    MPI_File fh{};
    std::string filename2 = utils::format_with_number(m_filename, increment);
    pfc::mpi::throw_on_mpi_error(
        MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filename2.c_str()),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh),
        "MPI_File_open");

    MPI_Offset filesize = 0;
    MPI_Status status{};
    const unsigned int disp = 0;
    MPI_Datatype type = get_type(data);
    pfc::mpi::throw_on_mpi_error(MPI_File_set_size(fh, filesize),
                                 "MPI_File_set_size"); // truncate at offset 0
    pfc::mpi::throw_on_mpi_error(
        MPI_File_set_view(fh, disp, type, m_filetype, "native", MPI_INFO_NULL),
        "MPI_File_set_view");
    pfc::mpi::throw_on_mpi_error(MPI_File_write_all(fh, data.data(),
                                                    static_cast<int>(data.size()),
                                                    type, &status),
                                 "MPI_File_write_all");

    int written = 0;
    pfc::mpi::throw_on_mpi_error(MPI_Get_count(&status, type, &written),
                                 "MPI_Get_count");
    if (written != MPI_UNDEFINED && written != static_cast<int>(data.size())) {
      pfc::mpi::throw_on_mpi_error(MPI_File_close(&fh), "MPI_File_close");
      std::ostringstream oss;
      oss << "Short write to \"" << filename2 << "\": wrote " << written
          << " elements, expected " << data.size();
      throw std::runtime_error(oss.str());
    }

    pfc::mpi::throw_on_mpi_error(MPI_File_close(&fh), "MPI_File_close");
    return status;
  }
};

} // namespace pfc

#endif // PFC_BINARY_WRITER_HPP
