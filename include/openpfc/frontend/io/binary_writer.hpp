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
#include <openpfc/kernel/simulation/results_writer.hpp>

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
  MPI_Datatype m_filetype;

  static MPI_Datatype get_type(RealField) { return MPI_DOUBLE; }
  static MPI_Datatype get_type(ComplexField) { return MPI_DOUBLE_COMPLEX; }

public:
  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) override {
    MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                             arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                             &m_filetype);
    MPI_Type_commit(&m_filetype);
  }

  MPI_Status write(int increment, const RealField &data) override {
    return write_mpi_binary(increment, data);
  }

  MPI_Status write(int increment, const ComplexField &data) override {
    return write_mpi_binary(increment, data);
  }

  template <typename T>
  MPI_Status write_mpi_binary(int increment, const std::vector<T> &data) {
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

#endif // PFC_BINARY_WRITER_HPP
