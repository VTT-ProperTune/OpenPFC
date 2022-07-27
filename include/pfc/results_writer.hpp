#pragma once

#include <array>
#include <iostream>
#include <mpi.h>
#include <pfc/utils.hpp>
#include <vector>

namespace pfc {

class ResultsWriter {
public:
  ResultsWriter(const std::string &filename) { m_filename = filename; }

  virtual void set_domain(const std::array<int, 3> &arr_global,
                          const std::array<int, 3> &arr_local,
                          const std::array<int, 3> &arr_offset) = 0;

  virtual MPI_Status write(int increment, const std::vector<double> &data) = 0;

  MPI_Status write(const std::vector<double> &data) { return write(0, data); }

protected:
  std::string m_filename;

  static std::string format(const std::string &filename, int increment) {
    if (filename.find('%') != std::string::npos) {
      return utils::string_format(filename, increment);
    } else {
      return filename;
    }
  }
};

class BinaryWriter : public ResultsWriter {
  using ResultsWriter::ResultsWriter;

private:
  MPI_Datatype m_filetype;

public:
  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) {
    MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                             arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                             &m_filetype);
    MPI_Type_commit(&m_filetype);
  };

  MPI_Status write(int increment, const std::vector<double> &data) {
    MPI_File fh;
    std::string filename2 = format(m_filename, increment);
    MPI_File_open(MPI_COMM_WORLD, filename2.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    MPI_Status status;
    const unsigned int disp = 0;
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, MPI_DOUBLE, m_filetype, "native",
                      MPI_INFO_NULL);
    MPI_File_write_all(fh, data.data(), data.size(), MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    return status;
  }
};
} // namespace pfc
