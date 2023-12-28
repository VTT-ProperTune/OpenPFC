#pragma once

#include "types.hpp"
#include <mpi.h>

#include <iostream>

namespace pfc {

class BinaryReader {

private:
  MPI_Datatype m_filetype;

public:
  void set_domain(const Vec3<int> &arr_global, const Vec3<int> &arr_local, const Vec3<int> &arr_offset) {
    MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(), arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                             &m_filetype);
    MPI_Type_commit(&m_filetype);
  };

  MPI_Status read(const std::string &filename, Field &data) {
    MPI_File fh;
    MPI_Status status;
    if (MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)) {
      std::cout << "Unable to open file!" << std::endl;
    }
    MPI_File_set_view(fh, 0, MPI_DOUBLE, m_filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, data.data(), data.size(), MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    return status;
  }
};

} // namespace pfc
