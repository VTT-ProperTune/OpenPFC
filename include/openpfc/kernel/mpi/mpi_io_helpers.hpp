// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <mpi.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace pfc::mpi {

inline std::string mpi_error_string(int errcode) {
  char err[MPI_MAX_ERROR_STRING] = {};
  int errlen = 0;
  if (MPI_Error_string(errcode, err, &errlen) == MPI_SUCCESS && errlen > 0) {
    return std::string(err, err + errlen);
  }
  return "MPI error " + std::to_string(errcode);
}

inline void throw_on_mpi_error(int err, const char *what) {
  if (err == MPI_SUCCESS) {
    return;
  }
  std::ostringstream oss;
  oss << what << " failed: " << mpi_error_string(err);
  throw std::runtime_error(oss.str());
}

/**
 * @brief RAII guard for an MPI_File handle -- closes it unconditionally on
 * every exit path (including exceptions thrown after a successful
 * MPI_File_open), mirroring the MPI_Type_guard pattern in halo_mpi_types.hpp.
 */
struct MPI_File_guard {
  MPI_File file = MPI_FILE_NULL;
  MPI_File_guard() = default;
  explicit MPI_File_guard(MPI_File f) : file(f) {}
  ~MPI_File_guard() {
    if (file != MPI_FILE_NULL) {
      MPI_File_close(&file);
    }
  }
  MPI_File_guard(MPI_File_guard &&other) noexcept : file(other.file) {
    other.file = MPI_FILE_NULL;
  }
  MPI_File_guard &operator=(MPI_File_guard &&other) noexcept {
    if (this != &other) {
      if (file != MPI_FILE_NULL) {
        MPI_File_close(&file);
      }
      file = other.file;
      other.file = MPI_FILE_NULL;
    }
    return *this;
  }
  MPI_File_guard(const MPI_File_guard &) = delete;
  MPI_File_guard &operator=(const MPI_File_guard &) = delete;
};

} // namespace pfc::mpi
