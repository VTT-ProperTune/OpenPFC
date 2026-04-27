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

} // namespace pfc::mpi
