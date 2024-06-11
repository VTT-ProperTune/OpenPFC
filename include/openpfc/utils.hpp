/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#ifndef PFC_UTILS_HPP
#define PFC_UTILS_HPP

#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfc {
namespace utils {

template <typename... Args> std::string string_format(const std::string &format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

std::string format_with_number(const std::string &filename, int increment) {
  if (filename.find('%') != std::string::npos) {
    return utils::string_format(filename, increment);
  } else {
    return filename;
  }
}

template <typename T> size_t sizeof_vec(std::vector<T> &V) {
  return V.size() * sizeof(T);
}

} // namespace utils

namespace mpi {

int get_comm_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int get_comm_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

} // namespace mpi

} // namespace pfc

#endif
