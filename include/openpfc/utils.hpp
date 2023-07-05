#pragma once

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

template <typename T, std::size_t D> std::string array_to_string(const std::array<T, D> &arr) {
  std::ostringstream oss;
  oss << '{';
  for (std::size_t i = 0; i < D; ++i) {
    oss << arr[i];
    if (i != D - 1) oss << ", ";
  }
  oss << '}';
  return oss.str();
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
