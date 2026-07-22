// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <mpi.h>

#include <array>
#include <climits>
#include <cstddef>
#include <limits>
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
 *
 * Destructor is noexcept(false) and fails closed via throw_on_mpi_error
 * (same policy as environment::~environment).
 */
struct MPI_File_guard {
  MPI_File file = MPI_FILE_NULL;
  MPI_File_guard() = default;
  explicit MPI_File_guard(MPI_File f) : file(f) {}
  ~MPI_File_guard() noexcept(false) {
    if (file != MPI_FILE_NULL) {
      // Fail closed: silent MPI_File_close errors mask I/O / MPI state issues.
      pfc::mpi::throw_on_mpi_error(MPI_File_close(&file),
                                   "MPI_File_close in ~MPI_File_guard");
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

/**
 * @brief Fail closed when a local MPI-IO element count exceeds INT_MAX.
 *
 * Classic MPI-IO count parameters are `int`. Returning a truncated
 * `static_cast<int>(n)` would post a wrong collective count. Callers must
 * invoke this before `MPI_File_open` / `MPI_File_*_all`.
 *
 * @param n Local element count (`size_t`).
 * @param context Caller label included in the exception message.
 * @return `n` as `int` when `n <= INT_MAX`.
 * @throws std::overflow_error when `n > INT_MAX`.
 */
[[nodiscard]] inline int expect_mpi_io_count(std::size_t n, const char *context) {
  if (n > static_cast<std::size_t>(INT_MAX)) {
    throw std::overflow_error(std::string(context) +
                              ": local element count exceeds INT_MAX");
  }
  return static_cast<int>(n);
}

/**
 * @brief Fail closed when an MPI message element count exceeds INT_MAX.
 *
 * Classic MPI `count` arguments are `int`. A silent `static_cast<int>(n)`
 * truncates and posts the wrong length. Call before every SparseVector or
 * packed-face `MPI_Send` / `Recv` / `Isend` / `Irecv` that takes an element
 * count derived from `size_t`.
 *
 * @param n Element count (`size_t`).
 * @param what Caller label included in the exception message.
 * @return `n` as `int` when `n <= INT_MAX`.
 * @throws std::overflow_error when `n > INT_MAX`.
 */
[[nodiscard]] inline int ensure_mpi_int_count(std::size_t n, const char *what) {
  if (n > static_cast<std::size_t>(INT_MAX)) {
    std::ostringstream oss;
    oss << what << ": MPI count " << n << " exceeds INT_MAX";
    throw std::overflow_error(oss.str());
  }
  return static_cast<int>(n);
}

/**
 * @brief Overflow-safe product of local brick extents for MPI-IO buffers.
 *
 * Used by BinaryWriter / BinaryReader to compute the expected local element
 * count from `set_domain` extents. Rejects non-positive extents and silent
 * wrap on multiply. Does **not** clamp to `INT_MAX` (see
 * `expect_mpi_io_count`).
 *
 * @param local Local brick extents `(Lx, Ly, Lz)`.
 * @param context Caller label included in exception messages.
 * @return `local[0] * local[1] * local[2]` as `std::size_t`.
 * @throws std::invalid_argument if any extent is `<= 0`.
 * @throws std::overflow_error if the product would exceed `size_t` max.
 */
[[nodiscard]] inline std::size_t
checked_local_extent_product(const std::array<int, 3> &local,
                             const char *context) {
  for (int i = 0; i < 3; ++i) {
    if (local[i] <= 0) {
      throw std::invalid_argument(std::string(context) +
                                  ": local extent must be positive");
    }
  }

  unsigned long long n = 1;
  const auto max_sz =
      static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)());
  for (int i = 0; i < 3; ++i) {
    const auto li = static_cast<unsigned long long>(local[i]);
    if (n > max_sz / li) {
      throw std::overflow_error(std::string(context) +
                                ": local extent product overflows size_t");
    }
    n *= li;
  }
  return static_cast<std::size_t>(n);
}

} // namespace pfc::mpi
