// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file mpi_file_guard_test_utils.hpp
 * @brief Shared test utilities for verifying MPI_File_guard does not leak
 * file descriptors when an MPI-IO call fails after MPI_File_open succeeds.
 */

#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace pfc::test {

/**
 * @brief Check whether this process currently holds an open file descriptor
 * pointing at `path`, by scanning /proc/self/fd symlinks.
 *
 * Used to detect MPI_File handle leaks for a *specific* file. A raw open-fd
 * *count* is not reliable here: OpenMPI's own internal machinery (BTL
 * modules, progress threads, out-of-band channels) opens and closes
 * unrelated file descriptors as a side effect of collective I/O calls,
 * including ones that fail -- so the total fd count can shift by +/-1 for
 * reasons unconnected to the file under test. Checking whether a descriptor
 * resolves to this exact (canonicalized) path sidesteps that noise.
 * Linux-only, matching this project's HPC target environment.
 */
inline bool is_path_open(const std::string &path) {
  std::error_code ec;
  const std::filesystem::path canonical = std::filesystem::canonical(path, ec);
  if (ec) {
    return false;
  }
  for (const auto &entry :
       std::filesystem::directory_iterator("/proc/self/fd", ec)) {
    if (ec) {
      break;
    }
    std::error_code link_ec;
    const std::filesystem::path target =
        std::filesystem::read_symlink(entry.path(), link_ec);
    if (!link_ec && target == canonical) {
      return true;
    }
  }
  return false;
}

} // namespace pfc::test
