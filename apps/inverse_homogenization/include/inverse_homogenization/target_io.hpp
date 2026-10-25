// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file target_io.hpp
 * @brief Load a 6×6 Voigt C_target and a Fortran-order design brick.
 *
 * Text C_target is six lines of six doubles (engineering Voigt,
 * 11 22 33 23 13 12). Binary h is nx*ny*nz float64, i-fastest, matching
 * openpfc_homogenize --load-bin.
 */

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <inverse_homogenization/homogenization.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

namespace pfc::apps::inverse {

[[nodiscard]] inline bool load_voigt6_file(const std::string &path, Voigt6 &C) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  int row = 0;
  while (row < 6 && std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    for (int col = 0; col < 6; ++col) {
      if (!(iss >> C(row, col))) return false;
    }
    ++row;
  }
  return row == 6;
}

inline bool write_voigt6_file(const std::string &path, const Voigt6 &C) {
  std::ofstream out(path);
  if (!out) return false;
  out << "# OpenPFC Voigt 6x6 (11 22 33 23 13 12)\n";
  out << std::setprecision(16);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (j) out << ' ';
      out << C(i, j);
    }
    out << '\n';
  }
  return static_cast<bool>(out);
}

[[nodiscard]] inline bool load_fortran_bin(const std::string &path, int nx, int ny,
                                           int nz, pfc::data::Field<double> &h) {
  const std::size_t ntot = static_cast<std::size_t>(nx) *
                           static_cast<std::size_t>(ny) *
                           static_cast<std::size_t>(nz);
  std::vector<double> a(ntot, 0.0);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int ok = 1;
  if (rank == 0) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(a.data()),
           static_cast<std::streamsize>(ntot * sizeof(double)));
    if (!f || static_cast<std::size_t>(f.gcount()) != ntot * sizeof(double)) ok = 0;
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) return false;
  MPI_Bcast(a.data(), static_cast<int>(ntot), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const std::size_t idx =
            static_cast<std::size_t>(g[0]) +
            static_cast<std::size_t>(nx) *
                (static_cast<std::size_t>(g[1]) +
                 static_cast<std::size_t>(ny) * static_cast<std::size_t>(g[2]));
        h(i, j, k) = a[idx];
      }
  h.note_host_write();
  return true;
}

} // namespace pfc::apps::inverse
