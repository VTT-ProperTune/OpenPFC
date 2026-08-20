// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file hdf5_writer.hpp
 * @brief Optional HDF5 + XDMF field writer (OpenPFC_ENABLE_HDF5)
 *
 * @details
 * Serial (nproc=1) writer of a 3D double dataset `/field` plus a sibling
 * `.xdmf` sidecar for ParaView. Multi-rank jobs fail closed until parallel
 * HDF5 is wired. Complex fields fail closed.
 *
 * Dataset layout is C-order `(nz, ny, nx)` so the last index is x-fastest,
 * matching OpenPFC owned-brick linearization.
 */

#ifndef PFC_HDF5_WRITER_HPP
#define PFC_HDF5_WRITER_HPP

#include <array>
#include <mpi.h>
#include <openpfc/frontend/io/file_results_writer.hpp>
#include <string>

namespace pfc {

#ifdef OPENPFC_HAS_HDF5

class HDF5Writer : public FileResultsWriter {
  MPI_Comm m_comm = MPI_COMM_WORLD;
  std::array<int, 3> m_global{};
  std::array<int, 3> m_local{};
  std::array<int, 3> m_offset{};
  bool m_domain_valid = false;

public:
  explicit HDF5Writer(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD)
      : FileResultsWriter(filename), m_comm(comm) {}

  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) override;

  MPI_Status write(int increment, const RealField &data) override;
  MPI_Status write(int increment, const ComplexField &data) override;
};

#endif // OPENPFC_HAS_HDF5

} // namespace pfc

#endif // PFC_HDF5_WRITER_HPP
