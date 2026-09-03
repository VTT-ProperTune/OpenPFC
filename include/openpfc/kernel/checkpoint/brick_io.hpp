// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file brick_io.hpp
 * @brief Collective MPI-IO of one owned-cell float64 brick (checkpoint fields).
 *
 * Same subarray contract as `BinaryWriter` / `BinaryReader` (Fortran order,
 * x-fastest). Lives in kernel so checkpoint save does not include frontend.
 */

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/mpi/domain_geometry.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc::checkpoint {

inline void write_real_brick_mpi(const std::string &path, MPI_Comm comm,
                                 const std::array<int, 3> &global,
                                 const std::array<int, 3> &local,
                                 const std::array<int, 3> &offset,
                                 pfc::field::FieldView<double> data) {
  pfc::mpi::validate_subarray_domain(global, local, offset, "write_real_brick_mpi");
  const std::size_t expected =
      pfc::mpi::checked_local_extent_product(local, "write_real_brick_mpi");
  int local_ok = data.size() == expected ? 1 : 0;
  int global_ok = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm),
      "MPI_Allreduce write_real_brick_mpi");
  if (global_ok == 0) {
    std::ostringstream oss;
    oss << "write_real_brick_mpi: buffer size mismatch (expected " << expected
        << ", got " << data.size() << ")";
    throw std::runtime_error(oss.str());
  }

  MPI_Datatype filetype = MPI_DATATYPE_NULL;
  pfc::mpi::throw_on_mpi_error(
      MPI_Type_create_subarray(3, global.data(), local.data(), offset.data(),
                               MPI_ORDER_FORTRAN, MPI_DOUBLE, &filetype),
      "MPI_Type_create_subarray");
  pfc::mpi::throw_on_mpi_error(MPI_Type_commit(&filetype), "MPI_Type_commit");
  struct TypeFree {
    MPI_Datatype t{MPI_DATATYPE_NULL};
    explicit TypeFree(MPI_Datatype dt) : t(dt) {}
    ~TypeFree() {
      if (t != MPI_DATATYPE_NULL) {
        pfc::mpi::abort_on_mpi_error(MPI_Type_free(&t),
                                     "MPI_Type_free in write_real_brick_mpi");
      }
    }
  } type_free(filetype);

  MPI_File fh{};
  pfc::mpi::throw_on_mpi_error(MPI_File_open(comm, const_cast<char *>(path.c_str()),
                                             MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                             MPI_INFO_NULL, &fh),
                               "MPI_File_open");
  pfc::mpi::MPI_File_guard guard(fh);
  pfc::mpi::throw_on_mpi_error(MPI_File_set_size(fh, 0), "MPI_File_set_size");
  pfc::mpi::throw_on_mpi_error(
      MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL),
      "MPI_File_set_view");
  const int count =
      pfc::mpi::expect_mpi_io_count(data.size(), "write_real_brick_mpi");
  MPI_Status status{};
  pfc::mpi::throw_on_mpi_error(
      MPI_File_write_all(fh, data.data(), count, MPI_DOUBLE, &status),
      "MPI_File_write_all");
}

} // namespace pfc::checkpoint
