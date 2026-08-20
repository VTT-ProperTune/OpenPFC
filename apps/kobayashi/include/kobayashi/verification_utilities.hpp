// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file verification_utilities.hpp
 * @brief Kobayashi PNG helper plus re-exports of shared XY gather/stats.
 */

#ifndef KOBAYASHI_VERIFICATION_UTILITIES_HPP
#define KOBAYASHI_VERIFICATION_UTILITIES_HPP

#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc_apps/gather.hpp>

namespace {

using pfc::apps::FieldStats;
using pfc::apps::gather_global_xy_rank0;
using pfc::apps::pack_owned_xy0;
using pfc::apps::stats_global_ordered;

void write_phi_png(int rank, const pfc::decomposition::Decomposition &decomp,
                   const pfc::data::Field<double, pfc::HostSpace> &phi,
                   const std::string &path) {
  std::vector<double> local;
  pack_owned_xy0(phi, local);
  pfc::io::write_mpi_scalar_field_png_xy(MPI_COMM_WORLD, decomp, rank, local, path,
                                         0.0, 1.0);
}

} // namespace

#endif // KOBAYASHI_VERIFICATION_UTILITIES_HPP
