// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file vtk_snapshot.hpp
 * @brief Drive `pfc::VTKWriter` from `PaddedBrick` or a contiguous owned slab
 *        (CPU `PaddedBrick`, GPU separated host buffers).
 */

#include <array>
#include <filesystem>
#include <string>

#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace wave2d {

/** @brief Create parent directory of the first VTK file (rank 0 only). */
inline void mkdir_vtk_parent_rank0(const std::string &pattern, int rank) {
  if (rank != 0) {
    return;
  }
  const std::string sample = pfc::utils::format_with_number(pattern, 0);
  std::filesystem::path p(sample);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
}

/** @brief Owned cells only, x fastest (matches VTK Piece layout). */
inline void pack_brick_owned(const pfc::field::PaddedBrick<double> &u,
                             pfc::RealField &out) {
  out.resize(static_cast<std::size_t>(u.nx()) * static_cast<std::size_t>(u.ny()) *
             static_cast<std::size_t>(u.nz()));
  std::size_t p = 0;
  pfc::field::for_each_owned(u, [&](int i, int j, int k) { out[p++] = u(i, j, k); });
}

inline void vtk_configure_writer(pfc::VTKWriter &w,
                                 const pfc::field::PaddedBrick<double> &u) {
  const auto g = u.global_size();
  const std::array<int, 3> global{g[0], g[1], g[2]};
  const std::array<int, 3> local{u.nx(), u.ny(), u.nz()};
  const auto lo = u.lower_global();
  const std::array<int, 3> off{lo[0], lo[1], lo[2]};
  w.set_domain(global, local, off);
  const auto o = u.origin();
  const auto s = u.spacing();
  w.set_origin({o[0], o[1], o[2]});
  w.set_spacing({s[0], s[1], s[2]});
  w.set_field_name("u");
}

inline void vtk_write_increment(pfc::VTKWriter &w, int increment,
                                const pfc::field::PaddedBrick<double> &u,
                                pfc::RealField &buf) {
  pack_brick_owned(u, buf);
  (void)w.write(increment, buf);
}

/**
 * @brief Configure VTK writer for one rank’s owned brick (no halos), x fastest.
 *
 * Matches the layout of `SeparatedFaceHaloExchanger` local vectors (`nx*ny*nz`).
 */
inline void vtk_configure_writer_owned_slab(pfc::VTKWriter &w,
                                            const std::array<int, 3> &global_size,
                                            const std::array<int, 3> &local_owned,
                                            const std::array<int, 3> &lower_global,
                                            const std::array<double, 3> &origin,
                                            const std::array<double, 3> &spacing) {
  w.set_domain(global_size, local_owned, lower_global);
  w.set_origin(origin);
  w.set_spacing(spacing);
  w.set_field_name("u");
}

/** @brief Write one time slice from a contiguous owned `u` buffer (same layout as
 * above). */
inline void vtk_write_u_owned_buffer(pfc::VTKWriter &w, int increment,
                                     const double *u, int nx, int ny, int nz,
                                     pfc::RealField &buf) {
  const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz);
  buf.assign(u, u + n);
  (void)w.write(increment, buf);
}

} // namespace wave2d
