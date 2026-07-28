// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef OPENPFC_KERNEL_FIELD_FIELD_FACTORY_HPP
#define OPENPFC_KERNEL_FIELD_FIELD_FACTORY_HPP

/**
 * @file field_factory.hpp
 * @brief Construction helpers for pfc::data::Field from decomposition geometry
 *
 * @details
 * Provides free function templates that construct pfc::data::Field<T,HostSpace>
 * from decomposition geometry (local_box + domain), maintaining the decomposition
 * dependency boundary in kernel/field rather than kernel/data.
 */

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <stdexcept>
#include <memory>

namespace pfc::data {

/**
 * @brief Create a canonical Field<T,HostSpace> from decomposition geometry
 *
 * Constructs a pfc::data::Field<T,HostSpace> with its owned box derived from
 * the decomposition's local_box for the given rank, and its global domain from
 * the decomposition's domain. This provides a convenient factory for allocating
 * fields that match the subdomain layout returned by decomposition.
 *
 * The resulting field layout is bit-for-bit identical to:
 * - pfc::data::Field with halo=0 (migrated from LocalField::from_subdomain)
 * - PaddedBrick with halo=n
 *
 * @tparam T Element type (e.g., double, std::complex<double>)
 * @param decomp The decomposition describing the subdomain layout
 * @param rank The rank/subdomain index (0 to num_domains-1)
 * @param halo Halo width (must be non-negative)
 * @return Field<T,HostSpace> with geometry matching the decomposition's subdomain
 *
 * @throws std::invalid_argument if halo is negative
 * @throws std::out_of_range if rank is not a valid subdomain index
 *
 * @example
 * ```cpp
 * #include <openpfc/kernel/field/field_factory.hpp>
 * #include <openpfc/kernel/decomposition/decomposition.hpp>
 *
 * using namespace pfc;
 * auto domain = domain::create({128, 128, 128});
 * auto decomp = decomposition::create(domain, 4); // 4 subdomains
 *
 * // Allocate a field for rank 0's subdomain with halo width 1
 * Field<double, HostSpace> field =
 *     pfc::data::field_from_subdomain<double>(decomp, 0, 1);
 * ```
 */
template <typename T>
Field<T, HostSpace> field_from_subdomain(const decomposition::Decomposition& decomp,
                                          int rank, int halo) {
  if (halo < 0) {
    throw std::invalid_argument("halo must be non-negative");
  }
  // PaddedBrick-compatible: storage padding == iteration halo.
  return Field<T, HostSpace>(decomposition::domain(decomp),
                              decomposition::local_box(decomp, rank),
                              halo);
}

/**
 * @brief LocalField-compatible factory: unpadded storage + iteration halo.
 *
 * Storage is tightly packed `nx*ny*nz` (face-halos live elsewhere). The
 * `iteration_halo` is exposed via `Field::halo_width()` / `for_each_interior`,
 * matching `LocalField::from_subdomain(decomp, rank, halo_width)`.
 */
template <typename T>
Field<T, HostSpace>
field_from_subdomain_unpadded(const decomposition::Decomposition &decomp,
                              int rank, int iteration_halo = 0) {
  if (iteration_halo < 0) {
    throw std::invalid_argument("iteration_halo must be non-negative");
  }
  return Field<T, HostSpace>(decomposition::domain(decomp),
                              decomposition::local_box(decomp, rank),
                              /*storage_halo=*/0, iteration_halo);
}

/**
 * @brief Create an unpadded Field from an FFT inbox box + global Domain.
 *
 * LocalField-compatible replacement for `LocalField::from_inbox(world, inbox)`.
 * Spectral apps use halo=0 (no per-rank halos in the inbox layout).
 */
template <typename T>
Field<T, HostSpace> field_from_inbox(const pfc::Domain &domain,
                                     const pfc::Box3i &inbox) {
  return Field<T, HostSpace>(domain, inbox, /*storage_halo=*/0,
                              /*iteration_halo=*/0);
}

/**
 * @brief Create a unique_ptr wrapper for a Field from a PaddedBrick.
 *
 * This adapter creates a Field with the same geometry and halo width as a
 * PaddedBrick for migration purposes. The Field is heap-allocated and returned
 * as a unique_ptr to avoid copying data during migration.
 *
 * @tparam T Element type
 * @param brick Reference to PaddedBrick to adapt
 * @return unique_ptr<Field<T, HostSpace>> with matching geometry and halo
 */
template <typename T>
std::unique_ptr<Field<T, HostSpace>> field_from_subdomain_brick(
    const pfc::field::PaddedBrick<T>& brick) {
  // Get the original Box3i from the decomposition - this is preferrable to 
  // reconstructing it from components because it ensures consistency
  const auto& decomp = brick.decomposition();
  const int rank = brick.rank();
  const auto local_box = pfc::decomposition::local_box(decomp, rank);
  
  auto field = std::make_unique<Field<T, HostSpace>>(
      pfc::decomposition::domain(decomp), 
      local_box,
      brick.halo_width(), brick.halo_width());
  
  // Copy data from brick to field
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  const int hw = brick.halo_width();
  
  for (int k = -hw; k < nz + hw; ++k) {
    for (int j = -hw; j < ny + hw; ++j) {
      for (int i = -hw; i < nx + hw; ++i) {
        (*field)(i, j, k) = brick(i, j, k);
      }
    }
  }
  
  return field;
}

} // namespace pfc::data

#endif // OPENPFC_KERNEL_FIELD_FIELD_FACTORY_HPP
