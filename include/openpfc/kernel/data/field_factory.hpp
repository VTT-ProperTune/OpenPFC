// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef OPENPFC_KERNEL_DATA_FIELD_FACTORY_HPP
#define OPENPFC_KERNEL_DATA_FIELD_FACTORY_HPP

/**
 * @file field_factory.hpp
 * @brief Factory functions for creating pfc::data::Field from decomposition
 *
 * @details
 * Provides free function templates that construct pfc::data::Field<T,HostSpace>
 * from decomposition geometry (local_box + domain), simplifying field creation
 * for subdomains in parallel simulations.
 */

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::data {

/**
 * @brief Create a canonical Field<T,HostSpace> from decomposition geometry
 *
 * Constructs a pfc::data::Field<T,HostSpace> with its owned box derived from
 * the decomposition's local_box for the given rank, and its global domain from
 * the decomposition's domain. This provides a convenient factory for allocating
 * fields that match the subdomain layout returned by decomposition.
 *
 * @tparam T Element type (e.g., double, std::complex<double>)
 * @param decomp The decomposition describing the subdomain layout
 * @param rank The rank/subdomain index (0 to num_domains-1)
 * @param halo Halo width (default 0 for unpadded interior)
 * @return Field<T,HostSpace> with geometry matching the decomposition's subdomain
 *
 * @throws std::out_of_range if rank is not a valid subdomain index
 *
 * @example
 * ```cpp
 * #include <openpfc/kernel/data/field_factory.hpp>
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
                                          int rank, int halo = 0) {
  const Box3i owned_box = decomposition::local_box(decomp, rank);
  const Domain global_domain = decomposition::domain(decomp);
  return Field<T, HostSpace>(global_domain, owned_box, halo);
}

} // namespace pfc::data

#endif // OPENPFC_KERNEL_DATA_FIELD_FACTORY_HPP
