// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef PFC_KERNEL_FIELD_FIELD_FACTORY_HPP
#define PFC_KERNEL_FIELD_FIELD_FACTORY_HPP

/**
 * @file field_factory.hpp
 * @brief Construct the canonical `pfc::data::Field` from a `Decomposition`.
 *
 * The M2 migration replaces the legacy per-rank field construction
 *
 *     field::LocalField<T>::from_subdomain(decomp, rank, halo)   // halo 0
 *     field::PaddedBrick<T>(decomp, rank, halo)                  // halo n
 *
 * with the one owning container `pfc::data::Field<T, pfc::HostSpace>`. Both
 * legacy types derived identical geometry from the decomposition -- the owned
 * index box from the rank's subworld, and global size/origin/spacing from the
 * global world. The decomposition's canonical accessors `domain()` and
 * `local_box()` (M1.3) are defined to be numerically equal to that derivation,
 * so this factory is the drop-in construction bridge:
 *
 *     pfc::data::field_from_subdomain<T>(decomp, rank, halo)
 *
 * It lives in `kernel/field` (not `kernel/data`) so `grid_field.hpp` keeps no
 * dependency on `decomposition`/`fft` headers -- the M2 layering invariant.
 *
 * Behavioural notes vs the legacy types:
 *  - `Field` stores only the geometry POD (`Domain`) + `Box3i` by value; it does
 *    NOT keep the `Decomposition`/rank the way `PaddedBrick` did. Consumers that
 *    called `brick.decomposition()` / `brick.rank()` must thread those
 *    separately -- they are not a pure construction swap.
 *  - `Field`'s ctor rejects only a negative halo / inconsistent box; it does not
 *    reproduce `LocalField::from_subdomain`'s stricter "interior must exceed the
 *    halo" check. Callers that relied on that guard keep asserting it themselves.
 */

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>

namespace pfc::data {

/**
 * @brief Owning host `Field` for `rank`'s subdomain, halo-padded by `halo_width`.
 *
 * Geometry matches the legacy `LocalField`/`PaddedBrick` construction
 * bit-for-bit: the owned index box is `decomposition::local_box(decomp, rank)`
 * and the geometry POD is `decomposition::domain(decomp)`.
 *
 * @tparam T          Element type (e.g. `double`, `std::complex<double>`).
 * @param  decomp     The domain decomposition.
 * @param  rank       The MPI rank whose subdomain to allocate.
 * @param  halo_width Halo cells per side (0 == old LocalField, n == PaddedBrick).
 * @throws std::invalid_argument on a negative halo (from the `Field` ctor).
 */
template <typename T>
Field<T, pfc::HostSpace>
field_from_subdomain(const pfc::decomposition::Decomposition &decomp, int rank,
                     int halo_width = 0) {
  return Field<T, pfc::HostSpace>(pfc::decomposition::domain(decomp),
                                  pfc::decomposition::local_box(decomp, rank),
                                  halo_width);
}

} // namespace pfc::data

#endif // PFC_KERNEL_FIELD_FIELD_FACTORY_HPP
