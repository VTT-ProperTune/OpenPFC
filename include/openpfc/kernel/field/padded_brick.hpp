// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_brick.hpp
 * @brief A halo-padded brick for laboratory-style explicit stencil loops.
 *
 * @details
 * `pfc::field::PaddedBrick<T>` is the layout the **laboratory-style** FD
 * driver wants: a single contiguous buffer sized
 * `(nx + 2*hw) * (ny + 2*hw) * (nz + 2*hw)` so the user can write
 *
 *     u(i+1, j, k) - 2 * u(i, j, k) + u(i-1, j, k)
 *
 * for owned cell `i in [0, nx)` and have `u(-1, j, k)` and `u(nx, j, k)`
 * legitimately reach into the **left/right halo ring** that the in-place
 * `pfc::PaddedHaloExchanger<T>` writes into.
 *
 * Compared to its older siblings:
 *
 * - `pfc::field::LocalField<T>` carries no halo storage at all (size is
 *   exactly `nx*ny*nz`); the FD path keeps its halos in six separate
 *   face vectors and exchanges them with `SeparatedFaceHaloExchanger`.
 * - `pfc::HaloExchanger<T>` is the existing in-place exchanger but its
 *   "no extra padding" face-type spec (see
 *   `kernel/decomposition/halo_mpi_types.hpp` line 105) **overwrites the
 *   outermost owned cells with neighbor data** — there is no ghost ring,
 *   so `u(-1, ...)` is meaningless.
 *
 * `PaddedBrick<T>` therefore stores both the **owned core**
 * `[0, n) x [0, n) x [0, n)` and a **halo ring** of width `hw` on each
 * side of every axis in **one** contiguous buffer, and exposes
 * `T &operator()(int i, int j, int k)` valid for any
 * `i,j,k in [-hw, n+hw)`.
 *
 * The internal storage is row-major with **x fastest**, matching the rest
 * of OpenPFC (`kernel/field/fd_gradient.hpp`, `kernel/field/fd_apply.hpp`,
 * `kernel/decomposition/halo_mpi_types.hpp`, FFT inbox iteration, ...).
 *
 * @see kernel/decomposition/padded_halo_mpi_types.hpp for the matching
 *      MPI subarray helper that handles the padded layout.
 * @see kernel/decomposition/padded_halo_exchange.hpp for the in-place
 *      non-blocking halo exchanger built around it.
 * @see kernel/field/brick_iteration.hpp for `for_each_owned/inner/border`
 *      helpers that yield `(i, j, k)` triples over a `PaddedBrick`.
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::field {

/**
 * @brief Halo-padded contiguous brick: one buffer, `[-hw, n+hw)` indexing.
 *
 * Storage layout (row-major, x fastest):
 *   `linear = (i + hw) + (j + hw) * nx_pad + (k + hw) * nx_pad * ny_pad`
 * where `nx_pad = nx + 2*hw`, etc. The MPI subarray types built by
 * `pfc::halo::create_padded_face_types_6` use the same convention.
 *
 * @tparam T Element type (`double` for the heat equation; `float` /
 *           `std::complex<double>` are equally valid).
 */
template <class T> class PaddedBrick {
public:
  /**
   * @brief Construct a padded brick from an existing decomposition.
   *
   * Geometry comes from `decomposition::get_subworld(decomp, rank)` for
   * the owned size + lower global index, and from the **global world**
   * for the physical origin/spacing (so `global_coords(i, j, k)`
   * returns the right physical position even for halo cells `i = -1`,
   * which conceptually live at the rank's left neighbor).
   *
   * The returned buffer is value-initialized (`T{}`).
   *
   * @throws std::invalid_argument if `halo_width < 0`. Unlike
   *         `LocalField::from_subdomain` we do **not** require the
   *         owned region to exceed `2*halo_width` per axis — a
   *         padded brick with a tiny owned core is still valid; the
   *         user just doesn't get a non-empty inner region for that
   *         halo width.
   */
  PaddedBrick(const pfc::decomposition::Decomposition &decomp, int rank,
              int halo_width)
      : m_halo(halo_width) {
    if (halo_width < 0) {
      throw std::invalid_argument(
          "pfc::field::PaddedBrick: halo_width must be non-negative (got " +
          std::to_string(halo_width) + ")");
    }
    const auto &gw = pfc::decomposition::get_world(decomp);
    const auto &local = pfc::decomposition::get_subworld(decomp, rank);
    m_size = pfc::world::get_size(local);
    m_lower = pfc::world::get_lower(local);
    m_global_size = pfc::world::get_size(gw);
    m_origin = pfc::world::get_origin(gw);
    m_spacing = pfc::world::get_spacing(gw);

    const auto npx = padded_extent_(m_size[0]);
    const auto npy = padded_extent_(m_size[1]);
    const auto npz = padded_extent_(m_size[2]);
    m_data.assign(npx * npy * npz, T{});
  }

  // ---- Storage ------------------------------------------------------------

  /**
   * @brief Pointer to the **start of the padded buffer** (i.e. the
   *        `(-hw, -hw, -hw)` cell, **not** the `(0, 0, 0)` owned cell).
   *
   * This is the pointer that `pfc::PaddedHaloExchanger<T>` and the
   * matching MPI subarray types in
   * `kernel/decomposition/padded_halo_mpi_types.hpp` operate on.
   */
  T *data() noexcept { return m_data.data(); }
  const T *data() const noexcept { return m_data.data(); }

  /// Total number of elements in the padded buffer (`nx_pad*ny_pad*nz_pad`).
  std::size_t size() const noexcept { return m_data.size(); }

  /** Underlying vector — pass to allocators, exchangers, raw kernels. */
  std::vector<T> &vec() noexcept { return m_data; }
  const std::vector<T> &vec() const noexcept { return m_data; }

  // ---- Geometry queries ---------------------------------------------------

  /// Local owned size `(nx, ny, nz)` (excludes the halo ring).
  pfc::Int3 size3() const noexcept { return m_size; }
  /// Local **padded** size `(nx + 2hw, ny + 2hw, nz + 2hw)`.
  pfc::Int3 padded_size3() const noexcept {
    return pfc::Int3{padded_extent_(m_size[0]), padded_extent_(m_size[1]),
                     padded_extent_(m_size[2])};
  }
  pfc::Int3 lower_global() const noexcept { return m_lower; }
  pfc::Int3 global_size() const noexcept { return m_global_size; }
  pfc::Real3 origin() const noexcept { return m_origin; }
  pfc::Real3 spacing() const noexcept { return m_spacing; }
  int halo_width() const noexcept { return m_halo; }

  /// Owned-x extent.
  int nx() const noexcept { return m_size[0]; }
  /// Owned-y extent.
  int ny() const noexcept { return m_size[1]; }
  /// Owned-z extent.
  int nz() const noexcept { return m_size[2]; }
  /// Padded-x extent (`nx + 2*hw`).
  int nx_padded() const noexcept { return padded_extent_(m_size[0]); }
  /// Padded-y extent (`ny + 2*hw`).
  int ny_padded() const noexcept { return padded_extent_(m_size[1]); }
  /// Padded-z extent (`nz + 2*hw`).
  int nz_padded() const noexcept { return padded_extent_(m_size[2]); }

  // ---- Indexing helpers ---------------------------------------------------

  /**
   * @brief Flat row-major index for local `(i, j, k)`. Valid for any
   *        `i,j,k in [-hw, n+hw)`. Halo cells map to the outer ring.
   *
   * @note No bounds checking in the release build. Out-of-range arguments
   *       are undefined behaviour; pair with `for_each_owned` / `_inner`
   *       / `_border` to guarantee the iteration stays in-range.
   */
  std::size_t idx(int i, int j, int k) const noexcept {
    const auto npx = static_cast<std::size_t>(padded_extent_(m_size[0]));
    const auto npy = static_cast<std::size_t>(padded_extent_(m_size[1]));
    const auto hw = static_cast<std::size_t>(m_halo);
    return (static_cast<std::size_t>(i) + hw) +
           (static_cast<std::size_t>(j) + hw) * npx +
           (static_cast<std::size_t>(k) + hw) * npx * npy;
  }

  /// Element access for any `i,j,k in [-hw, n+hw)`.
  T &operator()(int i, int j, int k) noexcept { return m_data[idx(i, j, k)]; }
  const T &operator()(int i, int j, int k) const noexcept {
    return m_data[idx(i, j, k)];
  }

  /**
   * @brief Global cell index `(gi, gj, gk)` of local `(i, j, k)`.
   *
   * For halo cells (e.g. `i = -1`) this returns the conceptual global
   * index **before** any periodic wrap (so callers can implement either
   * periodic or Dirichlet semantics in their own filling code).
   */
  pfc::Int3 global(int i, int j, int k) const noexcept {
    return pfc::Int3{m_lower[0] + i, m_lower[1] + j, m_lower[2] + k};
  }

  /**
   * @brief Physical coordinates `(x, y, z)` of local `(i, j, k)`.
   *
   * Computed as `origin + (lower + i) * spacing` — same as
   * `LocalField::coords` but valid across the halo ring as well.
   */
  pfc::Real3 global_coords(int i, int j, int k) const noexcept {
    return pfc::Real3{
        m_origin[0] + static_cast<double>(m_lower[0] + i) * m_spacing[0],
        m_origin[1] + static_cast<double>(m_lower[1] + j) * m_spacing[1],
        m_origin[2] + static_cast<double>(m_lower[2] + k) * m_spacing[2]};
  }

  // ---- Convenience iteration ---------------------------------------------
  //
  // The richer `for_each_owned/inner/border` helpers that yield raw
  // `(i, j, k)` triples live in `kernel/field/brick_iteration.hpp` so
  // this header stays focused on the data layout.

  /**
   * @brief Fill **every owned cell** by sampling `fn` at its physical
   *        coordinates. Halo cells are left untouched (zero-initialized).
   *
   * Lambda may be either of these signatures (auto-detected):
   *  - `T(double x, double y, double z)`
   *  - `T(const Real3& x)`
   */
  template <class Fn> void apply(Fn &&fn) {
    for (int k = 0; k < m_size[2]; ++k) {
      for (int j = 0; j < m_size[1]; ++j) {
        for (int i = 0; i < m_size[0]; ++i) {
          const auto x = global_coords(i, j, k);
          m_data[idx(i, j, k)] = static_cast<T>(invoke_with_coords_(fn, x));
        }
      }
    }
  }

private:
  int padded_extent_(int n) const noexcept { return n + 2 * m_halo; }

  template <class Fn> static auto invoke_with_coords_(Fn &&fn, const pfc::Real3 &x) {
    if constexpr (std::is_invocable_v<Fn, double, double, double>) {
      return fn(x[0], x[1], x[2]);
    } else {
      return fn(x);
    }
  }

  std::vector<T> m_data{};
  pfc::Int3 m_size{};
  pfc::Int3 m_lower{};
  pfc::Int3 m_global_size{};
  pfc::Real3 m_origin{};
  pfc::Real3 m_spacing{};
  int m_halo{0};
};

} // namespace pfc::field
