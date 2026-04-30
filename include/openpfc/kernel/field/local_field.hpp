// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file local_field.hpp
 * @brief Typed bundle of `(storage + size + lower + origin + spacing + halo)`
 *        for a rank's local field — the small "ndarray" the OpenPFC apps want.
 *
 * @details
 * `pfc::field::LocalField<T>` is a header-only owning container that ties a
 * contiguous `std::vector<T>` to the geometry it lives in:
 *
 *  - **size**         : local extents `(nx, ny, nz)`
 *  - **lower_global** : global index of cell `(0,0,0)` local
 *  - **global_size**  : global grid extents `(Nx, Ny, Nz)` (needed by spectral
 *                       evaluators for the per-axis Fourier symbols)
 *  - **origin**       : physical origin of the global `(0,0,0)` cell
 *  - **spacing**      : grid spacing per axis
 *  - **halo_width**   : per-rank halo region to skip in interior loops
 *
 * Storage is **unpadded**: size is `nx*ny*nz`. Halo data, if any, lives in a
 * separate buffer (e.g. `pfc::halo::FaceHalos<T>` for the FD path); this
 * container only knows the halo *width* so its `for_each_interior` skips the
 * `[0, hw)` and `[n-hw, n)` slabs in each axis. This matches the existing
 * `pfc::field::FdGradient` and `pfc::sim::for_each_interior` conventions.
 *
 * Two named constructors cover the common layout sources:
 *
 *  - `LocalField::from_subdomain(decomp, rank, halo_width = 0)` — geometry
 *    derived from `decomposition::get_subworld(decomp, rank)`. Use this for
 *    pure FD apps that own the halo exchange.
 *  - `LocalField::from_inbox(global_world, inbox)` — geometry derived from
 *    an FFT inbox `Box3i` plus the global world. Use this for spectral apps.
 *
 * Helpers expose physical coordinates (`coords`), global indices (`global`),
 * the row-major flat index (`idx`), and three iteration patterns
 * (`apply`, `for_each_owned`, `for_each_interior`). Each callable can be
 * written either coordinate-tuple style `(x, y, z)` or `Real3` style; the
 * container detects the signature with `if constexpr`.
 *
 * Layout: row-major `[nx, ny, nz]` with **x varying fastest**, matching the
 * rest of OpenPFC's FD and FFT stack (`pfc::field::fd::*`,
 * `pfc::field::FdGradient`, `pfc::field::SpectralGradient`,
 * `pfc::halo::*`, FFT inbox iteration).
 *
 * @see kernel/field/operations.hpp for the free-function siblings
 *      (`pfc::field::apply_subdomain`, `for_each_interior_with_coords`)
 * @see kernel/field/fd_gradient.hpp / spectral_gradient.hpp for evaluators
 *      that consume the underlying buffer via `data()` / `vec()`
 */

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

namespace pfc::field {

template <class T> class LocalField {
public:
  // ---- Named constructors -------------------------------------------------

  /**
   * @brief FD subdomain layout: geometry from `decomp.get_subworld(rank)`,
   *        physical metadata from the global world.
   *
   * Storage is sized to `nx*ny*nz` and value-initialized. `halo_width` is
   * stored so `for_each_interior` can skip the per-rank halo region; the
   * actual halo data, if needed, lives in a separate buffer (e.g.
   * `pfc::halo::FaceHalos<T>`).
   */
  static LocalField from_subdomain(const pfc::decomposition::Decomposition &decomp,
                                   int rank, int halo_width = 0) {
    const auto &gw = pfc::decomposition::get_world(decomp);
    const auto &local = pfc::decomposition::get_subworld(decomp, rank);
    return LocalField(pfc::world::get_size(local), pfc::world::get_lower(local),
                      pfc::world::get_size(gw), pfc::world::get_origin(gw),
                      pfc::world::get_spacing(gw), halo_width);
  }

  /**
   * @brief FFT inbox layout: geometry from a `pfc::fft::Box3i` (typically
   *        `IFFT::get_inbox_bounds()`) plus the global world. `halo_width`
   *        is `0` (spectral apps do not carry per-rank halos).
   */
  static LocalField from_inbox(const pfc::World &global_world,
                               const pfc::fft::Box3i &inbox) {
    return LocalField(inbox.size, inbox.low, pfc::world::get_size(global_world),
                      pfc::world::get_origin(global_world),
                      pfc::world::get_spacing(global_world), 0);
  }

  // ---- Storage ------------------------------------------------------------

  T *data() noexcept { return m_data.data(); }
  const T *data() const noexcept { return m_data.data(); }
  std::size_t size() const noexcept { return m_data.size(); }

  /** Underlying vector — pass to FFT / halo exchanger / stepper APIs. */
  std::vector<T> &vec() noexcept { return m_data; }
  const std::vector<T> &vec() const noexcept { return m_data; }

  // ---- Geometry queries ---------------------------------------------------

  pfc::Int3 size3() const noexcept { return m_size; }
  pfc::Int3 lower_global() const noexcept { return m_lower; }
  pfc::Int3 global_size() const noexcept { return m_global_size; }
  pfc::Real3 origin() const noexcept { return m_origin; }
  pfc::Real3 spacing() const noexcept { return m_spacing; }
  int halo_width() const noexcept { return m_halo; }

  // ---- Indexing helpers ---------------------------------------------------

  /** Flat row-major index for local logical `(ix, iy, iz)` (x fastest). */
  std::size_t idx(int ix, int iy, int iz) const noexcept {
    const auto nx = static_cast<std::size_t>(m_size[0]);
    const auto ny = static_cast<std::size_t>(m_size[1]);
    return static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * nx +
           static_cast<std::size_t>(iz) * nx * ny;
  }

  /** Global index `(gi, gj, gk)` of local logical `(ix, iy, iz)`. */
  pfc::Int3 global(int ix, int iy, int iz) const noexcept {
    return pfc::Int3{m_lower[0] + ix, m_lower[1] + iy, m_lower[2] + iz};
  }

  /** Physical coordinates `(x, y, z)` of local logical `(ix, iy, iz)`. */
  pfc::Real3 coords(int ix, int iy, int iz) const noexcept {
    return pfc::Real3{
        m_origin[0] + static_cast<double>(m_lower[0] + ix) * m_spacing[0],
        m_origin[1] + static_cast<double>(m_lower[1] + iy) * m_spacing[1],
        m_origin[2] + static_cast<double>(m_lower[2] + iz) * m_spacing[2]};
  }

  T &operator()(int ix, int iy, int iz) noexcept { return m_data[idx(ix, iy, iz)]; }
  const T &operator()(int ix, int iy, int iz) const noexcept {
    return m_data[idx(ix, iy, iz)];
  }

  // ---- Iteration ----------------------------------------------------------

  /**
   * @brief Fill **every owned cell** by sampling `fn` at its physical
   *        coordinates.
   *
   * Lambda may be either of these signatures (auto-detected):
   *  - `T(double x, double y, double z)`
   *  - `T(const Real3& x)`
   */
  template <class Fn> void apply(Fn &&fn) {
    for_each_index_(
        0, m_size[0], 0, m_size[1], 0, m_size[2], [&](int ix, int iy, int iz) {
          const auto x = coords(ix, iy, iz);
          m_data[idx(ix, iy, iz)] = static_cast<T>(invoke_with_coords_(fn, x));
        });
  }

  /**
   * @brief Iterate every owned cell, exposing `coords` and `value`.
   *
   * Lambda may be either of these signatures (auto-detected):
   *  - `void(double x, double y, double z, T value)`
   *  - `void(const Real3& x, T value)`
   */
  template <class Fn> void for_each_owned(Fn &&fn) const {
    for_each_index_(0, m_size[0], 0, m_size[1], 0, m_size[2],
                    [&](int ix, int iy, int iz) {
                      const auto x = coords(ix, iy, iz);
                      invoke_with_coords_value_(fn, x, m_data[idx(ix, iy, iz)]);
                    });
  }

  /**
   * @brief Iterate the **interior** `[hw, n-hw)` of every axis, exposing
   *        `coords` and `value`. No-op if the interior is empty.
   *
   * Lambda may be either of these signatures (auto-detected):
   *  - `void(double x, double y, double z, T value)`
   *  - `void(const Real3& x, T value)`
   */
  template <class Fn> void for_each_interior(Fn &&fn) const {
    const int hw = m_halo;
    const int imin = hw;
    const int imax = m_size[0] - hw;
    const int jmin = hw;
    const int jmax = m_size[1] - hw;
    const int kmin = hw;
    const int kmax = m_size[2] - hw;
    if (imin >= imax || jmin >= jmax || kmin >= kmax) {
      return;
    }
    for_each_index_(imin, imax, jmin, jmax, kmin, kmax, [&](int ix, int iy, int iz) {
      const auto x = coords(ix, iy, iz);
      invoke_with_coords_value_(fn, x, m_data[idx(ix, iy, iz)]);
    });
  }

private:
  LocalField(pfc::Int3 size, pfc::Int3 lower, pfc::Int3 global_size,
             pfc::Real3 origin, pfc::Real3 spacing, int halo)
      : m_data(static_cast<std::size_t>(size[0]) *
                   static_cast<std::size_t>(size[1]) *
                   static_cast<std::size_t>(size[2]),
               T{}),
        m_size(size), m_lower(lower), m_global_size(global_size), m_origin(origin),
        m_spacing(spacing), m_halo(halo) {}

  template <class Body>
  static void for_each_index_(int imin, int imax, int jmin, int jmax, int kmin,
                              int kmax, Body &&body) {
    for (int iz = kmin; iz < kmax; ++iz) {
      for (int iy = jmin; iy < jmax; ++iy) {
        for (int ix = imin; ix < imax; ++ix) {
          body(ix, iy, iz);
        }
      }
    }
  }

  template <class Fn> static auto invoke_with_coords_(Fn &&fn, const pfc::Real3 &x) {
    if constexpr (std::is_invocable_v<Fn, double, double, double>) {
      return fn(x[0], x[1], x[2]);
    } else {
      return fn(x);
    }
  }

  template <class Fn>
  static void invoke_with_coords_value_(Fn &&fn, const pfc::Real3 &x, const T &v) {
    if constexpr (std::is_invocable_v<Fn, double, double, double, T>) {
      fn(x[0], x[1], x[2], v);
    } else {
      fn(x, v);
    }
  }

  std::vector<T> m_data;
  pfc::Int3 m_size{};
  pfc::Int3 m_lower{};
  pfc::Int3 m_global_size{};
  pfc::Real3 m_origin{};
  pfc::Real3 m_spacing{};
  int m_halo{0};
};

} // namespace pfc::field
