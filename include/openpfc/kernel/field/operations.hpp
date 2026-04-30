// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file operations.hpp
 * @brief Functional, coordinate-space field operations (header-only)
 *
 * @details
 * This header provides zero-overhead, template-based helpers to apply
 * user-defined functions over real-space fields using coordinate-space
 * callbacks. It transparently respects the local MPI inbox via FFT layout
 * and avoids boilerplate nested loops in initial/boundary conditions.
 *
 * Core goals:
 * - Work in coordinate space: Fn(Real3) -> double, or Fn(Real3, t)
 * - Operate over the local inbox only (distributed-memory friendly)
 * - Header-only, zero-cost abstractions
 * - Backward compatible: usable directly with Model or raw components
 *
 * Example:
 * @code
 * using namespace pfc;
 * auto world = world::create(GridSize({64,64,64}));
 * auto decomp = decomposition::create(world, 1);
 * auto fft = fft::create(decomp);
 * std::vector<double> u(fft.size_inbox());
 *
 * // Set Gaussian pulse
 * pfc::field::apply(u, world, fft, [](const Real3& x){
 *   const double r2 = (x[0]*x[0]) + (x[1]*x[1]) + (x[2]*x[2]);
 *   return std::exp(-r2/2.0);
 * });
 * @endcode
 */

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/model.hpp>
// Local iteration implemented inline to work with HeFFTe inbox type

namespace pfc::field {

/**
 * @brief Spatial coordinate function \f$f(x,y,z)\f$.
 *
 * Type-erased lambda used by helpers that walk every local cell of a field
 * and write a coordinate-derived value (initial conditions, source terms,
 * spatial coefficient profiles, ...). Prefer the templated `apply` helpers
 * for hot paths where the function is known at compile time; use `PointFn`
 * when the function is selected at runtime (e.g. swappable initial
 * conditions).
 */
using PointFn = std::function<double(double, double, double)>;

/**
 * @brief Space-time coordinate function \f$f(x,y,z,t)\f$.
 *
 * Used by helpers that need the simulation time (e.g. boundary-value
 * providers, time-dependent forcing).
 */
using PointFnT = std::function<double(double, double, double, double)>;

/**
 * @brief Apply a coordinate-space function over a real field (local inbox)
 *
 * @tparam Fn Callable: double(const Real3&) or double(Real3)
 * @param field Real-valued field storage (local inbox size)
 * @param world Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param fn    Coordinate-space function returning new value
 */
template <typename Fn>
inline void apply(RealField &field, const World &world, const fft::IFFT &fft,
                  Fn &&fn) {
  const auto inbox = pfc::fft::get_inbox(fft);
  // Safety: ensure field size matches inbox voxel count
  const auto nx = inbox.size[0];
  const auto ny = inbox.size[1];
  const auto nz = inbox.size[2];
  const auto expected = static_cast<size_t>(nx) * ny * nz;
  if (field.size() != expected) {
    throw std::invalid_argument(
        "field::apply: field size does not match FFT inbox size");
  }

  size_t linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const pfc::Int3 idx{i, j, k};
        const auto x = pfc::world::to_coords(world, idx);
        field[linear_idx++] = static_cast<double>(fn(x));
      }
    }
  }
}

/**
 * @brief Apply a space-time function over a real field (local inbox)
 *
 * @tparam Fn Callable: double(const Real3&, double) or double(Real3,double)
 * @param field Real-valued field storage (local inbox size)
 * @param world Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param t     Simulation time passed to the function
 * @param fn    Space-time function returning new value
 */
template <typename Fn>
inline void apply_with_time(RealField &field, const World &world,
                            const fft::IFFT &fft, double t, Fn &&fn) {
  const auto inbox = pfc::fft::get_inbox(fft);
  const auto nx = inbox.size[0];
  const auto ny = inbox.size[1];
  const auto nz = inbox.size[2];
  const auto expected = static_cast<size_t>(nx) * ny * nz;
  if (field.size() != expected) {
    throw std::invalid_argument(
        "field::apply_with_time: field size does not match FFT inbox size");
  }

  size_t linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const pfc::Int3 idx{i, j, k};
        const auto x = pfc::world::to_coords(world, idx);
        field[linear_idx++] = static_cast<double>(fn(x, t));
      }
    }
  }
}

/**
 * @brief Apply a coordinate-space function in-place over a real field (local inbox)
 *
 * The callable receives both the coordinates and the current field value and
 * must return the updated value. Returning the current value leaves the cell
 * unchanged, enabling partial updates (e.g., boundary bands).
 *
 * @tparam Fn Callable: double(const Real3&, double current)
 * @param field Real-valued field storage (local inbox size)
 * @param world Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param fn    Coordinate-space function returning new value given (x, current)
 */
template <typename Fn>
inline void apply_inplace(RealField &field, const World &world, const fft::IFFT &fft,
                          Fn &&fn) {
  const auto inbox = pfc::fft::get_inbox(fft);
  const auto nx = inbox.size[0];
  const auto ny = inbox.size[1];
  const auto nz = inbox.size[2];
  const auto expected = static_cast<size_t>(nx) * ny * nz;
  if (field.size() != expected) {
    throw std::invalid_argument(
        "field::apply_inplace: field size does not match FFT inbox size");
  }

  size_t linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const pfc::Int3 idx{i, j, k};
        const auto x = pfc::world::to_coords(world, idx);
        field[linear_idx] = static_cast<double>(fn(x, field[linear_idx]));
        ++linear_idx;
      }
    }
  }
}

/**
 * @brief Apply a space-time function in-place over a real field (local inbox)
 *
 * @tparam Fn Callable: double(const Real3&, double current, double t)
 */
template <typename Fn>
inline void apply_inplace_with_time(RealField &field, const World &world,
                                    const fft::IFFT &fft, double t, Fn &&fn) {
  const auto inbox = pfc::fft::get_inbox(fft);
  const auto nx = inbox.size[0];
  const auto ny = inbox.size[1];
  const auto nz = inbox.size[2];
  const auto expected = static_cast<size_t>(nx) * ny * nz;
  if (field.size() != expected) {
    throw std::invalid_argument(
        "field::apply_inplace_with_time: field size does not match FFT inbox size");
  }

  size_t linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const pfc::Int3 idx{i, j, k};
        const auto x = pfc::world::to_coords(world, idx);
        field[linear_idx] = static_cast<double>(fn(x, field[linear_idx], t));
        ++linear_idx;
      }
    }
  }
}

/**
 * @brief Model overload: apply in-place to a named field
 */
template <typename Fn>
inline void apply_inplace(Model &model, std::string_view field_name, Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  apply_inplace(f, pfc::get_world(model), pfc::get_fft(model), std::forward<Fn>(fn));
}

/**
 * @brief Model overload: apply in-place with time to a named field
 */
template <typename Fn>
inline void apply_inplace_with_time(Model &model, std::string_view field_name,
                                    double t, Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  apply_inplace_with_time(f, pfc::get_world(model), pfc::get_fft(model), t,
                          std::forward<Fn>(fn));
}

/**
 * @brief Apply a coordinate-space function to a named model field (local inbox)
 *
 * Convenience overload that retrieves `field` and `world` from `model`.
 */
template <typename Fn>
inline void apply(Model &model, std::string_view field_name, Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  apply(f, pfc::get_world(model), pfc::get_fft(model), std::forward<Fn>(fn));
}

/**
 * @brief Apply a space-time function to a named model field (local inbox)
 */
template <typename Fn>
inline void apply_with_time(Model &model, std::string_view field_name, double t,
                            Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  apply_with_time(f, pfc::get_world(model), pfc::get_fft(model), t,
                  std::forward<Fn>(fn));
}

/**
 * @brief Apply a coordinate-space function over a real field laid out as a
 *        rank-owned **FD subdomain**.
 *
 * Companion to `apply()` for callers that work directly with a
 * `Decomposition` (no `IFFT` needed) — typically pure finite-difference apps
 * with their own halo exchange. Resizes `field` to `nx*ny*nz` and writes
 * `fn(x, y, z)` at every cell owned by `rank`, where `(x,y,z)` is the
 * physical coordinate computed from the global world's `origin/spacing`.
 *
 * Layout: row-major `[nx, ny, nz]` with **x varying fastest**, matching
 * `pfc::field::fd::*` and the `HaloExchanger` family.
 *
 * @tparam Fn Callable: `double(double, double, double)`.
 * @param field  Output field (resized to `nx*ny*nz`).
 * @param decomp Domain decomposition.
 * @param rank   MPI rank owning the subdomain.
 * @param fn     Coordinate function to evaluate at every owned cell.
 */
template <typename Fn>
inline void apply_subdomain(std::vector<double> &field,
                            const pfc::decomposition::Decomposition &decomp,
                            int rank, Fn &&fn) {
  const auto &gw = pfc::decomposition::get_world(decomp);
  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  const auto lo = pfc::world::get_lower(local);
  const auto sz = pfc::world::get_size(local);
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const std::size_t sxy =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  const auto origin = pfc::world::get_origin(gw);
  const auto spacing = pfc::world::get_spacing(gw);
  field.assign(sxy * static_cast<std::size_t>(nz), 0.0);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gi = lo[0] + ix;
        const int gj = lo[1] + iy;
        const int gk = lo[2] + iz;
        const double x = origin[0] + static_cast<double>(gi) * spacing[0];
        const double y = origin[1] + static_cast<double>(gj) * spacing[1];
        const double z = origin[2] + static_cast<double>(gk) * spacing[2];
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * sxy;
        field[idx] = static_cast<double>(fn(x, y, z));
      }
    }
  }
}

/**
 * @brief Iterate over the **interior** of a rank's FD subdomain, exposing
 *        physical coordinates and field values.
 *
 * Companion read-side helper to `apply_subdomain`: walks the slab
 * `[hw, nx-hw) x [hw, ny-hw) x [hw, nz-hw)` of `field` (sized to `nx*ny*nz`)
 * and calls `fn(coords, value)` for each interior cell, where `coords` is
 * the physical coordinate computed from the global world's `origin/spacing`
 * and `value` is `field[idx]`.
 *
 * Use this for reductions, error norms, post-processing — anything whose
 * loop matched the manual triple-nested pattern in early FD apps.
 * Cells in the per-rank halo region (`[0, hw)` and `[n-hw, n)`) are skipped,
 * matching the convention used by `pfc::field::FdGradient` and
 * `pfc::sim::for_each_interior`.
 *
 * Layout: row-major `[nx, ny, nz]` with **x varying fastest**.
 *
 * @tparam Fn Callable: `void(const Real3& coords, double value)`.
 * @param field       Input field (size `nx*ny*nz`).
 * @param decomp      Domain decomposition.
 * @param rank        MPI rank owning the subdomain.
 * @param halo_width  Width of the per-rank halo region to skip on each side.
 * @param fn          Function called once per interior cell.
 */
template <typename Fn>
inline void
for_each_interior_with_coords(const std::vector<double> &field,
                              const pfc::decomposition::Decomposition &decomp,
                              int rank, int halo_width, Fn &&fn) {
  const auto &gw = pfc::decomposition::get_world(decomp);
  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  const auto lo = pfc::world::get_lower(local);
  const auto sz = pfc::world::get_size(local);
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  const int kmin = hw;
  const int kmax = nz - hw;
  if (imin >= imax || jmin >= jmax || kmin >= kmax) {
    return;
  }
  const std::size_t sxy =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  const auto origin = pfc::world::get_origin(gw);
  const auto spacing = pfc::world::get_spacing(gw);
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const int gi = lo[0] + ix;
        const int gj = lo[1] + iy;
        const int gk = lo[2] + iz;
        const pfc::Real3 coords{origin[0] + static_cast<double>(gi) * spacing[0],
                                origin[1] + static_cast<double>(gj) * spacing[1],
                                origin[2] + static_cast<double>(gk) * spacing[2]};
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * sxy;
        fn(coords, field[idx]);
      }
    }
  }
}

} // namespace pfc::field
