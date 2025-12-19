// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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

#include <type_traits>

#include "openpfc/core/world.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/model.hpp"
#include "openpfc/types.hpp"
// Local iteration implemented inline to work with HeFFTe inbox type

namespace pfc {
namespace field {

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
inline void apply(RealField &field, const World &world, const FFT &fft, Fn &&fn) {
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
inline void apply_with_time(RealField &field, const World &world, const FFT &fft,
                            double t, Fn &&fn) {
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
inline void apply_inplace(RealField &field, const World &world, const FFT &fft,
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
                                    const FFT &fft, double t, Fn &&fn) {
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
  auto &f = model.get_real_field(field_name);
  apply_inplace(f, model.get_world(), model.get_fft(), std::forward<Fn>(fn));
}

/**
 * @brief Model overload: apply in-place with time to a named field
 */
template <typename Fn>
inline void apply_inplace_with_time(Model &model, std::string_view field_name,
                                    double t, Fn &&fn) {
  auto &f = model.get_real_field(field_name);
  apply_inplace_with_time(f, model.get_world(), model.get_fft(), t,
                          std::forward<Fn>(fn));
}

/**
 * @brief Apply a coordinate-space function to a named model field (local inbox)
 *
 * Convenience overload that retrieves `field` and `world` from `model`.
 */
template <typename Fn>
inline void apply(Model &model, std::string_view field_name, Fn &&fn) {
  auto &f = model.get_real_field(field_name);
  apply(f, model.get_world(), model.get_fft(), std::forward<Fn>(fn));
}

/**
 * @brief Apply a space-time function to a named model field (local inbox)
 */
template <typename Fn>
inline void apply_with_time(Model &model, std::string_view field_name, double t,
                            Fn &&fn) {
  auto &f = model.get_real_field(field_name);
  apply_with_time(f, model.get_world(), model.get_fft(), t, std::forward<Fn>(fn));
}

} // namespace field
} // namespace pfc
