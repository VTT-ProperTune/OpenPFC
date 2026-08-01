// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file field_operations.hpp
 * @brief Functional, coordinate-space field operations
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
 */

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/model_free_functions.hpp>

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
 * @brief Apply a coordinate-space function over a real field (local inbox) with Domain
 *
 * @tparam Fn Callable: double(const Real3&) or double(Real3)
 * @param field Real-valued field storage (local inbox size)
 * @param domain Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param fn    Coordinate-space function returning new value
 */
template <typename Fn>
inline void apply(RealField &field, const Domain &domain, const fft::IFFT &fft,
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
        const auto x = pfc::domain::to_coords(domain, idx);
        field[linear_idx++] = fn(x);
      }
    }
  }
}

/**
 * @brief Apply a coordinate-space function over a real field (local inbox) with World (deprecated)
 *
 * @tparam Fn Callable: double(const Real3&) or double(Real3)
 * @param field Real-valued field storage (local inbox size)
 * @param world Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param fn    Coordinate-space function returning new value
 * @deprecated Use the Domain overload instead
 */
template <typename Fn>
[[deprecated("Use apply(field, domain, fft, fn) instead")]]
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
        field[linear_idx++] = fn(x);
      }
    }
  }
}

/**
 * @brief Apply a coordinate-space function in-place over a real field (local inbox) with Domain
 *
 * @tparam Fn Callable: double(const Real3&, double) or double(Real3, double)
 * @param field Real-valued field storage (local inbox size)
 * @param domain Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param fn    Coordinate-space function returning updated value
 */
template <typename Fn>
inline void apply_inplace(RealField &field, const Domain &domain, const fft::IFFT &fft,
                          Fn &&fn) {
  const auto inbox = pfc::fft::get_inbox(fft);
  // Safety: ensure field size matches inbox voxel count
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
        const auto x = pfc::domain::to_coords(domain, idx);
        field[linear_idx] = static_cast<double>(fn(x, field[linear_idx]));
        ++linear_idx;
      }
    }
  }
}

/**
 * @brief Apply a coordinate-space function in-place over a real field (local inbox) with World (deprecated)
 *
 * @tparam Fn Callable: double(const Real3&, double) or double(Real3, double)
 * @param field Real-valued field storage (local inbox size)
 * @param world Global domain descriptor
 * @param fft   FFT object (provides local inbox extents)
 * @param fn    Coordinate-space function returning updated value
 * @deprecated Use the Domain overload instead
 */
template <typename Fn>
[[deprecated("Use apply_inplace(field, domain, fft, fn) instead")]]
inline void apply_inplace(RealField &field, const World &world, const fft::IFFT &fft,
                          Fn &&fn) {
  const auto inbox = pfc::fft::get_inbox(fft);
  // Safety: ensure field size matches inbox voxel count
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
 * @brief Apply a space-time coordinate function to a named model field (local inbox)
 *
 * @tparam Fn Callable: double(const Real3&, double) or double(Real3, double)
 * @param model Model containing the field
 * @param field_name Name of the field to modify
 * @param time Current simulation time
 * @param fn Space-time coordinate function returning new value
 */
template <typename Fn>
inline void apply_with_time(Model &model, std::string_view field_name, double time,
                            Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  const auto domain = pfc::get_domain(model);
  const auto &fft = pfc::get_fft(model);
  const auto inbox = pfc::fft::get_inbox(fft);
  
  const auto nx = inbox.size[0];
  const auto ny = inbox.size[1];
  const auto nz = inbox.size[2];
  const auto expected = static_cast<size_t>(nx) * ny * nz;
  if (f.size() != expected) {
    throw std::invalid_argument(
        "field::apply_with_time: field size does not match FFT inbox size");
  }

  size_t linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const pfc::Int3 idx{i, j, k};
        const auto x = pfc::domain::to_coords(domain, idx);
        f[linear_idx++] = fn(x, time);
      }
    }
  }
}

/**
 * @brief Apply a space-time coordinate function in-place to a named model field
 *
 * @tparam Fn Callable: double(const Real3&, double, double) or double(Real3, double, double)
 * @param model Model containing the field
 * @param field_name Name of the field to modify
 * @param time Current simulation time
 * @param fn Space-time coordinate function returning updated value
 */
template <typename Fn>
inline void apply_inplace_with_time(Model &model, std::string_view field_name,
                                    double t, Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  const auto domain = pfc::get_domain(model);
  const auto &fft = pfc::get_fft(model);
  const auto inbox = pfc::fft::get_inbox(fft);
  
  const auto nx = inbox.size[0];
  const auto ny = inbox.size[1];
  const auto nz = inbox.size[2];
  const auto expected = static_cast<size_t>(nx) * ny * nz;
  if (f.size() != expected) {
    throw std::invalid_argument(
        "field::apply_inplace_with_time: field size does not match FFT inbox size");
  }

  size_t linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
        const pfc::Int3 idx{i, j, k};
        const auto x = pfc::domain::to_coords(domain, idx);
        f[linear_idx] = static_cast<double>(fn(x, f[linear_idx], t));
        ++linear_idx;
      }
    }
  }
}

/**
 * @brief Apply a coordinate-space function to a named model field (local inbox)
 *
 * Convenience overload that retrieves `field` and `world` from `model`.
 */
template <typename Fn>
inline void apply(Model &model, std::string_view field_name, Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  apply(f, pfc::get_domain(model), pfc::get_fft(model), std::forward<Fn>(fn));
}

/**
 * @brief Apply a coordinate-space function in-place to a named model field
 *
 * Convenience overload that retrieves `field` and `domain` from `model`.
 */
template <typename Fn>
inline void apply_inplace(Model &model, std::string_view field_name, Fn &&fn) {
  auto &f = pfc::get_real_field(model, field_name);
  apply_inplace(f, pfc::get_domain(model), pfc::get_fft(model), std::forward<Fn>(fn));
}

} // namespace pfc::field