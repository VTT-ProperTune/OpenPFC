// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file padded_face_flux.hpp
 * @brief Host/device conservative diffusive face flux over padded fields.
 */

#include <cmath>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/face_flux.hpp>

namespace pfc::field::fd {

/**
 * @brief Non-owning device-safe `a_face * grad(p)` in local owned coordinates.
 *
 * The low-index cell of a face is `(i,j,k)`, including `-1` at a lower wall.
 * Values come from width-one-or-larger padded storage. Compose with
 * face_divergence for `+div(a grad p)`; select Zero walls explicitly for
 * impermeable boundaries and Model on MPI cuts and periodic seams.
 */
struct PaddedFaceFlux {
  const double *coefficient;
  const double *potential;
  std::ptrdiff_t coefficient_sy, coefficient_sz;
  std::ptrdiff_t potential_sy, potential_sz;
  double spacing[3];
  FaceAverage average;

  OPENPFC_HD double operator()(int axis, int i, int j, int k) const {
    const auto a = i + j * coefficient_sy + k * coefficient_sz;
    const auto p = i + j * potential_sy + k * potential_sz;
    const auto da = axis == 0 ? 1 : (axis == 1 ? coefficient_sy : coefficient_sz);
    const auto dp = axis == 0 ? 1 : (axis == 1 ? potential_sy : potential_sz);
    return face_flux(average_face(average, coefficient[a], coefficient[a + da]),
                     potential[p + dp], potential[p], spacing[axis]);
  }
};

/**
 * @brief Bind two compatible padded fields to a host/device face flux.
 *
 * @param coefficient Transport coefficient (non-negative for harmonic mean).
 * @param potential Scalar whose gradient drives the flux.
 * @param average Arithmetic or harmonic coefficient averaging.
 * @return Non-owning evaluator; fields and their buffers must outlive use.
 * @throws std::invalid_argument For incompatible geometry or missing padding.
 * @note Device fields synchronize stale storage to the device. This does not
 * exchange halos or impose boundary data. Prepare both fields before every
 * stage; no field-sized host readback, allocation or kernel launch is done.
 * The buffers must not be reallocated while the evaluator is in use.
 */
template <class Space>
[[nodiscard]] PaddedFaceFlux
padded_face_flux(data::Field<double, Space> &coefficient,
                 data::Field<double, Space> &potential,
                 FaceAverage average = FaceAverage::Arithmetic) {
  if (coefficient.storage_halo() < 1 || potential.storage_halo() < 1 ||
      coefficient.box().low != potential.box().low ||
      coefficient.box().size != potential.box().size ||
      coefficient.global_size() != potential.global_size() ||
      coefficient.spacing() != potential.spacing() ||
      coefficient.origin() != potential.origin()) {
    throw std::invalid_argument("padded_face_flux: incompatible padded fields");
  }
  const auto spacing = potential.spacing();
  for (int axis = 0; axis < 3; ++axis) {
    if (!(spacing[axis] > 0) || !std::isfinite(spacing[axis])) {
      throw std::invalid_argument("padded_face_flux: invalid spacing");
    }
  }
  if constexpr (!data::Field<double, Space>::is_host_space) {
    coefficient.sync_to_device();
    potential.sync_to_device();
  }
  return {coefficient.data() + coefficient.idx(0, 0, 0),
          potential.data() + potential.idx(0, 0, 0),
          coefficient.padded_extent(0),
          std::ptrdiff_t(coefficient.padded_extent(0)) *
              coefficient.padded_extent(1),
          potential.padded_extent(0),
          std::ptrdiff_t(potential.padded_extent(0)) * potential.padded_extent(1),
          {spacing[0], spacing[1], spacing[2]},
          average};
}

} // namespace pfc::field::fd
