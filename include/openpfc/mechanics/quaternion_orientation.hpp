// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file quaternion_orientation.hpp
 * @brief Crystal-to-sample unit quaternion to passive FCC sample-to-crystal Q.
 *
 * q=(w,x,y,z) follows the SolidificationLab KWC3D-Q convention: it actively
 * maps vectors from the crystal frame to the sample/lab frame. The FCC law
 * requires Q=R(q)^T. A 3x3 field is never materialized; the tensor lives only
 * within a constitutive evaluation. Quaternion sign is a gauge: q and -q
 * represent the same orientation.
 */

#include <cmath>
#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

/// Returns false for missing, nonfinite or non-unit quaternions. Accepted
/// small roundoff in q is renormalized; this is not a physical approximation.
[[nodiscard]] OPENPFC_INLINE_HD bool
quaternion_sample_to_crystal(const double (&q)[4], Tensor2 &Q) noexcept {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  if (!std::isfinite(w) || !std::isfinite(x) || !std::isfinite(y) ||
      !std::isfinite(z))
    return false;
  const double norm2 = w * w + x * x + y * y + z * z;
  if (!(norm2 > 0.0) || !std::isfinite(norm2) ||
      !(std::abs(norm2 - 1.0) <= 1e-5))
    return false;
  const double inv = 1.0 / std::sqrt(norm2);
  const double a = w * inv, b = x * inv, c = y * inv, d = z * inv;
  Q(0, 0) = 1.0 - 2.0 * (c * c + d * d);
  Q(0, 1) = 2.0 * (b * c + a * d);
  Q(0, 2) = 2.0 * (b * d - a * c);
  Q(1, 0) = 2.0 * (b * c - a * d);
  Q(1, 1) = 1.0 - 2.0 * (b * b + d * d);
  Q(1, 2) = 2.0 * (c * d + a * b);
  Q(2, 0) = 2.0 * (b * d + a * c);
  Q(2, 1) = 2.0 * (c * d - a * b);
  Q(2, 2) = 1.0 - 2.0 * (b * b + c * c);
  return true;
}

} // namespace pfc::finite_strain
