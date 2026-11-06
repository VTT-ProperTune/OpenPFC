// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file affine_map.hpp
 * @brief Constant affine map from a Cartesian reference cell to physical space.
 *
 * Reference coordinates `ξ` are the coordinates of the structured Cartesian
 * cell that the grid and the periodic FFT already index. Physical coordinates
 * are `x = A ξ + b`. Reference components are not physical Cartesian
 * components: gradients and wavevectors transform as covectors,
 * `grad_x u = A^{-T} grad_ξ u` and `k_x = A^{-T} k_ξ`.
 *
 * The Laplacian symbol on that Fourier cell is
 * `-k_ξ^T (A^{-1} A^{-T}) k_ξ`. The translation `b` does not enter it.
 * When `A` is exactly the identity, `uses_cartesian_laplacian()` is true and
 * the spectral operator calls `k_laplacian_value` instead of forming a metric.
 * A spatially varying curvilinear map is a different type.
 *
 * `A` is stored column-major. Column `j` is the image of reference basis
 * vector `e_j`. The volume weight is `|det A|`, so `dV_x = |det A| dV_ξ`.
 * Factories validate on the host. Cached geometry fields are read-only: an
 * unchecked mutation must not invalidate the inverse, determinant, or
 * identity fast-path while retained downstream code still reads the cache.
 */

#include <openpfc/kernel/geometry/vec3.hpp>

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace pfc::geometry {

/**
 * @brief Trivially copyable constant map `x = A ξ + b`.
 *
 * Device methods are non-throwing arithmetic. `affine_map_from_matrix`
 * rejects a non-finite or singular matrix on the host.
 */
class AffineMap {
public:
  /// Column-major Jacobian `A = dx/dξ`. Column `j` is `matrix[j * 3 + i]`.
  const double matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  /// Column-major `A^{-1}`; immutable with the Jacobian.
  const double inverse[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  /// Translation `b` in physical coordinates.
  const double translation[3] = {0, 0, 0};
  /// `det A`. Identity is exactly 1.
  const double determinant = 1;
  /// True only when `A` is exactly `I`. The Laplacian then skips the metric.
  const bool cartesian = true;

  OPENPFC_HD AffineMap() = default;

private:
  friend AffineMap affine_map_from_matrix(const double *, Vec3);

  // Host-only validated construction. The explicit raw-array member
  // initializers preserve device-friendly indexed access without mutability.
  // clang-format off
  AffineMap(const double a[9], const double inv[9], Vec3 shift, double det,
            bool is_cartesian)
      : matrix{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]},
        inverse{inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6],
                inv[7], inv[8]},
        translation{shift.x, shift.y, shift.z}, determinant(det),
        cartesian(is_cartesian) {}
  // clang-format on

public:
  OPENPFC_HD Vec3 column(int j) const noexcept {
    return {matrix[j * 3], matrix[j * 3 + 1], matrix[j * 3 + 2]};
  }

  OPENPFC_HD Vec3 inverse_column(int j) const noexcept {
    return {inverse[j * 3], inverse[j * 3 + 1], inverse[j * 3 + 2]};
  }

  OPENPFC_HD bool uses_cartesian_laplacian() const noexcept { return cartesian; }

  OPENPFC_HD double volume_scale() const noexcept { return absd(determinant); }

  /// `x = A ξ + b`.
  OPENPFC_HD Vec3 to_physical(Vec3 xi) const noexcept {
    return column(0) * xi.x + column(1) * xi.y + column(2) * xi.z +
           Vec3{translation[0], translation[1], translation[2]};
  }

  /// `ξ = A^{-1} (x - b)`.
  OPENPFC_HD Vec3 to_reference(Vec3 x) const noexcept {
    const Vec3 d{x.x - translation[0], x.y - translation[1], x.z - translation[2]};
    return inverse_column(0) * d.x + inverse_column(1) * d.y +
           inverse_column(2) * d.z;
  }

  /// `grad_x u = A^{-T} grad_ξ u`.
  OPENPFC_HD Vec3 physical_gradient(Vec3 reference_gradient) const noexcept {
    return {dot(inverse_column(0), reference_gradient),
            dot(inverse_column(1), reference_gradient),
            dot(inverse_column(2), reference_gradient)};
  }

  /// `k_x = A^{-T} k_ξ`. Same map as `physical_gradient`.
  OPENPFC_HD Vec3 physical_wavevector(Vec3 reference_wavevector) const noexcept {
    return physical_gradient(reference_wavevector);
  }

  /// `k_ξ = A^T k_x`.
  OPENPFC_HD Vec3 reference_wavevector(Vec3 physical_k) const noexcept {
    return {dot(column(0), physical_k), dot(column(1), physical_k),
            dot(column(2), physical_k)};
  }

  /// Reference metric `A^T A`, column-major.
  OPENPFC_HD void metric(double out[9]) const noexcept {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        out[j * 3 + i] = dot(column(i), column(j));
      }
    }
  }

  /// Inverse reference metric `A^{-1} A^{-T}`, column-major.
  OPENPFC_HD void inverse_metric(double out[9]) const noexcept {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        double sum = 0;
        for (int k = 0; k < 3; ++k) {
          sum += inverse[k * 3 + i] * inverse[k * 3 + j];
        }
        out[j * 3 + i] = sum;
      }
    }
  }

  /**
   * @brief `-k_ξ^T (A^{-1} A^{-T}) k_ξ`, including when `A` is `I`.
   *
   * The Cartesian fast path does not call this. It calls `k_laplacian_value`.
   */
  OPENPFC_HD double metric_laplacian_symbol(double kx, double ky,
                                            double kz) const noexcept {
    const Vec3 kp = physical_wavevector({kx, ky, kz});
    return -((kp.x * kp.x) + (kp.y * kp.y) + (kp.z * kp.z));
  }
};

static_assert(std::is_trivially_copyable_v<AffineMap>);

/// Identity Jacobian and zero translation. Valid on the device.
OPENPFC_HD inline AffineMap identity_affine_map() noexcept { return {}; }

/// `dV_x` integrated over a reference region of volume `reference_volume`.
OPENPFC_HD inline double physical_volume(const AffineMap &map,
                                         double reference_volume) noexcept {
  return map.volume_scale() * reference_volume;
}

/// Host factory. `matrix` is column-major and must contain nine finite entries.
[[nodiscard]] inline AffineMap affine_map_from_matrix(const double matrix[9],
                                                      Vec3 translation) {
  if (matrix == nullptr || !std::isfinite(translation.x) ||
      !std::isfinite(translation.y) || !std::isfinite(translation.z)) {
    throw std::invalid_argument("affine map matrix or translation is not finite");
  }
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(matrix[i])) {
      throw std::invalid_argument("affine map matrix or translation is not finite");
    }
  }

  const double eye[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool cartesian = true;
  for (int i = 0; i < 9; ++i) {
    if (matrix[i] != eye[i]) cartesian = false;
  }
  if (cartesian) {
    return AffineMap(matrix, eye, translation, 1.0, true);
  }

  const double a00 = matrix[0];
  const double a10 = matrix[1];
  const double a20 = matrix[2];
  const double a01 = matrix[3];
  const double a11 = matrix[4];
  const double a21 = matrix[5];
  const double a02 = matrix[6];
  const double a12 = matrix[7];
  const double a22 = matrix[8];
  const double det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) +
                     a02 * (a10 * a21 - a11 * a20);
  if (!std::isfinite(det) || det == 0.0) {
    throw std::invalid_argument("affine map matrix is singular");
  }
  const double inv_det = 1.0 / det;
  const double inv[9] = {
      (a11 * a22 - a12 * a21) * inv_det, (a12 * a20 - a10 * a22) * inv_det,
      (a10 * a21 - a11 * a20) * inv_det, (a02 * a21 - a01 * a22) * inv_det,
      (a00 * a22 - a02 * a20) * inv_det, (a01 * a20 - a00 * a21) * inv_det,
      (a01 * a12 - a02 * a11) * inv_det, (a02 * a10 - a00 * a12) * inv_det,
      (a00 * a11 - a01 * a10) * inv_det};
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(inv[i])) {
      throw std::invalid_argument("affine map matrix is singular");
    }
  }
  return AffineMap(matrix, inv, translation, det, false);
}

[[nodiscard]] inline AffineMap affine_map_from_columns(Vec3 c0, Vec3 c1, Vec3 c2,
                                                       Vec3 translation) {
  const double matrix[9] = {c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z};
  return affine_map_from_matrix(matrix, translation);
}

} // namespace pfc::geometry
