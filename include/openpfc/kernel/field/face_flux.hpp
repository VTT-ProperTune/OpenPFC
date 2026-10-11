// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file face_flux.hpp
 * @brief Conservative face flux on a structured Cartesian grid.
 *
 * @details
 * For each face between two cells,
 *
 * \f[
 *   F = a_{\mathrm{face}}\,(p_{\mathrm{high}} - p_{\mathrm{low}}) / \Delta x,
 * \f]
 *
 * and the cell update is the difference of the face fluxes that bound it.
 * Internal faces cancel in a periodic sum, so `sum(rhs)` is zero to
 * round-off when every face value is shared by its two cells.
 *
 * Arithmetic averaging applies to any real samples. Harmonic averaging is
 * for a non-negative transport coefficient (mobility, diffusivity,
 * conductivity). It is not an interpolant for a signed coefficient: if
 * the two samples sum to zero or less, the face coefficient is zero.
 *
 * This operator does not know a material law, a pressure, or a boundary
 * condition, and it does not exchange halos. Neighbours outside the owned
 * box are read from width-1 separated face halos in the order
 * `+X,-X,+Y,-Y,+Z,-Z` (`halo_face_layout.hpp`). A periodic exchange is one
 * producer of those slots. A Neumann, reaction, or other non-periodic
 * boundary can fill the same slots. This function only reads them.
 *
 * An axis is skipped only when the caller marks it inactive, which means
 * the global extent is 1. A local owned extent of 1 on an active axis
 * still forms both face fluxes, both from the halos. A 2-D grid is the
 * case where the global z extent is 1.
 *
 * For a model-supplied face value that is not `a_face grad p` (diffusive
 * plus anti-trapping, for example), use `face_divergence` in
 * `face_current.hpp`. That operator does not replace this one.
 *
 * @see face_current.hpp
 */

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace pfc::field::fd {

/// How the two cell coefficients that share a face are combined.
enum class FaceAverage { Arithmetic, Harmonic };

/**
 * @brief Face coefficient from the samples on the low-index and high-index
 *        cells.
 *
 * `Arithmetic` is the mean of the two samples. `Harmonic` is for a
 * non-negative transport coefficient. It is zero when either sample is
 * zero and the other is positive, and it returns zero whenever
 * `low + high <= 0`. That rule is not a signed harmonic mean.
 */
[[nodiscard]] constexpr double average_face(FaceAverage kind, double low,
                                            double high) {
  switch (kind) {
  case FaceAverage::Harmonic: {
    const double sum = low + high;
    return (sum > 0.0) ? (2.0 * low * high / sum) : 0.0;
  }
  case FaceAverage::Arithmetic:
  default: return 0.5 * (low + high);
  }
}

/// @brief `a_face * (p_high - p_low) / spacing`.
[[nodiscard]] constexpr double face_flux(double face_coefficient, double p_high,
                                         double p_low, double spacing) {
  return face_coefficient * (p_high - p_low) / spacing;
}

/**
 * @brief Overwrite @p rhs with the conservative divergence of
 *        `a_face * grad p`.
 *
 * Owned arrays are x-fastest, length `nx*ny*nz`, and those extents are
 * the local owned box. `active_axes` is the global topology:
 * `{Nx > 1, Ny > 1, Nz > 1}`. A halo pointer may be null only on an
 * inactive axis, so those six slots stay raw pointers: a span would
 * require an extent for a pointer that is intentionally absent.
 * `dx`, `dy`, and `dz` are the cell spacings.
 */
inline void divergence_separated(
    std::span<const double> coefficient, std::span<const double> potential,
    const std::array<const double *, 6> &coefficient_halos,
    const std::array<const double *, 6> &potential_halos, std::span<double> rhs,
    int nx, int ny, int nz, std::array<bool, 3> active_axes, double dx, double dy,
    double dz, FaceAverage average) {
  if (nx <= 0 || ny <= 0 || nz <= 0) return;
  const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz);
  if (coefficient.size() != n || potential.size() != n || rhs.size() != n) {
    throw std::invalid_argument(
        "pfc::field::fd::divergence_separated: owned arrays must have length "
        "nx*ny*nz");
  }
  const std::ptrdiff_t sy = nx;
  const std::ptrdiff_t sz = static_cast<std::ptrdiff_t>(nx) * ny;
  const bool use_x = active_axes[0];
  const bool use_y = active_axes[1];
  const bool use_z = active_axes[2];
  const double *cpx = coefficient_halos[0];
  const double *cnx = coefficient_halos[1];
  const double *cpy = coefficient_halos[2];
  const double *cny = coefficient_halos[3];
  const double *cpz = coefficient_halos[4];
  const double *cnz = coefficient_halos[5];
  const double *ppx = potential_halos[0];
  const double *pnx = potential_halos[1];
  const double *ppy = potential_halos[2];
  const double *pny = potential_halos[3];
  const double *ppz = potential_halos[4];
  const double *pnz = potential_halos[5];

  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix) +
                                 static_cast<std::ptrdiff_t>(iy) * sy +
                                 static_cast<std::ptrdiff_t>(iz) * sz;
        const double ac = coefficient[c];
        const double pc = potential[c];
        double div = 0.0;
        if (use_x) {
          const std::ptrdiff_t face = static_cast<std::ptrdiff_t>(iz) * ny + iy;
          const double a_hi = (ix + 1 < nx) ? coefficient[c + 1] : cpx[face];
          const double a_lo = (ix - 1 >= 0) ? coefficient[c - 1] : cnx[face];
          const double p_hi = (ix + 1 < nx) ? potential[c + 1] : ppx[face];
          const double p_lo = (ix - 1 >= 0) ? potential[c - 1] : pnx[face];
          const double f_hi =
              face_flux(average_face(average, ac, a_hi), p_hi, pc, dx);
          const double f_lo =
              face_flux(average_face(average, a_lo, ac), pc, p_lo, dx);
          div += (f_hi - f_lo) / dx;
        }
        if (use_y) {
          const std::ptrdiff_t face = static_cast<std::ptrdiff_t>(iz) * nx + ix;
          const double a_hi = (iy + 1 < ny) ? coefficient[c + sy] : cpy[face];
          const double a_lo = (iy - 1 >= 0) ? coefficient[c - sy] : cny[face];
          const double p_hi = (iy + 1 < ny) ? potential[c + sy] : ppy[face];
          const double p_lo = (iy - 1 >= 0) ? potential[c - sy] : pny[face];
          const double f_hi =
              face_flux(average_face(average, ac, a_hi), p_hi, pc, dy);
          const double f_lo =
              face_flux(average_face(average, a_lo, ac), pc, p_lo, dy);
          div += (f_hi - f_lo) / dy;
        }
        if (use_z) {
          const std::ptrdiff_t face = static_cast<std::ptrdiff_t>(iy) * nx + ix;
          const double a_hi = (iz + 1 < nz) ? coefficient[c + sz] : cpz[face];
          const double a_lo = (iz - 1 >= 0) ? coefficient[c - sz] : cnz[face];
          const double p_hi = (iz + 1 < nz) ? potential[c + sz] : ppz[face];
          const double p_lo = (iz - 1 >= 0) ? potential[c - sz] : pnz[face];
          const double f_hi =
              face_flux(average_face(average, ac, a_hi), p_hi, pc, dz);
          const double f_lo =
              face_flux(average_face(average, a_lo, ac), pc, p_lo, dz);
          div += (f_hi - f_lo) / dz;
        }
        rhs[c] = div;
      }
    }
  }
}

} // namespace pfc::field::fd
