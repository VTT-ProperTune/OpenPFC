// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kspace.hpp
 * @brief K-space (Fourier space) helper functions for spectral methods
 *
 * @details
 * This file provides zero-cost helper functions for computing wave vector
 * components in Fourier space for FFT-based spectral methods. These functions
 * encapsulate the standard frequency scaling and Nyquist folding logic that
 * is duplicated across 120+ lines in multiple examples.
 *
 * The problem: Every spectral method example contains the same 30-line pattern:
 * @code
 * // Duplicated in 04_diffusion_model.cpp, 12_cahn_hilliard.cpp, etc.
 * double pi = std::atan(1.0) * 4.0;
 * double fx = 2.0 * pi / (spacing[0] * size[0]);
 * double fy = 2.0 * pi / (spacing[1] * size[1]);
 * double fz = 2.0 * pi / (spacing[2] * size[2]);
 * for (int k = o_low[2]; k <= o_high[2]; k++) {
 *   for (int j = o_low[1]; j <= o_high[1]; j++) {
 *     for (int i = o_low[0]; i <= o_high[0]; i++) {
 *       double ki = (i <= size[0] / 2) ? i * fx : (i - size[0]) * fx;
 *       double kj = (j <= size[1] / 2) ? j * fy : (j - size[1]) * fy;
 *       double kk = (k <= size[2] / 2) ? k * fz : (k - size[2]) * fz;
 *       double kLap = -(ki * ki + kj * kj + kk * kk);
 *       // ... use kLap ...
 *     }
 *   }
 * }
 * @endcode
 *
 * The solution: Extract reusable helpers:
 * @code
 * auto [fx, fy, fz] = k_frequency_scaling(world);
 * for (int k = o_low[2]; k <= o_high[2]; k++) {
 *   for (int j = o_low[1]; j <= o_high[1]; j++) {
 *     for (int i = o_low[0]; i <= o_high[0]; i++) {
 *       double ki = k_component(i, size[0], fx);
 *       double kj = k_component(j, size[1], fy);
 *       double kk = k_component(k, size[2], fz);
 *       double kLap = k_laplacian_value(ki, kj, kk);
 *       // ... use kLap ...
 *     }
 *   }
 * }
 * @endcode
 *
 * All functions are inline, noexcept, and zero-cost abstractions that compile
 * to identical machine code as the manual implementation.
 *
 * @see core/world.hpp for World coordinate system
 * @see fft.hpp for FFT interface
 * @see constants.hpp for mathematical constants (two_pi)
 *
 * @author OpenPFC Contributors
 * @date 2025-11-23
 */

#pragma once

#include <array>
#include <cmath>
#include <openpfc/constants.hpp>
#include <openpfc/core/world.hpp>

namespace pfc {
namespace fft {
namespace kspace {

/**
 * @brief Compute frequency scaling factors for each dimension.
 *
 * Calculates f = 2π / (spacing * size) for each dimension, which converts
 * grid indices to wave vector components in Fourier space.
 *
 * In spectral methods using FFT, derivatives in real space become
 * multiplications by wave vectors in Fourier space. The frequency scaling
 * establishes the relationship between grid indices and physical wave numbers.
 *
 * @tparam T World coordinate system type (e.g., CartesianTag)
 * @param world Simulation world containing grid size and spacing
 * @return Array of frequency scaling factors [fx, fy, fz] in rad/unit_length
 *
 * @note This is a zero-cost abstraction (inline, noexcept)
 * @note Works correctly for 1D, 2D, and 3D domains
 * @note Replaces: `double fx = 2.0 * pi / (spacing[0] * size[0]);`
 *
 * @code
 * auto world = world::uniform(128, 0.1);
 * auto [fx, fy, fz] = k_frequency_scaling(world);
 * // fx = fy = fz = 2π / (0.1 * 128) ≈ 0.490873...
 * @endcode
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
template <typename T>
inline std::array<double, 3>
k_frequency_scaling(const world::World<T> &world) noexcept {
  const auto spacing = world::get_spacing(world);
  const auto size = world::get_size(world);
  return {two_pi / (spacing[0] * size[0]), two_pi / (spacing[1] * size[1]),
          two_pi / (spacing[2] * size[2])};
}

/**
 * @brief Compute k-space wave vector component with Nyquist folding.
 *
 * Converts a grid index to the corresponding wave vector component, handling
 * the Nyquist frequency folding that occurs in real-to-complex FFTs.
 *
 * FFT convention:
 * - Indices 0 to size/2: Positive frequencies k = index * freq_scale
 * - Indices > size/2: Negative (aliased) frequencies k = (index-size) * freq_scale
 *
 * @param index Grid index (0 to size-1)
 * @param size Domain size in this dimension
 * @param freq_scale Frequency scaling factor (from k_frequency_scaling)
 * @return Wave vector component in rad/unit_length
 *
 * @note This is a zero-cost abstraction (inline, noexcept)
 * @note Handles both positive and negative frequencies correctly
 * @note Replaces: `double ki = (i <= size / 2) ? i * fx : (i - size) * fx;`
 *
 * @code
 * int Lx = 128;
 * double fx = two_pi / (0.1 * Lx);
 *
 * double k0 = k_component(0, Lx, fx);      // 0.0 (DC component)
 * double k1 = k_component(1, Lx, fx);      // 1*fx (first mode)
 * double k64 = k_component(64, Lx, fx);    // 64*fx (Nyquist)
 * double k127 = k_component(127, Lx, fx);  // -1*fx (negative)
 * @endcode
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
inline double k_component(int index, int size, double freq_scale) noexcept {
  return (index <= size / 2) ? index * freq_scale : (index - size) * freq_scale;
}

/**
 * @brief Compute Laplacian operator value -k² in Fourier space.
 *
 * Calculates -(k_x² + k_y² + k_z²), the Fourier representation of the
 * Laplacian operator ∇². Used in diffusion, heat, and phase-field equations.
 *
 * Mathematical relationship:
 * - Real space: ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
 * - Fourier space: ∇² ↔ -k² = -(k_x² + k_y² + k_z²)
 *
 * The negative sign is conventional, making the Laplacian negative definite.
 *
 * @param ki Wave vector x-component
 * @param kj Wave vector y-component
 * @param kk Wave vector z-component
 * @return Laplacian operator value -k² ≤ 0
 *
 * @note This is a zero-cost abstraction (inline, noexcept)
 * @note Returns negative value for non-zero k
 * @note Returns exactly 0.0 for DC component (ki=kj=kk=0)
 * @note Replaces: `double kLap = -(ki * ki + kj * kj + kk * kk);`
 *
 * @code
 * // Diffusion equation: ∂u/∂t = D∇²u
 * // In Fourier space: ∂û/∂t = D(-k²)û
 * double ki = k_component(i, Lx, fx);
 * double kj = k_component(j, Ly, fy);
 * double kk = k_component(k, Lz, fz);
 * double kLap = k_laplacian_value(ki, kj, kk);
 *
 * // Implicit time stepping
 * opL[idx] = 1.0 / (1.0 - D * dt * kLap);
 * @endcode
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
inline double k_laplacian_value(double ki, double kj, double kk) noexcept {
  return -(ki * ki + kj * kj + kk * kk);
}

/**
 * @brief Compute squared magnitude k² = k_x² + k_y² + k_z².
 *
 * Calculates the positive squared magnitude of the wave vector, useful for
 * higher-order operators, filtering, and wave number magnitudes.
 *
 * Mathematical relationships:
 * - k² = k_x² + k_y² + k_z² = |k|²
 * - |k| = √k² (wave number magnitude)
 * - Biharmonic: ∇⁴ ↔ k⁴ = (k²)²
 * - Related to Laplacian: k_laplacian_value(ki, kj, kk) = -k_squared_value(ki, kj,
 * kk)
 *
 * @param ki Wave vector x-component
 * @param kj Wave vector y-component
 * @param kk Wave vector z-component
 * @return k² = k_x² + k_y² + k_z² ≥ 0
 *
 * @note This is a zero-cost abstraction (inline, noexcept)
 * @note Returns positive value (magnitude squared)
 * @note Replaces: `double k2 = ki * ki + kj * kj + kk * kk;`
 *
 * @code
 * double ki = k_component(i, Lx, fx);
 * double kj = k_component(j, Ly, fy);
 * double kk = k_component(k, Lz, fz);
 *
 * double k2 = k_squared_value(ki, kj, kk);
 * double k_mag = std::sqrt(k2);  // Wave vector magnitude
 * double k4 = k2 * k2;  // Biharmonic operator
 * @endcode
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
inline double k_squared_value(double ki, double kj, double kk) noexcept {
  return ki * ki + kj * kj + kk * kk;
}

} // namespace kspace
} // namespace fft
} // namespace pfc
