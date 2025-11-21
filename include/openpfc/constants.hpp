// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file constants.hpp
 * @brief Mathematical and physical constants
 *
 * Provides compile-time constants for common mathematical values used in
 * phase field simulations and spectral methods. All constants are `constexpr
 * double` for maximum precision and compile-time evaluation, resulting in zero
 * runtime overhead.
 *
 * Constants are organized in the `pfc::constants` namespace but are also
 * available directly in the `pfc` namespace via using declarations for
 * convenience.
 *
 * @code{.cpp}
 * #include <openpfc/constants.hpp>
 *
 * // Option 1: Namespace qualification
 * double k = 2.0 * pfc::pi / L;
 *
 * // Option 2: Using directive (recommended for many constants)
 * using namespace pfc::constants;
 * double k = 2.0 * pi / L;
 * double height = lattice_constant * sqrt3 / 2.0;
 *
 * // Option 3: Individual using (recommended for few constants)
 * using pfc::pi;
 * using pfc::sqrt2;
 * double k = 2.0 * pi / L;
 * double diagonal = side * sqrt2;
 * @endcode
 *
 * @note These constants match the naming convention of C++20's std::numbers
 * for easy migration when OpenPFC moves to C++20.
 */

#ifndef PFC_CONSTANTS_HPP
#define PFC_CONSTANTS_HPP

namespace pfc {

namespace constants {

// ============================================================================
// Mathematical Constants
// ============================================================================

/**
 * @brief The ratio of a circle's circumference to its diameter (π)
 *
 * π = 3.14159265358979323846...
 *
 * Commonly used in:
 * - FFT wave numbers: k = 2π/L
 * - Trigonometric functions
 * - Fourier transforms and spectral methods
 *
 * @code{.cpp}
 * // Wave number calculation in spectral methods
 * double k_x = 2.0 * pfc::pi / domain_length;
 *
 * // Fourier phase calculation
 * std::complex<double> phase = std::exp(
 *     std::complex<double>(0, pfc::two_pi * k_dot_r)
 * );
 * @endcode
 */
constexpr double pi = 3.14159265358979323846;

/**
 * @brief Two times pi (2π)
 *
 * 2π = 6.28318530717958647692...
 *
 * Commonly used in:
 * - FFT wave number calculations
 * - Full circle rotations
 * - Periodic boundary conditions
 *
 * @code{.cpp}
 * // Standard wave number formula in FFT
 * double k = pfc::two_pi / wavelength;
 * @endcode
 */
constexpr double two_pi = 2.0 * pi;

/**
 * @brief Pi divided by two (π/2)
 *
 * π/2 = 1.57079632679489661923...
 *
 * Commonly used in:
 * - Quarter circle rotations
 * - Trigonometric identities
 * - Phase shifts
 */
constexpr double pi_2 = pi / 2.0;

/**
 * @brief Pi divided by four (π/4)
 *
 * π/4 = 0.78539816339744830961...
 *
 * Commonly used in:
 * - Eighth circle rotations
 * - Diagonal angles (45 degrees)
 */
constexpr double pi_4 = pi / 4.0;

/**
 * @brief The reciprocal of pi (1/π)
 *
 * 1/π = 0.31830988618379067153...
 *
 * Commonly used in:
 * - Normalization factors
 * - Inverse FFT calculations
 */
constexpr double inv_pi = 1.0 / pi;

/**
 * @brief The square root of pi (√π)
 *
 * √π = 1.77245385090551602729...
 *
 * Commonly used in:
 * - Gaussian distributions
 * - Error function calculations
 * - Statistical mechanics
 */
constexpr double sqrt_pi = 1.77245385090551602729;

/**
 * @brief The square root of two (√2)
 *
 * √2 = 1.41421356237309504880...
 *
 * Commonly used in:
 * - Diagonal distances in square lattices
 * - FCC crystal nearest neighbor distances
 * - Normalization factors
 *
 * @code{.cpp}
 * // FCC nearest neighbor distance
 * double nearest = lattice_constant / pfc::sqrt2;
 * @endcode
 */
constexpr double sqrt2 = 1.41421356237309504880;

/**
 * @brief The square root of three (√3)
 *
 * √3 = 1.73205080756887729352...
 *
 * Commonly used in:
 * - Hexagonal lattice geometry
 * - HCP crystal structures
 * - Triangular lattice calculations
 *
 * @code{.cpp}
 * // Height of equilateral triangle
 * double height = side_length * pfc::sqrt3 / 2.0;
 *
 * // Hexagonal lattice spacing
 * double vertical_spacing = lattice_constant * pfc::sqrt3;
 * @endcode
 */
constexpr double sqrt3 = 1.73205080756887729352;

/**
 * @brief Euler's number (e)
 *
 * e = 2.71828182845904523536...
 *
 * Commonly used in:
 * - Exponential functions
 * - Growth and decay processes
 * - Natural logarithms
 *
 * @code{.cpp}
 * // Exponential decay
 * double concentration = initial * std::exp(-time / tau);
 * @endcode
 */
constexpr double e = 2.71828182845904523536;

/**
 * @brief The natural logarithm of 2 (ln(2))
 *
 * ln(2) = 0.69314718055994530941...
 *
 * Commonly used in:
 * - Half-life calculations
 * - Binary logarithm conversions
 * - Information theory
 */
constexpr double ln2 = 0.69314718055994530941;

/**
 * @brief The natural logarithm of 10 (ln(10))
 *
 * ln(10) = 2.30258509299404568401...
 *
 * Commonly used in:
 * - Base-10 to natural logarithm conversions
 * - pH calculations
 * - Decibel conversions
 */
constexpr double ln10 = 2.30258509299404568401;

/**
 * @brief The golden ratio (φ)
 *
 * φ = (1 + √5) / 2 = 1.61803398874989484820...
 *
 * Commonly used in:
 * - Quasicrystal structures
 * - Optimization algorithms (golden section search)
 * - Fibonacci sequence
 */
constexpr double phi = 1.61803398874989484820;

// ============================================================================
// Phase Field Crystal Lattice Constants
// ============================================================================

/**
 * @brief Lattice constant for 1D ordered phase (stripes)
 *
 * a1D = 2π
 *
 * Represents the characteristic wavelength of striped patterns in 1D PFC.
 */
constexpr double a1D = 2 * pi;

/**
 * @brief Lattice constant for 2D ordered phase (triangular lattice)
 *
 * a2D = 2π × 2/√3 = 4π/√3
 *
 * Represents the characteristic wavelength of triangular/hexagonal patterns
 * in 2D PFC simulations.
 */
constexpr double a2D = 2 * pi * 2 / sqrt3;

/**
 * @brief Lattice constant for 3D ordered phase (BCC lattice)
 *
 * a3D = 2π√2
 *
 * Represents the characteristic wavelength of body-centered cubic patterns
 * in 3D PFC simulations.
 */
constexpr double a3D = 2 * pi * sqrt2;

// ============================================================================
// FFT Configuration Constants
// ============================================================================

/**
 * @brief Direction for real-to-complex FFT symmetry reduction
 *
 * When the input of an FFT transform consists of all real numbers, the output
 * comes in conjugate pairs which can be exploited to reduce both the floating
 * point operations and MPI communications. Given a global set of indexes,
 * HeFFTe can compute the corresponding DFT and exploit the real-to-complex
 * symmetry by selecting a dimension and reducing the indexes by roughly half
 * (the exact formula is floor(n / 2) + 1).
 *
 * @warning Do not change this value. It is hardcoded in the HeFFTe integration.
 */
constexpr int r2c_direction = 0;

} // namespace constants

// ============================================================================
// Convenience: Import Common Constants into pfc Namespace
// ============================================================================
// This allows both pfc::pi and pfc::constants::pi to work

using constants::e;
using constants::pi;
using constants::sqrt2;
using constants::sqrt3;
using constants::two_pi;

} // namespace pfc

#endif // PFC_CONSTANTS_HPP
