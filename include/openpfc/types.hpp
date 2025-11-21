// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file types.hpp
 * @brief Common type aliases used throughout OpenPFC
 *
 * @details
 * This file defines standard type aliases for field data and containers:
 * - Field: std::vector<double> for real-valued field data
 * - RealField: Alias for Field
 * - ComplexField: std::vector<std::complex<double>> for Fourier space data
 * - RealFieldSet: Map of named real fields
 * - ComplexFieldSet: Map of named complex fields
 * - Vec3<T>: 3D vector (std::array<T, 3>)
 *
 * These aliases provide consistent types across the codebase and simplify
 * template instantiations.
 *
 * @see model.hpp for field management
 * @see array.hpp for multi-dimensional arrays
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_TYPES_HPP
#define PFC_TYPES_HPP

#include <array>
#include <complex>
#include <unordered_map>
#include <vector>

namespace pfc {

template <class T> using Vec3 = std::array<T, 3>;

using Field = std::vector<double>;
using RealField = std::vector<double>;
using RealFieldSet = std::unordered_map<std::string, RealField &>;
using ComplexField = std::vector<std::complex<double>>;
using ComplexFieldSet = std::unordered_map<std::string, ComplexField &>;

// template <class T> using Field = std::vector<T>;

} // namespace pfc

#endif
