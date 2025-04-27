// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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
