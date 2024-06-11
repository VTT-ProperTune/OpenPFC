/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

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
