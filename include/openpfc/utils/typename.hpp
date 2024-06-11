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

#ifndef PFC_TYPENAME_HPP
#define PFC_TYPENAME_HPP

#include <complex>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace pfc {

// Type trait to retrieve the human-readable type name
template <typename T> struct TypeName {
  static std::string get() { return typeid(T).name(); }
};

// Specialization for int
template <> struct TypeName<int> {
  static std::string get() { return "int"; }
};

// Specialization for float
template <> struct TypeName<float> {
  static std::string get() { return "float"; }
};

// Specialization for double
template <> struct TypeName<double> {
  static std::string get() { return "double"; }
};

// Specialization for complex<double>
template <typename T> struct TypeName<std::complex<T>> {
  static std::string get() { return "complex<" + TypeName<T>::get() + ">"; }
};

} // namespace pfc

#endif
