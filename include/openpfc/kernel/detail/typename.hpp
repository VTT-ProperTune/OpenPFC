// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file typename.hpp
 * @brief Human-readable type names (kernel layer; no frontend dependency)
 */

#ifndef PFC_KERNEL_DETAIL_TYPENAME_HPP
#define PFC_KERNEL_DETAIL_TYPENAME_HPP

#include <complex>
#include <string>
#include <typeinfo>

namespace pfc {

template <typename T> struct TypeName {
  static std::string get() { return typeid(T).name(); }
};

template <> struct TypeName<int> {
  static std::string get() { return "int"; }
};

template <> struct TypeName<float> {
  static std::string get() { return "float"; }
};

template <> struct TypeName<double> {
  static std::string get() { return "double"; }
};

template <typename T> struct TypeName<std::complex<T>> {
  static std::string get() { return "complex<" + TypeName<T>::get() + ">"; }
};

} // namespace pfc

#endif
