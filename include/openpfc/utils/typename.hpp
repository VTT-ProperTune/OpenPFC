// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file typename.hpp
 * @brief Get human-readable type names at runtime
 *
 * @details
 * This header provides the TypeName<T> template struct for obtaining
 * human-readable string representations of C++ types.
 *
 * The generic template uses typeid().name() (compiler-dependent),
 * while specializations provide clean names for common types:
 * - int, float, double
 * - std::complex<float>, std::complex<double>
 *
 * This is useful for:
 * - Debug output and logging
 * - Error messages
 * - Runtime type inspection
 * - Generic template code that needs to report types
 *
 * @code
 * #include <openpfc/utils/typename.hpp>
 *
 * std::cout << pfc::TypeName<double>::get() << std::endl;  // "double"
 * std::cout << pfc::TypeName<std::complex<float>>::get();  // "complex<float>"
 * @endcode
 *
 * @see utils/show.hpp for usage in array display
 * @see utils.hpp for other utility functions
 *
 * @author OpenPFC Development Team
 * @date 2025
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
