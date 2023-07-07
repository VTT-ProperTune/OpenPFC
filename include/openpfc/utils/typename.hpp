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
