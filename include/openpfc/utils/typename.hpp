#ifndef PFC_TYPENAME_HPP
#define PFC_TYPENAME_HPP

#include <string>
#include <type_traits>
#include <typeinfo>

namespace pfc {
namespace utils {

// Helper function to demangle the type name
std::string demangle(const char *name) {
  return name;
}

// Type trait to retrieve the human-readable type name
template <typename T> struct TypeName {
  static std::string get() { return demangle(typeid(T).name()); }
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

} // namespace utils
} // namespace pfc

#endif
