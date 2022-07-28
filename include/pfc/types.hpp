#pragma once

#include <array>
#include <vector>

namespace pfc {

template <class T> using Vec3 = std::array<T, 3>;
using Field = std::vector<double>;
// template <class T> using Field = std::vector<T>;

} // namespace pfc
