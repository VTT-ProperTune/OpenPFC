// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "world.hpp"
#include <iomanip>
#include <stdexcept>

namespace pfc {

namespace world {

using pfc::csys::CartesianTag;
using pfc::csys::CoordinateSystem;
using CartesianCS = CoordinateSystem<CartesianTag>;

// Constructors

const Int3 calc_size(const Int3 &lower, const Int3 &upper) {
  Int3 size;
  for (std::size_t i = 0; i < 3; ++i) {
    if (lower[i] > upper[i]) {
      throw std::invalid_argument(
          "Lower bounds must be less than or equal to upper bounds.");
    }
    size[i] = upper[i] - lower[i] + 1;
    if (size[i] <= 0) {
      throw std::invalid_argument("Size values must be positive.");
    }
  }
  return size;
}

template <typename CoordTag>
World<CoordTag>::World(const Int3 &lower, const Int3 &upper,
                       const CoordinateSystem<CoordTag> &cs)
    : m_lower(lower), m_upper(upper), m_size(calc_size(lower, upper)), m_cs(cs) {
  for (std::size_t i = 0; i < 3; ++i) {
    if (m_size[i] <= 0) {
      throw std::invalid_argument("Size values must be positive.");
    }
  }
}

// Old compatibility constructor taking size, offset and spacing, the rest is
// calculated or assumed. These are a bit hazardous as the user must know the order
// of the arguments and the meaning of the parameters. The preferred way is to use
// the strong typedef constructors, which are more explicit and less error-prone.
CartesianWorld create(const Int3 &size, const Real3 &offset, const Real3 &spacing) {
  Int3 lower{0, 0, 0};                               // default lower bounds
  Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1}; // default upper bounds
  return World(lower, upper, CartesianCS(offset, spacing));
}

// old compatibility constructor taking only size, and default lower bounds and
// spacing and assuming pretty much everything else this is the most common use
// case
CartesianWorld create(const Int3 &size) {
  Int3 lower{0, 0, 0};                               // default lower bounds
  Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1}; // default upper bounds
  return World(lower, upper, CartesianCS());
}

// Strong typedef constructors. We don't necessary need these at all as now we
// have a separate World and CoordinateSystem making this less harazardous.

/*

// We don't have to manually define the values for both upper bounds and spacing as
// we can calulcate one from another

CartesianWorld create(const Size3 &size, const LowerBounds3 &lower,
                      const UpperBounds3 &upper, const Periodic3 &periodic) {
  Spacing3 spacing = compute_spacing(size, lower, upper, periodic);
  CartesianCS cs(lower.value, spacing.value, periodic.value);
  return World(size.value, cs);
}

CartesianWorld create(const Size3 &size, const LowerBounds3 &lower,
                      const Spacing3 &spacing, const Periodic3 &periodic) {
  CartesianCS cs(lower.value, spacing.value, periodic.value);
  return World(size.value, cs);
}
*/

// This is the most common use case, where we assume the lower bounds are {0,0,0} and
// we have cartesian coordinate system with periodic boundaries and spacing is
// calculated from the size and lower bounds
/*
CartesianWorld create(const Size3 &size, const UpperBounds3 &upper) {
  LowerBounds3 lower{{0.0, 0.0, 0.0}};
  Periodic3 periodic{{true, true, true}};
  Spacing3 spacing = compute_spacing(size, lower, upper, periodic);
  CartesianCS cs(lower.value, spacing.value, periodic.value);
  return World(size.value, cs);
}
*/

// Operators

// Equality operator
template <>
bool World<CartesianTag>::operator==(
    const World<CartesianTag> &other) const noexcept {
  return m_size == other.m_size && m_cs.m_offset == other.m_cs.m_offset &&
         m_cs.m_spacing == other.m_cs.m_spacing &&
         m_cs.m_periodic == other.m_cs.m_periodic;
}

// Inequality operator
template <typename CoordTag>
bool World<CoordTag>::operator!=(const World<CoordTag> &other) const noexcept {
  return !(*this == other);
}

inline const char *name_of(CartesianTag) { return "Cartesian"; }
// Add more as needed...

template <typename CoordTag>
std::ostream &operator<<(std::ostream &os, const World<CoordTag> &w) noexcept {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "World Summary\n";
  out << "  Size           : {" << w.m_size[0] << ", " << w.m_size[1] << ", "
      << w.m_size[2] << "}\n";
  out << "  Coordinate Sys : " << name_of(CoordTag{}) << "\n";

  if constexpr (std::is_same_v<CoordTag, CartesianTag>) {
    const auto &offset = w.m_cs.m_offset;
    const auto &spacing = w.m_cs.m_spacing;
    const auto &periodic = w.m_cs.m_periodic;
    out << "  Offset         : {" << offset[0] << ", " << offset[1] << ", "
        << offset[2] << "}\n";
    out << "  Spacing        : {" << spacing[0] << ", " << spacing[1] << ", "
        << spacing[2] << "}\n";
    out << "  Periodicity    : {" << (periodic[0] ? "true" : "false") << ", "
        << (periodic[1] ? "true" : "false") << ", "
        << (periodic[2] ? "true" : "false") << "}\n";
  }

  return os << out.str();
}

// Explicit instantiation of operator<< for CartesianTag
template std::ostream &
operator<< <CartesianTag>(std::ostream &os, const World<CartesianTag> &w) noexcept;

// Conversion between physical coordinates and grid indices

template <typename CoordTag>
const Real3 to_coords(const World<CoordTag> &w, const Int3 &indices) noexcept {
  return to_coords(w.m_coordinate_system, indices);
}

template <typename CoordTag>
const Int3 to_indices(const World<CoordTag> &w, const Real3 &coordinates) noexcept {
  return to_index(w.m_coordinate_system, coordinates);
}

Real3 get_lower(const CartesianWorld &w) noexcept {
  Int3 zero = {0, 0, 0};
  return to_coords(w.m_cs, zero);
}

Real3 get_upper(const CartesianWorld &w) noexcept {
  return to_coords(w.m_cs, get_size(w));
}

double get_lower(const CartesianWorld &w, int i) noexcept { return get_lower(w)[i]; }

double get_upper(const CartesianWorld &w, int i) noexcept { return get_upper(w)[i]; }

// Explicit instantiations for CartesianTag
template int total_size<CartesianTag>(const World<CartesianTag> &w) noexcept;
template Real3 to_coords<CartesianTag>(const World<CartesianTag> &w,
                                       const Int3 &indices) noexcept;
template Int3 to_indices<CartesianTag>(const World<CartesianTag> &w,
                                       const Real3 &coordinates) noexcept;
template bool
World<CartesianTag>::operator!=(const World<CartesianTag> &other) const noexcept;

} // namespace world
} // namespace pfc
