// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iomanip>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <stdexcept>

namespace pfc::world {

using pfc::csys::CartesianTag;
using pfc::csys::CoordinateSystem;
using CartesianCS = CoordinateSystem<CartesianTag>;

// Constructors

Int3 calc_size(const Int3 &lower, const Int3 &upper) {
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

World::World(const Int3 &lower, const Int3 &upper, const CartesianCS &cs)
    : m_lower(lower), m_upper(upper), m_size(calc_size(lower, upper)), m_cs(cs) {
  for (std::size_t i = 0; i < 3; ++i) {
    if (m_size[i] <= 0) {
      throw std::invalid_argument("Size values must be positive.");
    }
  }
}

// Strong-type API (PREFERRED) - type-safe World construction
// Uses GridSize, PhysicalOrigin, GridSpacing from strong_types.hpp
[[nodiscard]] CartesianWorld create(const GridSize &size,
                                    const PhysicalOrigin &origin,
                                    const GridSpacing &spacing,
                                    const pfc::types::Bool3 &periodic) {
  // Extract raw values (zero-cost - just references)
  const Int3 &raw_size = size.get();
  const Real3 &raw_origin = origin.get();
  const Real3 &raw_spacing = spacing.get();

  // Create world with extracted values. Periodicity is plumbed into the
  // coordinate system (previously silently dropped -> always all-periodic).
  Int3 lower{0, 0, 0};
  Int3 upper{raw_size[0] - 1, raw_size[1] - 1, raw_size[2] - 1};
  return World(lower, upper, CartesianCS(raw_origin, raw_spacing, periodic));
}

// old compatibility constructor taking only size, and default lower bounds and
// spacing and assuming pretty much everything else this is the most common use
// case
[[nodiscard]] CartesianWorld create(const Int3 &size) {
  Int3 lower{0, 0, 0};                               // default lower bounds
  Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1}; // default upper bounds
  return World(lower, upper, CartesianCS());
}

// Operators

inline const char *name_of(CartesianTag tag) {
  (void)tag;
  return "Cartesian";
}

std::ostream &operator<<(std::ostream &os, const World &w) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "World Summary\n";
  out << "  Size           : {" << w.m_size[0] << ", " << w.m_size[1] << ", "
      << w.m_size[2] << "}\n";
  out << "  Coordinate Sys : " << name_of(CartesianTag{}) << "\n";

  const auto &offset = w.m_cs.m_offset;
  const auto &spacing = w.m_cs.m_spacing;
  const auto &periodic = w.m_cs.m_periodic;
  out << "  Offset         : {" << offset[0] << ", " << offset[1] << ", "
      << offset[2] << "}\n";
  out << "  Spacing        : {" << spacing[0] << ", " << spacing[1] << ", "
      << spacing[2] << "}\n";
  out << "  Periodicity    : {" << (periodic[0] ? "true" : "false") << ", "
      << (periodic[1] ? "true" : "false") << ", " << (periodic[2] ? "true" : "false")
      << "}\n";

  return os << out.str();
}

} // namespace pfc::world
