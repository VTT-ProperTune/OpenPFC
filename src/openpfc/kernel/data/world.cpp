// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iomanip>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <stdexcept>

namespace pfc::world {

using pfc::Domain;

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

World::World(const Int3 &lower, const Int3 &upper, const Domain &domain)
    // calc_size validates lower <= upper and positivity; m_domain carries the
    // (global) coordinate system, its size aligned to this box for consistency.
    : m_box{lower, upper, calc_size(lower, upper)},
      m_domain{m_box.size, domain.spacing, domain.origin, domain.periodic} {}

// Strong-type API (PREFERRED) - type-safe World construction
// Uses GridSize, PhysicalOrigin, GridSpacing from strong_types.hpp
[[nodiscard]] CartesianWorld create(const GridSize &size,
                                    const PhysicalOrigin &origin,
                                    const GridSpacing &spacing,
                                    const pfc::types::Bool3 &periodic) {
  // The World's box spans the whole global grid; its coordinate system is the
  // canonical Domain (origin/spacing/per-axis periodicity all carried through).
  const Int3 &raw_size = size.get();
  Int3 lower{0, 0, 0};
  Int3 upper{raw_size[0] - 1, raw_size[1] - 1, raw_size[2] - 1};
  return World(lower, upper, pfc::domain::create(size, origin, spacing, periodic));
}

// old compatibility constructor taking only size, and default lower bounds and
// spacing and assuming pretty much everything else this is the most common use
// case
[[nodiscard]] CartesianWorld create(const Int3 &size) {
  Int3 lower{0, 0, 0};                               // default lower bounds
  Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1}; // default upper bounds
  return World(lower, upper, pfc::domain::create(size));
}

// Operators

std::ostream &operator<<(std::ostream &os, const World &w) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "World Summary\n";
  out << "  Size           : {" << w.m_box.size[0] << ", " << w.m_box.size[1] << ", "
      << w.m_box.size[2] << "}\n";
  out << "  Coordinate Sys : Cartesian\n";

  const auto &offset = w.m_domain.origin;
  const auto &spacing = w.m_domain.spacing;
  const auto &periodic = w.m_domain.periodic;
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
