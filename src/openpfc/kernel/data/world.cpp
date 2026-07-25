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
    // calc_size validates lower <= upper and positivity; domain_ carries the
    // (global) coordinate system, its size aligned to this box for consistency.
    : subdomain_{lower, upper, calc_size(lower, upper)},
      domain_{calc_size(lower, upper), domain.spacing, domain.origin, domain.periodic} {}

// NOTE: world::create() implementations moved to pfc::domain::create() in
// include/openpfc/domain/create.hpp and src/kernel/data/world_factory.cpp.
// The world::create() functions in world_factory.hpp are now deprecated
// inline forwarders that call the new domain::create() functions.

// Operators

std::ostream &operator<<(std::ostream &os, const World &w) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "World Summary\n";
  out << "  Size           : {" << w.domain_.size[0] << ", " << w.domain_.size[1] << ", "
      << w.domain_.size[2] << "}\n";
  out << "  Coordinate Sys : Cartesian\n";

  const auto &origin = w.domain_.origin;
  const auto &spacing = w.domain_.spacing;
  const auto &periodic = w.domain_.periodic;
  out << "  Offset         : {" << origin[0] << ", " << origin[1] << ", "
      << origin[2] << "}\n";
  out << "  Spacing        : {" << spacing[0] << ", " << spacing[1] << ", "
      << spacing[2] << "}\n";
  out << "  Periodicity    : {" << (periodic[0] ? "true" : "false") << ", "
      << (periodic[1] ? "true" : "false") << ", "
      << (periodic[2] ? "true" : "false") << "}\n";

  return os << out.str();
}

} // namespace pfc::world
