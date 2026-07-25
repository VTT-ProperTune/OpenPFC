// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/world.hpp>

namespace pfc {

namespace domain {

// Create a World with just size (defaults: origin=0, spacing=1, fully periodic)
[[nodiscard]] world::World create_world(const Int3 &size) {
  const Int3 lower{0, 0, 0};
  const Int3 upper{size[0] - 1, size[1] - 1, size[2] - 1};

  // Create Domain and construct World with full-grid subdomain
  const Domain domain_obj = pfc::domain::create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                                                GridSpacing({1.0, 1.0, 1.0}));
  return world::World(lower, upper, domain_obj);
}

// Create a World with full strong-type specification
[[nodiscard]] world::World create_world(const GridSize &size,
                           const PhysicalOrigin &origin,
                           const GridSpacing &spacing,
                           const Bool3 &periodic) {
  const Int3 &raw_size = size.get();
  const Int3 lower{0, 0, 0};
  const Int3 upper{raw_size[0] - 1, raw_size[1] - 1, raw_size[2] - 1};

  // Create Domain and construct World with full-grid subdomain
  const Domain domain_obj = pfc::domain::create(size, origin, spacing, periodic);
  return world::World(lower, upper, domain_obj);
}

} // namespace domain

} // namespace pfc
