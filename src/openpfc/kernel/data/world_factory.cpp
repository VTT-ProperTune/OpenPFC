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

// Helper functions (moved from world_helpers.hpp)

// Create uniform grid with unit spacing at origin.
[[nodiscard]] world::World create_world_uniform(int size, Bool3 periodic) {
  if (size <= 0) {
    throw std::invalid_argument("Grid size must be positive, got: " +
                                std::to_string(size));
  }
  return create_world(GridSize({size, size, size}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({1.0, 1.0, 1.0}), periodic);
}

// Create uniform grid with specified spacing.
[[nodiscard]] world::World create_world_uniform(int size, double spacing, Bool3 periodic) {
  if (size <= 0) {
    throw std::invalid_argument("Grid size must be positive, got: " +
                                std::to_string(size));
  }
  if (spacing <= 0.0) {
    throw std::invalid_argument("Spacing must be positive, got: " +
                                std::to_string(spacing));
  }
  return create_world(GridSize({size, size, size}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({spacing, spacing, spacing}), periodic);
}

// Create grid from physical bounds (automatically computes spacing).
[[nodiscard]] world::World create_world_from_bounds(Int3 size, Real3 lower, Real3 upper,
                                                     Bool3 periodic) {
  // Validate inputs
  for (int i = 0; i < 3; ++i) {
    if (size[i] <= 0) {
      throw std::invalid_argument("Grid size must be positive in all dimensions");
    }
    if (upper[i] <= lower[i]) {
      throw std::invalid_argument("Upper bound must be greater than lower bound");
    }
  }

  // Compute spacing based on periodicity
  Real3 spacing;
  for (int i = 0; i < 3; ++i) {
    if (periodic[i]) {
      spacing[i] = (upper[i] - lower[i]) / size[i];
    } else {
      spacing[i] = (upper[i] - lower[i]) / (size[i] - 1);
    }
  }

  return create_world(GridSize(size), PhysicalOrigin(lower), GridSpacing(spacing),
                      periodic);
}

// Create grid with default origin but custom spacing.
[[nodiscard]] world::World create_world_with_spacing(Int3 size, Real3 spacing, Bool3 periodic) {
  // Validate
  for (int i = 0; i < 3; ++i) {
    if (size[i] <= 0) {
      throw std::invalid_argument("Grid size must be positive");
    }
    if (spacing[i] <= 0.0) {
      throw std::invalid_argument("Spacing must be positive");
    }
  }

  return create_world(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing(spacing), periodic);
}

// Create grid with custom origin but unit spacing.
[[nodiscard]] world::World create_world_with_origin(Int3 size, Real3 origin, Bool3 periodic) {
  // Validate
  for (int i = 0; i < 3; ++i) {
    if (size[i] <= 0) {
      throw std::invalid_argument("Grid size must be positive");
    }
  }

  return create_world(GridSize(size), PhysicalOrigin(origin),
                      GridSpacing({1.0, 1.0, 1.0}), periodic);
}

} // namespace domain

} // namespace pfc
