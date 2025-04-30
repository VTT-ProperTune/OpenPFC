// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/world.hpp"
#include "world.hpp"
#include <iomanip>
#include <stdexcept>

namespace pfc {

// Constructor helpers

// compute the spacing based on size, lower bounds and upper bounds
Spacing3 compute_spacing(const Size3 &size, const LowerBounds3 &lower,
                         const UpperBounds3 &upper, const Periodic3 &periodic) {
  std::array<double, 3> spacing;
  for (std::size_t i = 0; i < 3; ++i) {
    int divisor = periodic.value[i] ? size.value[i] : size.value[i] - 1;
    if (divisor <= 0) {
      throw std::invalid_argument("Invalid size vs periodicity.");
    }
    spacing[i] = (upper.value[i] - lower.value[i]) / divisor;
    if (spacing[i] <= 0.0) {
      throw std::invalid_argument("Spacing must be positive.");
    }
  }
  return Spacing3{spacing};
}

// compute the upper bounds based on size, lower bounds and spacing
UpperBounds3 compute_upper(const Size3 &size, const LowerBounds3 &lower,
                           const Spacing3 &spacing, const Periodic3 &periodic) {
  std::array<double, 3> upper;
  for (std::size_t i = 0; i < 3; ++i) {
    int n = periodic.value[i] ? size.value[i] : size.value[i] - 1;
    upper[i] = lower.value[i] + spacing.value[i] * n;
  }
  return UpperBounds3{upper};
}

// Get the dimension of the coordinate system
std::size_t get_cs_dimension(CoordinateSystemTag cs) {
  switch (cs) {
  case CoordinateSystemTag::Line: return 1;
  case CoordinateSystemTag::Plane: return 2;
  case CoordinateSystemTag::Polar: return 2;
  case CoordinateSystemTag::Cartesian:
  case CoordinateSystemTag::Cylindrical:
  case CoordinateSystemTag::Spherical: return 3;
  }
  return 3;
}

Bool3 get_cs_periodicity(CoordinateSystemTag cs) {
  Bool3 periodic{true, true, true};
  // Cartesian is fully periodic
  switch (cs) {
  case CoordinateSystemTag::Line:
    // variables: x
    break;
  case CoordinateSystemTag::Plane:
    // variables: x, y
    periodic[2] = false;
    break;
  case CoordinateSystemTag::Cartesian:
    // variables: x, y, z
    break;
  case CoordinateSystemTag::Polar:
    // variables: r, theta, where theta is periodic
    periodic[0] = false; // r
    periodic[1] = true;  // theta
    break;
  case CoordinateSystemTag::Cylindrical:
    // variables: r, theta, z, where theta is periodic
    periodic[0] = false; // r is not periodic
    periodic[1] = true;  // theta is periodic
    periodic[2] = false; // z is not periodic
    break;
  case CoordinateSystemTag::Spherical:
    // variables: r, theta, phi, where theta and phi are periodic
    periodic[0] = false; // r is not periodic
    periodic[1] = true;  // theta is periodic
    periodic[2] = true;  // phi is periodic
    break;
  }
  return periodic;
}

// Constructors

World::World(const Int3 &dimensions, const Real3 &lower, const Real3 &upper,
             const Real3 &spacing, const Bool3 &periodic,
             CoordinateSystemTag coordinate_system)
    : m_size(dimensions), m_lower(lower), m_upper(upper), m_spacing(spacing),
      m_periodic(periodic), m_coordinate_system(coordinate_system) {}

// Old compatibility constructor taking size, lower bounds and spacing, the rest is
// calculated or assumed. These are a bit hazardous as the user must know the order
// of the arguments and the meaning of the parameters. The preferred way is to use
// the strong typedef constructors, which are more explicit and less error-prone.
World create_world(const Int3 &size, const Real3 &lower, const Real3 &spacing) {

  if (size[0] <= 0 || size[1] <= 0 || size[2] <= 0) {
    throw std::invalid_argument("Invalid dimensions. Lengths must be positive.");
  }

  if (spacing[0] <= 0 || spacing[1] <= 0 || spacing[2] <= 0) {
    throw std::invalid_argument("Invalid spacing. Values must be positive.");
  }

  // Assume coordinate system is Cartesian
  CoordinateSystemTag coordinate_system = CoordinateSystemTag::Cartesian;

  // Assume periodicity for dimensions based on coordinate system
  Bool3 periodic = get_cs_periodicity(coordinate_system);

  // Calculate upper bounds. Note: user must ensure the symmetricity of the domain
  // and the periodicity of the world manually.
  Real3 upper;
  for (std::size_t i = 0; i < 3; ++i) {
    upper[i] = lower[i] + spacing[i] * (size[i] - 1);
  }

  return World(size, lower, upper, spacing, periodic, coordinate_system);
}

// old compatibility constructor taking only size, and default lower bounds and
// spacing and assuming pretty much everything else this is the most common use
// case
World create_world(const Int3 &size) {
  return create_world(size, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
}

// Strong typedef constructors

// These are the preferred way to create a world to minimize the risk of confusion of
// the order of the parameters
World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const UpperBounds3 &upper, const Spacing3 &spacing,
                   const Periodic3 &periodic, const CoordinateSystemTag &cs) {
  return World(size.value, lower.value, upper.value, spacing.value, periodic.value,
               cs);
}

// We don't have to manually define the values for both upper bounds and spacing as
// we can calulcate one from another
World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const UpperBounds3 &upper, const Periodic3 &periodic,
                   const CoordinateSystemTag &cs) {
  Spacing3 spacing = compute_spacing(size, lower, upper, periodic);
  return create_world(size, lower, upper, spacing, periodic, cs);
}

World create_world(const Size3 &size, const LowerBounds3 &lower,
                   const Spacing3 &spacing, const Periodic3 &periodic,
                   const CoordinateSystemTag &cs) {
  UpperBounds3 upper = compute_upper(size, lower, spacing, periodic);
  return create_world(size, lower, upper, spacing, periodic, cs);
}

// This is the most common use case, where we assume the lower bounds are {0,0,0} and
// we have cartesian coordinate system with periodic boundaries and spacing is
// calculated from the size and lower bounds
World create_world(const Size3 &size, const UpperBounds3 &upper) {
  LowerBounds3 lower{{0.0, 0.0, 0.0}};
  CoordinateSystemTag coordinate_system = CoordinateSystemTag::Cartesian;
  Bool3 periodic_bool = get_cs_periodicity(coordinate_system);
  Periodic3 periodic(periodic_bool);
  Spacing3 spacing = compute_spacing(size, lower, upper, periodic);
  return create_world(size, lower, upper, spacing, periodic, coordinate_system);
}

// Strong typedefs for constructor clarity

Size3::Size3(const std::array<int, 3> &v) : value(v) {
  for (int dim : value) {
    if (dim <= 0) {
      throw std::invalid_argument("Size values must be positive.");
    }
  }
}

LowerBounds3::LowerBounds3(const std::array<double, 3> &v) : value(v) {}

UpperBounds3::UpperBounds3(const std::array<double, 3> &v) : value(v) {}

Spacing3::Spacing3(const std::array<double, 3> &v) : value(v) {
  for (double dim : value) {
    if (dim <= 0.0) {
      throw std::invalid_argument("Spacing values must be positive.");
    }
  }
}

Periodic3::Periodic3(const std::array<bool, 3> &v) : value(v) {}

// Operators

bool World::operator==(const World &other) const noexcept {
  return (*this).m_size == other.m_size && (*this).m_lower == other.m_lower &&
         (*this).m_spacing == other.m_spacing;
}

bool World::operator!=(const World &other) const noexcept {
  return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, const World &w) noexcept {
  os << std::fixed << std::setprecision(2); // Set fixed-point notation and precision
  os << "World Object Details:\n";
  os << "  Size: {" << w.m_size[0] << ", " << w.m_size[1] << ", " << w.m_size[2]
     << "}\n";
  os << "  Origin: {" << w.m_lower[0] << ", " << w.m_lower[1] << ", " << w.m_lower[2]
     << "}\n";
  os << "  Spacing: {" << w.m_spacing[0] << ", " << w.m_spacing[1] << ", "
     << w.m_spacing[2] << "}\n";
  os << "  Periodicity: {" << (w.m_periodic[0] ? "true" : "false") << ", "
     << (w.m_periodic[1] ? "true" : "false") << ", "
     << (w.m_periodic[2] ? "true" : "false") << "}\n";
  os << "  Coordinate System: " << static_cast<int>(w.m_coordinate_system) << "\n";
  return os;
}

// Getters

Int3 get_size(const World &w) noexcept { return w.m_size; }
size_t get_size(const World &w, int i) noexcept { return w.m_size[i]; }

Real3 get_origin(const World &w) noexcept { return w.m_lower; }
double get_origin(const World &w, int i) noexcept { return w.m_lower[i]; }

Real3 get_lower(const World &w) noexcept { return w.m_lower; }
double get_lower(const World &w, int i) noexcept { return w.m_lower[i]; }

Real3 get_upper(const World &w) noexcept { return w.m_upper; }
double get_upper(const World &w, int i) noexcept { return w.m_upper[i]; }

Real3 get_spacing(const World &w) noexcept { return w.m_spacing; }
double get_spacing(const World &w, int i) noexcept { return w.m_spacing[i]; }

const Bool3 &get_periodicity(const World &w) noexcept { return w.m_periodic; }
bool is_periodic(const World &w, int i) noexcept { return w.m_periodic[i]; }
const bool &get_periodicity(const World &w, int i) noexcept {
  return is_periodic(w, i);
}

CoordinateSystemTag get_coordinate_system(const World &w) noexcept {
  return w.m_coordinate_system;
}

// Get the total number of grid points in the world

int total_size(const World &w) noexcept {
  return get_size(w, 0) * get_size(w, 1) * get_size(w, 2);
}

// Conversion between physical coordinates and grid indices

Real3 to_coords(const World &w, const Int3 &indices) noexcept {
  Real3 coordinates;
  Real3 origin = get_origin(w);
  Real3 spacing = get_spacing(w);
  for (int i = 0; i < 3; ++i) {
    coordinates[i] = origin[i] + indices[i] * spacing[i];
  }
  return coordinates;
}

Int3 to_indices(const World &w, const Real3 &coordinates) noexcept {
  Int3 indices;
  Real3 origin = get_origin(w);
  Real3 spacing = get_spacing(w);
  for (int i = 0; i < 3; ++i) {
    indices[i] = static_cast<int>((coordinates[i] - origin[i]) / spacing[i]);
  }
  return indices;
}

} // namespace pfc
