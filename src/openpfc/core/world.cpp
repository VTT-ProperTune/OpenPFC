// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/world.hpp"
#include <iomanip>
#include <stdexcept>

namespace pfc {

World::World(const Int3 &dimensions, const Real3 &origin, const Real3 &spacing)
    : m_size(dimensions), m_origin(origin), m_spacing(spacing) {
  if (m_size[0] <= 0 || m_size[1] <= 0 || m_size[2] <= 0) {
    throw std::invalid_argument("Invalid dimensions. Lengths must be positive.");
  }
  if (m_spacing[0] <= 0 || m_spacing[1] <= 0 || m_spacing[2] <= 0) {
    throw std::invalid_argument("Invalid spacing. Values must be positive.");
  }
}

World::World(const Int3 &dimensions) : World(dimensions, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}) {
}

World::Int3 World::get_size() const noexcept {
  return m_size;
}

World::Real3 World::get_origin() const noexcept {
  return m_origin;
}

World::Real3 World::get_spacing() const noexcept {
  return m_spacing;
}

World::Int3 World::size() const noexcept {
  return m_size;
}

World::Real3 World::origin() const noexcept {
  return m_origin;
}

World::Real3 World::spacing() const noexcept {
  return m_spacing;
}

int World::total_size() const noexcept {
  return m_size[0] * m_size[1] * m_size[2];
}

World::Real3 World::physical_coordinates(const Int3 &indices) const noexcept {
  return {m_origin[0] + indices[0] * m_spacing[0], m_origin[1] + indices[1] * m_spacing[1],
          m_origin[2] + indices[2] * m_spacing[2]};
}

World::Int3 World::grid_indices(const Real3 &coordinates) const noexcept {
  return {static_cast<int>((coordinates[0] - m_origin[0]) / m_spacing[0]),
          static_cast<int>((coordinates[1] - m_origin[1]) / m_spacing[1]),
          static_cast<int>((coordinates[2] - m_origin[2]) / m_spacing[2])};
}

bool World::operator==(const World &other) const noexcept {
  return m_size == other.m_size && m_origin == other.m_origin && m_spacing == other.m_spacing;
}

bool World::operator!=(const World &other) const noexcept {
  return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, const World &w) noexcept {
  os << std::fixed << std::setprecision(2); // Set fixed-point notation and precision
  os << "(size = {" << w.m_size[0] << ", " << w.m_size[1] << ", " << w.m_size[2] << "}";
  os << ", origin = {" << w.m_origin[0] << ", " << w.m_origin[1] << ", " << w.m_origin[2] << "}";
  os << ", spacing = {" << w.m_spacing[0] << ", " << w.m_spacing[1] << ", " << w.m_spacing[2] << "})";
  return os;
}

} // namespace pfc
