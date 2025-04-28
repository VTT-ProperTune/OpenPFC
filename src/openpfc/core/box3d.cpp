// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/box3d.hpp"
#include <cassert>

namespace pfc {

// Constructor
Box3D::Box3D(const Int3 &lower, const Int3 &upper) : m_lower(lower), m_upper(upper) {
  for (int i = 0; i < 3; ++i) {
    if (m_lower[i] > m_upper[i]) {
      throw std::invalid_argument("Box3D: lower corner must not exceed upper corner in any dimension.");
    }
  }
}

// Accessors
const Box3D::Int3 &Box3D::lower() const noexcept {
  return m_lower;
}

const Box3D::Int3 &Box3D::upper() const noexcept {
  return m_upper;
}

// Size computation
Box3D::Int3 Box3D::size() const noexcept {
  return {m_upper[0] - m_lower[0] + 1, m_upper[1] - m_lower[1] + 1, m_upper[2] - m_lower[2] + 1};
}

// Total number of elements
int Box3D::total_size() const noexcept {
  Int3 s = size();
  return s[0] * s[1] * s[2];
}

// Check if index is inside the box
bool Box3D::contains(const Int3 &index) const noexcept {
  for (int i = 0; i < 3; ++i) {
    if (index[i] < m_lower[i] || index[i] > m_upper[i]) {
      return false;
    }
  }
  return true;
}

// Comparison operators
bool Box3D::operator==(const Box3D &other) const noexcept {
  return m_lower == other.m_lower && m_upper == other.m_upper;
}

bool Box3D::operator!=(const Box3D &other) const noexcept {
  return !(*this == other);
}

// Output stream
std::ostream &operator<<(std::ostream &os, const Box3D &box) noexcept {
  os << "Box3D(lower={" << box.m_lower[0] << ", " << box.m_lower[1] << ", " << box.m_lower[2] << "}, "
     << "upper={" << box.m_upper[0] << ", " << box.m_upper[1] << ", " << box.m_upper[2] << "})";
  return os;
}

} // namespace pfc
