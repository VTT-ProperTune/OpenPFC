// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/io/vtk_writer_validate.hpp>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace pfc::io::vtk_validate {

namespace {

[[nodiscard]] constexpr bool fits_half_open_extent(long long offset,
                                                   long long length,
                                                   long long upper_exclusive) {
  return offset >= 0 && length >= 0 && upper_exclusive >= 0 &&
         offset + length <= upper_exclusive;
}

[[nodiscard]] bool spacing_ok(double s) { return std::isfinite(s) && s > 0.0; }

[[nodiscard]] bool origin_ok(double x) { return std::isfinite(x); }

[[nodiscard]] bool vtk_extent_endpoint_safe(int begin_index,
                                            unsigned long long extent_len_ul) {
  if (extent_len_ul > static_cast<unsigned long long>(INT_MAX) + 1ULL) {
    return false;
  }
  const auto extent_len = static_cast<long long>(extent_len_ul);
  const auto endpoint = static_cast<long long>(begin_index) + extent_len - 1LL;
  return endpoint <= static_cast<long long>(INT_MAX);
}

} // namespace

void validate_origin_array(const std::array<double, 3> &origin,
                           const char *context_label) {
  for (int i = 0; i < 3; ++i) {
    if (!origin_ok(origin[i])) {
      throw std::invalid_argument(std::string(context_label) +
                                  ": origin components must be finite");
    }
  }
}

void validate_spacing_array(const std::array<double, 3> &spacing,
                            const char *context_label) {
  for (int i = 0; i < 3; ++i) {
    if (!spacing_ok(spacing[i])) {
      throw std::invalid_argument(
          std::string(context_label) +
          ": spacing components must be finite and positive");
    }
  }
}

void validate_writer_domain(const std::array<int, 3> &global_size,
                            const std::array<int, 3> &local_size,
                            const std::array<int, 3> &offset,
                            const std::array<double, 3> &origin,
                            const std::array<double, 3> &spacing) {
  for (int i = 0; i < 3; ++i) {
    const int g = global_size[i];
    const int l = local_size[i];
    const int o = offset[i];
    if (g <= 0 || l <= 0) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: global/local dimensions must be positive");
    }
    if (o < 0) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: offsets must be non-negative");
    }

    const auto gu = static_cast<long long>(g);
    if (!fits_half_open_extent(static_cast<long long>(o), static_cast<long long>(l),
                               gu)) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: piece Extent does not lie inside WholeExtent "
          "(check offset + local_size vs global)");
    }

    const unsigned long long extent_len_ul = static_cast<unsigned long long>(l);
    if (!vtk_extent_endpoint_safe(o, extent_len_ul)) {
      throw std::overflow_error(
          "VTKWriter::set_domain: VTK extents overflow int range");
    }
  }

  validate_origin_array(origin, "VTKWriter::set_domain");
  validate_spacing_array(spacing, "VTKWriter::set_domain");

  for (int i = 0; i < 3; ++i) {
    const long long whole_hi = static_cast<long long>(global_size[i]) - 1LL;
    if (whole_hi > static_cast<long long>(INT_MAX)) {
      throw std::overflow_error(
          "VTKWriter::set_domain: VTK WholeExtent overflows int range");
    }
  }
}

std::size_t expect_local_point_count(const std::array<int, 3> &local_size) {
  for (int i = 0; i < 3; ++i) {
    if (local_size[i] <= 0) {
      throw std::invalid_argument(
          "VTKWriter::write: domain not configured (call set_domain with "
          "positive local sizes)");
    }
  }

  unsigned long long n = 1;
  for (int i = 0; i < 3; ++i) {
    const auto li = static_cast<unsigned long long>(local_size[i]);
    const auto max_sz =
        static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)());
    if (n > max_sz / li) {
      throw std::overflow_error(
          "VTKWriter::write: local field size product overflows size_t");
    }
    n *= li;
  }
  if (n > static_cast<unsigned long long>(INT_MAX)) {
    throw std::overflow_error(
        "VTKWriter::write: local field element count exceeds INT_MAX");
  }
  return static_cast<std::size_t>(n);
}

} // namespace pfc::io::vtk_validate
