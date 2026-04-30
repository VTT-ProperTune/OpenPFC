// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file vtk_writer_validate.hpp
 * @brief Domain and layout validation for VTK ImageData writers (VTKWriter)
 *
 * @details
 * All checks for global/local extents, offsets, origin, spacing, and local point
 * counts for parallel ImageData output live in this module. `VTKWriter` performs
 * XML generation and file/MPI I/O only; validation is not implemented there.
 */

#ifndef PFC_VTK_WRITER_VALIDATE_HPP
#define PFC_VTK_WRITER_VALIDATE_HPP

#include <array>
#include <cstddef>

namespace pfc::io::vtk_validate {

/**
 * @brief Validate global/local piece geometry and origin/spacing for ImageData
 *
 * @throws std::invalid_argument, std::overflow_error with messages prefixed for
 *         `VTKWriter::set_domain` or the given context where applicable.
 */
void validate_writer_domain(const std::array<int, 3> &global_size,
                            const std::array<int, 3> &local_size,
                            const std::array<int, 3> &offset,
                            const std::array<double, 3> &origin,
                            const std::array<double, 3> &spacing);

/**
 * @brief Ensure all origin components are finite
 */
void validate_origin_array(const std::array<double, 3> &origin,
                           const char *context_label);

/**
 * @brief Ensure all spacing components are finite and positive
 */
void validate_spacing_array(const std::array<double, 3> &spacing,
                            const char *context_label);

/**
 * @brief Product of local dimensions; throws if unset, overflow, or above INT_MAX
 *        point count (VTK constraint).
 */
[[nodiscard]] std::size_t
expect_local_point_count(const std::array<int, 3> &local_size);

} // namespace pfc::io::vtk_validate

#endif // PFC_VTK_WRITER_VALIDATE_HPP
