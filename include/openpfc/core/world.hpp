// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

namespace pfc {

/**
 * @brief Create a World object with the specified dimensions, origin, and
 * spacing.
 */
World create_world(const World::Int3 &dimensions, const World::Real3 &origin, const World::Real3 &spacing) noexcept;

/**
 * @brief Create a World object with the specified dimensions and default
 * origin and spacing.
 *
 * Default origin is {0.0, 0.0, 0.0}, and default spacing is {1.0, 1.0, 1.0}.
 */
World create_world(const World::Int3 &dimensions) noexcept;

} // namespace pfc
