// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file types.hpp
 * @brief Fundamental 3D array aliases shared across the kernel.
 *
 * @details
 * `Int3`, `Real3`, `Bool3` are the plain `std::array` aliases used throughout
 * OpenPFC for grid sizes, coordinates, and per-axis flags. They live in both
 * `pfc::types` and (re-exported) `pfc`.
 *
 * This is the 0.2 replacement for the old world-types header: its strong-type
 * structs (Size3/Periodic3/LowerBounds3/UpperBounds3/Spacing3) had no consumers
 * and were removed with the `World`→`Domain` migration (M1). The type-safe
 * construction wrappers that survive (`GridSize`/`PhysicalOrigin`/`GridSpacing`)
 * live in `strong_types.hpp`.
 */

#pragma once

#include <array>

namespace pfc {

namespace types {

using Int3 = std::array<int, 3>;
using Real3 = std::array<double, 3>;
using Bool3 = std::array<bool, 3>;

} // namespace types

using Int3 = types::Int3;
using Real3 = types::Real3;
using Bool3 = types::Bool3;

} // namespace pfc
