// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <ostream>
#include <stdexcept>

namespace pfc {
namespace world {

Spacing3 compute_spacing(const Size3 &size, const LowerBounds3 &lower,
                         const UpperBounds3 &upper, const Periodic3 &periodic);

UpperBounds3 compute_upper(const LowerBounds3 &lower, const Spacing3 &spacing,
                           const Size3 &size, const Periodic3 &periodic);

} // namespace world
} // namespace pfc
