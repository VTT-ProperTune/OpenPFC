// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_workspace.hpp
 * @brief Alias of `pfc::integrator::Workspace<T>` for stepper call sites.
 *
 * @details
 * Historical stepper-facing name. The single workspace type lives in
 * `openpfc/kernel/integrator/workspace.hpp`. Construct with
 * `(num_stages, local_size)` or `(extents, num_stages)`.
 */

#include <openpfc/kernel/integrator/workspace.hpp>

namespace pfc::sim::steppers {

template <typename T>
using StageWorkspace = pfc::integrator::Workspace<T>;

} // namespace pfc::sim::steppers
