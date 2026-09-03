// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_pointwise.inc
 * @brief Device instantiation of tungsten's pointwise nonlinearity.
 *
 * Stamped into tungsten_pointwise.cu / .hip. This is the only tungsten code a
 * device compiler sees: the physics functor plus the library launcher.
 */

#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>
#include <tungsten/tungsten_pointwise.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(tungsten::TungstenPointwise)
