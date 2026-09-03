// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file aluminum_pointwise.inc
 * @brief Device instantiation of aluminum's pointwise nonlinearity
 *        (stamped into aluminum_pointwise.cu / .hip).
 */

#include <aluminum/aluminum_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(aluminum::AluminumPointwise)
