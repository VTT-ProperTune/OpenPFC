// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file higher_order_pfc_pointwise.inc
 * @brief Device instantiation of the higher-order PFC pointwise n(psi)
 *        (stamped into higher_order_pfc_pointwise.hip).
 */

#include <higher_order_pfc/higher_order_pfc_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(higher_order_pfc::HigherOrderPFCPointwise)
