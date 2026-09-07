// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kawahara_pointwise.inc
 * @brief Device instantiation of the Kawahara \(u^2\) remainder.
 */

#include <kawahara/kawahara_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(kawahara::KawaharaPointwise)
