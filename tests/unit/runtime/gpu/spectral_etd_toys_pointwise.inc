// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_etd_toys_pointwise.inc
 * @brief Device instantiations of the toy pointwise functors (stamped into
 *        spectral_etd_toys_pointwise.cu / .hip). One TU per consumer target.
 */

#include <fixtures/spectral_etd_toys_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(pfc::test::SHPointwise)
OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(pfc::test::MeanFieldToyPointwise)
OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(pfc::test::MovingFrameToyPointwise)
