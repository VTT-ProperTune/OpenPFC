// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file cahn_hilliard_pointwise.inc
 * @brief Device instantiation of Cahn–Hilliard pointwise n(c)
 *        (stamped into cahn_hilliard_pointwise.hip).
 */

#include <cahn_hilliard/cahn_hilliard_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(cahn_hilliard::CahnHilliardPointwise)
