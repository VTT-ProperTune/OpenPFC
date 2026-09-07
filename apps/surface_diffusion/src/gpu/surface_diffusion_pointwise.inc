// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file surface_diffusion_pointwise.inc
 * @brief Device instantiation of the zero Mullins remainder.
 */

#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>
#include <surface_diffusion/surface_diffusion_pointwise.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(surface_diffusion::SurfaceDiffusionPointwise)
