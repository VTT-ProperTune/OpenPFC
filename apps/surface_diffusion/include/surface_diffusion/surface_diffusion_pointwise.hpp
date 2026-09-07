// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file surface_diffusion_pointwise.hpp
 * @brief Mullins surface diffusion is linear; the real-space remainder is 0.
 */

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace surface_diffusion {

struct SurfaceDiffusionPointwise {
  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &) const {
    return 0.0;
  }
};

} // namespace surface_diffusion
