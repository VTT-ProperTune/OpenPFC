// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kawahara_pointwise.hpp
 * @brief Real-space factor of the Kawahara flux: \(N=u^2\).
 *
 * Combined with \(M(k)=-i(\alpha/2)k_x\) this is \(-\alpha u\partial_x u\).
 */

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

namespace kawahara {

struct KawaharaPointwise {
  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return cell.psi * cell.psi;
  }
};

} // namespace kawahara
