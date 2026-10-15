// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "homogenization_hip.hpp requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <inverse_homogenization/homogenization.hpp>
#include <openpfc/solvers/microelasticity/microelasticity_device.hpp>

namespace pfc::apps {

/// The device homogenizer used by the HIP drivers; same implementation as
/// the host `PeriodicHomogenizer`.
using PeriodicHomogenizerHIP = pfc::solvers::PeriodicHomogenizer<
    pfc::solvers::DeviceEigenstrainMicroelasticity<pfc::HIPSpace>>;

} // namespace pfc::apps
