// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_session.hpp
 * @brief Aluminum = `AluminumPhysics` on the generic `SpectralETDSession`.
 *
 * One physics source, three backends. `register_catalog()` adds the
 * `seed_grid_fcc` initial condition and the `fixed` / `moving` boundary
 * conditions to the process-wide modifier catalog; call it once before
 * constructing a session.
 */

#include <aluminum/aluminum_physics.hpp>
#include <aluminum/seed_grid_fcc.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/solidification_bc_json.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace aluminum {

inline void register_catalog() {
  pfc::ui::register_field_modifier<SeedGridFCC>("seed_grid_fcc");
  pfc::ui::register_solidification_bcs();
}

using AluminumSession =
    pfc::ui::SpectralETDSession<AluminumPhysics<double, pfc::HostSpace>,
                                pfc::sim::stacks::SpectralCPUStack>;

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
using AluminumCUDASession =
    pfc::ui::SpectralETDSession<AluminumPhysics<double, pfc::CUDASpace>,
                                pfc::sim::stacks::GPUSpectralStack<pfc::CUDASpace>>;
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using AluminumHIPSession =
    pfc::ui::SpectralETDSession<AluminumPhysics<double, pfc::HIPSpace>,
                                pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;
#endif

} // namespace aluminum
