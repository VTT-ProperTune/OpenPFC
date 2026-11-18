// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_session.hpp
 * @brief Aluminum = `AluminumPhysics` on the generic `SpectralETDSession`.
 *
 * One physics source, three backends, two precisions (`model.precision`,
 * ADR 0006). `register_catalog()` adds the
 * `seed_grid_fcc` initial condition and the `fixed` / `moving` boundary
 * conditions to the process-wide modifier catalog; call it once before
 * constructing a session.
 */

#include <aluminum/aluminum_physics.hpp>
#include <aluminum/seed_grid_fcc.hpp>
#include <aluminum/solidification_bc_json.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace aluminum {

inline void register_catalog() {
  pfc::ui::register_field_modifier<SeedGridFCC>("seed_grid_fcc");
  register_solidification_bcs();
}

using AluminumSession =
    pfc::ui::SpectralETDSession<AluminumPhysics<double, pfc::HostSpace>,
                                pfc::sim::stacks::SpectralCPUStack>;
/// Single-precision CPU session, selected by `model.precision = "float"`
/// (ADR 0006). Same physics and inputs; float fields, FFTs and tables.
using AluminumFloatSession =
    pfc::ui::SpectralETDSession<AluminumPhysics<float, pfc::HostSpace>,
                                pfc::sim::stacks::BasicSpectralCPUStack<float>>;

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
using AluminumCUDASession =
    pfc::ui::SpectralETDSession<AluminumPhysics<double, pfc::CUDASpace>,
                                pfc::sim::stacks::GPUSpectralStack<pfc::CUDASpace>>;
using AluminumCUDAFloatSession = pfc::ui::SpectralETDSession<
    AluminumPhysics<float, pfc::CUDASpace>,
    pfc::sim::stacks::GPUSpectralStack<pfc::CUDASpace, float>>;
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using AluminumHIPSession =
    pfc::ui::SpectralETDSession<AluminumPhysics<double, pfc::HIPSpace>,
                                pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;
using AluminumHIPFloatSession = pfc::ui::SpectralETDSession<
    AluminumPhysics<float, pfc::HIPSpace>,
    pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace, float>>;
#endif

} // namespace aluminum
