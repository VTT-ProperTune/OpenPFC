// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_session.hpp
 * @brief Tungsten = `TungstenPhysics` on the generic `SpectralETDSession`.
 *
 * One physics source, three backends. Directional-solidification boundary
 * conditions (`fixed` / `moving`) are catalog modifiers local to this app;
 * call `tungsten::register_catalog()` once before constructing a session.
 */

#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <tungsten/solidification_bc_json.hpp>
#include <tungsten/tungsten_physics.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace tungsten {

/// Register tungsten's JSON `boundary_conditions` types on the default catalog.
inline void register_catalog() { register_solidification_bcs(); }

using TungstenSession =
    pfc::ui::SpectralETDSession<TungstenPhysics<double, pfc::HostSpace>,
                                pfc::sim::stacks::SpectralCPUStack>;
/// Single-precision CPU session, selected by `model.precision = "float"`
/// (ADR 0006). Same physics and inputs; float fields, FFTs and tables.
using TungstenFloatSession =
    pfc::ui::SpectralETDSession<TungstenPhysics<float, pfc::HostSpace>,
                                pfc::sim::stacks::BasicSpectralCPUStack<float>>;

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
using TungstenCUDASession =
    pfc::ui::SpectralETDSession<TungstenPhysics<double, pfc::CUDASpace>,
                                pfc::sim::stacks::GPUSpectralStack<pfc::CUDASpace>>;
using TungstenCUDAFloatSession = pfc::ui::SpectralETDSession<
    TungstenPhysics<float, pfc::CUDASpace>,
    pfc::sim::stacks::GPUSpectralStack<pfc::CUDASpace, float>>;
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using TungstenHIPSession =
    pfc::ui::SpectralETDSession<TungstenPhysics<double, pfc::HIPSpace>,
                                pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;
using TungstenHIPFloatSession = pfc::ui::SpectralETDSession<
    TungstenPhysics<float, pfc::HIPSpace>,
    pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace, float>>;
#endif

} // namespace tungsten
