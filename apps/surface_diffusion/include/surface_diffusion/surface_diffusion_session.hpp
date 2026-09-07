// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file surface_diffusion_session.hpp
 * @brief Surface diffusion = `SurfaceDiffusionPhysics` on `SpectralETDSession`.
 *
 * Primary field is `h`. The PDE is linear, so dealiasing is off.
 */

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <surface_diffusion/cosine_mode.hpp>
#include <surface_diffusion/surface_diffusion_physics.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace surface_diffusion {

inline void register_catalog() {
  pfc::ui::register_field_modifier<CosineMode>("cosine_mode");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  opt.dealias = false;
  return opt;
}

class SurfaceDiffusionSession : public pfc::ui::SpectralETDSession<
                                    SurfaceDiffusionPhysics<double, pfc::HostSpace>,
                                    pfc::sim::stacks::SpectralCPUStack> {
public:
  using Base =
      pfc::ui::SpectralETDSession<SurfaceDiffusionPhysics<double, pfc::HostSpace>,
                                  pfc::sim::stacks::SpectralCPUStack>;

  SurfaceDiffusionSession(const nlohmann::json &settings, int rank, int nproc,
                          MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
class SurfaceDiffusionHIPSession
    : public pfc::ui::SpectralETDSession<
          SurfaceDiffusionPhysics<double, pfc::HIPSpace>,
          pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> {
public:
  using Base =
      pfc::ui::SpectralETDSession<SurfaceDiffusionPhysics<double, pfc::HIPSpace>,
                                  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;

  SurfaceDiffusionHIPSession(const nlohmann::json &settings, int rank, int nproc,
                             MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};
static_assert(
    pfc::sim::SpectralETDPhysics<SurfaceDiffusionPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace surface_diffusion
