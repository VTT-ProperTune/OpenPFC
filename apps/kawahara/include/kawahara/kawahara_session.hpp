// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kawahara_session.hpp
 * @brief Kawahara = `KawaharaPhysics` on `SpectralETDSession`.
 *
 * Primary field is `u`. Orszag 2/3-rule dealiasing is on because \(u^2\)
 * is a quadratic product.
 */

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <kawahara/cosine_mode.hpp>
#include <kawahara/gaussian_pulse.hpp>
#include <kawahara/kawahara_physics.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace kawahara {

inline void register_catalog() {
  pfc::ui::register_field_modifier<CosineMode>("cosine_mode");
  pfc::ui::register_field_modifier<GaussianPulse>("gaussian_pulse");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "u";
  opt.dealias = true;
  return opt;
}

class KawaharaSession
    : public pfc::ui::SpectralETDSession<KawaharaPhysics<double, pfc::HostSpace>,
                                         pfc::sim::stacks::SpectralCPUStack> {
public:
  using Base = pfc::ui::SpectralETDSession<KawaharaPhysics<double, pfc::HostSpace>,
                                           pfc::sim::stacks::SpectralCPUStack>;

  KawaharaSession(const nlohmann::json &settings, int rank, int nproc,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
class KawaharaHIPSession : public pfc::ui::SpectralETDSession<
                               KawaharaPhysics<double, pfc::HIPSpace>,
                               pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> {
public:
  using Base =
      pfc::ui::SpectralETDSession<KawaharaPhysics<double, pfc::HIPSpace>,
                                  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;

  KawaharaHIPSession(const nlohmann::json &settings, int rank, int nproc,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};
static_assert(pfc::sim::SpectralETDPhysics<KawaharaPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace kawahara
