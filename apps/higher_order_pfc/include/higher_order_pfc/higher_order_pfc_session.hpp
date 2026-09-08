// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file higher_order_pfc_session.hpp
 * @brief Higher-order PFC = `HigherOrderPFCPhysics` on `SpectralETDSession`.
 *
 * @details
 * Primary field is `psi`. Orszag 2/3-rule dealiasing is on because the local
 * term is cubic: \f$\psi^3\f$ triples the highest wave number present, so
 * without dealiasing the aliased content folds straight back onto the
 * \f$k\approx1\f$ band the kernel is trying to select.
 *
 * `register_catalog()` adds `cosine_mode` and `seeded_noise`. HIP builds also
 * define `HigherOrderPFCHIPSession` on `GPUSpectralStack<HIPSpace>`.
 */

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <higher_order_pfc/cosine_mode.hpp>
#include <higher_order_pfc/higher_order_pfc_physics.hpp>
#include <higher_order_pfc/seeded_noise.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace higher_order_pfc {

inline void register_catalog() {
  pfc::ui::register_field_modifier<CosineMode>("cosine_mode");
  pfc::ui::register_field_modifier<SeededNoise>("seeded_noise");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "psi";
  opt.dealias = true;
  return opt;
}

class HigherOrderPFCSession
    : public pfc::ui::SpectralETDSession<HigherOrderPFCPhysics<double, pfc::HostSpace>,
                                         pfc::sim::stacks::SpectralCPUStack> {
public:
  using Base =
      pfc::ui::SpectralETDSession<HigherOrderPFCPhysics<double, pfc::HostSpace>,
                                  pfc::sim::stacks::SpectralCPUStack>;

  HigherOrderPFCSession(const nlohmann::json &settings, int rank, int nproc,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
class HigherOrderPFCHIPSession
    : public pfc::ui::SpectralETDSession<
          HigherOrderPFCPhysics<double, pfc::HIPSpace>,
          pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> {
public:
  using Base =
      pfc::ui::SpectralETDSession<HigherOrderPFCPhysics<double, pfc::HIPSpace>,
                                  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;

  HigherOrderPFCHIPSession(const nlohmann::json &settings, int rank, int nproc,
                           MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};
static_assert(
    pfc::sim::SpectralETDPhysics<HigherOrderPFCPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace higher_order_pfc
