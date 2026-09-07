// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file thin_film_session.hpp
 * @brief Thin film = `ThinFilmPhysics` on `SpectralETDSession`.
 *
 * Primary field is `h`. Orszag 2/3-rule dealiasing is on because \(\Pi(h)\)
 * is not a low-order polynomial.
 */

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <thin_film/cosine_mode.hpp>
#include <thin_film/thin_film_physics.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

namespace thin_film {

inline void register_catalog() {
  pfc::ui::register_field_modifier<CosineMode>("cosine_mode");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  opt.dealias = true;
  return opt;
}

class ThinFilmSession
    : public pfc::ui::SpectralETDSession<ThinFilmPhysics<double, pfc::HostSpace>,
                                         pfc::sim::stacks::SpectralCPUStack> {
public:
  using Base = pfc::ui::SpectralETDSession<ThinFilmPhysics<double, pfc::HostSpace>,
                                           pfc::sim::stacks::SpectralCPUStack>;

  ThinFilmSession(const nlohmann::json &settings, int rank, int nproc,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
class ThinFilmHIPSession : public pfc::ui::SpectralETDSession<
                               ThinFilmPhysics<double, pfc::HIPSpace>,
                               pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>> {
public:
  using Base =
      pfc::ui::SpectralETDSession<ThinFilmPhysics<double, pfc::HIPSpace>,
                                  pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;

  ThinFilmHIPSession(const nlohmann::json &settings, int rank, int nproc,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()) {}
};
static_assert(pfc::sim::SpectralETDPhysics<ThinFilmPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace thin_film
