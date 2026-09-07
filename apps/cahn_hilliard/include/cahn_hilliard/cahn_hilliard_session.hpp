// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cahn_hilliard_session.hpp
 * @brief Cahn–Hilliard = `CahnHilliardPhysics` on `SpectralETDSession`.
 *
 * Primary field is `c` (Cr mole fraction). Orszag 2/3-rule dealiasing is on
 * because the regular-solution remainder is not a low-order polynomial.
 * `register_catalog()` adds `cosine_mode` and `seeded_noise`. Optional
 * `diagnostics.csv` samples current-state mass, bounds, and total energy.
 * HIP builds
 * also define `CahnHilliardHIPSession` on `GPUSpectralStack<HIPSpace>`.
 */

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <cahn_hilliard/cahn_hilliard_physics.hpp>
#include <cahn_hilliard/cosine_mode.hpp>
#include <cahn_hilliard/seeded_noise.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/json_spectral_etd_session.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif

#include <cahn_hilliard/diagnostics.hpp>

namespace cahn_hilliard {

inline void register_catalog() {
  pfc::ui::register_field_modifier<CosineMode>("cosine_mode");
  pfc::ui::register_field_modifier<SeededNoise>("seeded_noise");
}

inline pfc::sim::SpectralETDOptions etd_options() {
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "c";
  opt.dealias = true;
  return opt;
}

template <class Space, class Stack>
class DiagnosticSession
    : public pfc::ui::SpectralETDSession<CahnHilliardPhysics<double, Space>, Stack> {
public:
  using Base =
      pfc::ui::SpectralETDSession<CahnHilliardPhysics<double, Space>, Stack>;

  DiagnosticSession(const nlohmann::json &settings, int rank, int nproc,
                    MPI_Comm comm = MPI_COMM_WORLD)
      : Base(settings, rank, nproc, comm, etd_options()), m_comm(comm) {}

  void run() {
    if (!this->settings().contains("diagnostics")) {
      Base::run();
      return;
    }
    const auto path =
        this->settings().at("diagnostics").at("csv").template get<std::string>();
    if (path.empty())
      throw std::invalid_argument("diagnostics.csv must not be empty");
    DiagnosticCSV csv(path, m_comm);
    Diagnostics<Space> diagnostics(this->domain(), this->fft(), m_comm);
    auto save = [&] {
      csv.write(pfc::time::increment(this->time()), pfc::time::current(this->time()),
                diagnostics.sample(this->psi(), this->system().physics().params));
    };
    // Restarts do not invoke the driver's initial-state callback.
    if (pfc::time::increment(this->time()) != 0 || pfc::time::done(this->time()))
      save();
    Base::run(save);
  }

private:
  MPI_Comm m_comm;
};

using CahnHilliardSession =
    DiagnosticSession<pfc::HostSpace, pfc::sim::stacks::SpectralCPUStack>;

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using CahnHilliardHIPSession =
    DiagnosticSession<pfc::HIPSpace,
                      pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;
static_assert(
    pfc::sim::SpectralETDPhysics<CahnHilliardPhysics<double, pfc::HIPSpace>>);
#endif

} // namespace cahn_hilliard
