// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file aluminum_etd_gpu_session.hpp
 * @brief JSON-driven GPU session: GPUSpectralStack + moving-frame ETD.
 *
 * ICs and fixed BCs run on the host mirror; FFT and ETD combine are device.
 */

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <aluminum/aluminum_etd_io.hpp>
#include <aluminum/aluminum_field_modifiers.hpp>
#include <aluminum/aluminum_physics.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc/runtime/gpu/moving_frame_mean_field_etd_gpu.hpp>

namespace aluminum {

template <class MemorySpace> class AluminumETDGPUSession {
public:
  using Physics = AluminumPhysics<double, MemorySpace>;
  using System = pfc::sim::DeviceMovingFrameMeanFieldETDSystem<Physics, MemorySpace>;

  AluminumETDGPUSession(const AluminumETDGPUSession &) = delete;
  AluminumETDGPUSession &operator=(const AluminumETDGPUSession &) = delete;
  AluminumETDGPUSession(AluminumETDGPUSession &&) = delete;
  AluminumETDGPUSession &operator=(AluminumETDGPUSession &&) = delete;

  AluminumETDGPUSession(const nlohmann::json &settings, int rank, int nproc,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : m_domain(pfc::ui::from_json<pfc::Domain>(settings)),
        m_time(pfc::ui::from_json<pfc::Time>(settings)),
        m_stack(m_domain, rank, nproc, comm) {
    Physics phys;
    phys.domain = m_domain;
    phys.box = m_stack.fft().get_inbox_bounds();
    if (settings.contains("model") && settings["model"].contains("params")) {
      apply_aluminum_json(settings["model"]["params"], phys.params);
    }
    phys.declare_fields(m_state);
    auto &psi = m_state.get_field<double, MemorySpace>("psi");
    psi.with_host_view([&](double *d, std::size_t n) {
      apply_ics_from_json(settings, psi.domain(), psi.box(), d, n);
    });
    m_bc = parse_fixed_bc(settings);
    m_writers.configure(settings, m_domain, m_stack.fft().get_inbox_bounds(), comm,
                        rank);
    pfc::sim::MovingFrameMeanFieldETDOptions opt{};
    opt.comm = comm;
    m_sys = std::make_unique<System>(std::move(phys), m_stack.fft(), m_state,
                                     pfc::time::dt(m_time), std::move(opt));
  }

  void run() {
    while (!pfc::time::done(m_time)) {
      if (pfc::time::increment(m_time) == 0) {
        apply_fixed_bc();
        write_psi();
      }
      pfc::time::next(m_time);
      apply_fixed_bc();
      m_sys->step(pfc::time::current(m_time));
      write_psi();
    }
  }

  [[nodiscard]] pfc::data::Field<double, MemorySpace> &psi() {
    return m_state.get_field<double, MemorySpace>("psi");
  }
  [[nodiscard]] const pfc::Time &time() const noexcept { return m_time; }
  [[nodiscard]] int dumps() const noexcept { return m_writers.dumps(); }
  [[nodiscard]] double last_free_energy() const { return m_sys->last_free_energy(); }

private:
  void apply_fixed_bc() {
    if (!m_bc) {
      return;
    }
    auto &psi = m_state.get_field<double, MemorySpace>("psi");
    psi.with_host_view([&](double *d, std::size_t) {
      aluminum::apply_fixed_bc(psi.domain(), psi.box(), d, *m_bc);
    });
  }

  void write_psi() {
    if (!m_writers.enabled()) {
      return;
    }
    auto &psi = m_state.get_field<double, MemorySpace>("psi");
    psi.with_host_view([&](double *d, std::size_t n) {
      m_writers.maybe_write(m_time, std::vector<double>(d, d + n));
    });
  }

  pfc::Domain m_domain{};
  pfc::Time m_time;
  pfc::sim::stacks::GPUSpectralStack<MemorySpace> m_stack;
  pfc::SimulationState m_state;
  std::optional<FixedBc> m_bc{};
  AluminumETDWriters m_writers{};
  std::unique_ptr<System> m_sys;
};

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using AluminumETDHIPSession = AluminumETDGPUSession<pfc::HIPSpace>;
#endif
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
using AluminumETDCUDASession = AluminumETDGPUSession<pfc::CUDASpace>;
#endif

} // namespace aluminum

#endif
