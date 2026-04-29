// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_cpu_stack.hpp
 * @brief World, decomposition, HeFFTe CPU FFT, and Time from application JSON
 *
 * @details
 * Separates **spectral CPU infrastructure** construction from the concrete
 * `Model` type (`SpectralSimulationSession`). That split follows Single
 * Responsibility: this type knows how to build the FFT-backed grid stack from
 * settings; physics models remain independent and are layered on top.
 */

#ifndef PFC_UI_SPECTRAL_CPU_STACK_HPP
#define PFC_UI_SPECTRAL_CPU_STACK_HPP

#include <mpi.h>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::ui {

/**
 * @brief Non-copyable, non-movable holder for the CPU spectral pipeline from JSON
 *
 * Mirrors the construction order used by the UI session: world → decomposition
 * → HeFFTe CPU FFT → time. The MPI communicator is stored for downstream
 * `Simulator` wiring (same comm passed into FFT creation).
 */
class SpectralCpuStack {
public:
  SpectralCpuStack(const SpectralCpuStack &) = delete;
  SpectralCpuStack &operator=(const SpectralCpuStack &) = delete;
  SpectralCpuStack(SpectralCpuStack &&) = delete;
  SpectralCpuStack &operator=(SpectralCpuStack &&) = delete;

  /**
   * @param settings Parsed application JSON (world, time, plan_options, …)
   * @param comm MPI communicator for FFT creation and later simulator context
   * @param rank_id This rank index for FFT creation
   * @param num_ranks Number of ranks for decomposition
   */
  explicit SpectralCpuStack(const nlohmann::json &settings, MPI_Comm comm,
                            int rank_id, int num_ranks);

  [[nodiscard]] World &world() noexcept { return m_world; }
  [[nodiscard]] const World &world() const noexcept { return m_world; }

  [[nodiscard]] decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const decomposition::Decomposition &decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] fft::CpuFft &fft() noexcept { return m_fft; }
  [[nodiscard]] const fft::CpuFft &fft() const noexcept { return m_fft; }

  [[nodiscard]] Time &time() noexcept { return m_time; }
  [[nodiscard]] const Time &time() const noexcept { return m_time; }

  /** @brief Communicator passed at construction (FFT + simulator) */
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  World m_world;
  decomposition::Decomposition m_decomp;
  fft::CpuFft m_fft;
  Time m_time;
  MPI_Comm m_comm{MPI_COMM_WORLD};
};

inline SpectralCpuStack::SpectralCpuStack(const nlohmann::json &settings,
                                          MPI_Comm comm, int rank_id, int num_ranks)
    : m_world(ui::from_json<World>(settings)),
      m_decomp(decomposition::create(m_world, num_ranks)),
      m_fft(fft::create(
          fft::layout::create(m_decomp, 0), rank_id,
          settings.contains("plan_options")
              ? ui::from_json<heffte::plan_options>(settings["plan_options"])
              : heffte::default_options<heffte::backend::fftw>(),
          comm)),
      m_time(ui::from_json<Time>(settings)), m_comm(comm) {}

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_CPU_STACK_HPP
