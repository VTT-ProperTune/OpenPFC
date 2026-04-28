// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_simulation_session.hpp
 * @brief Owns the spectral simulation object graph built from JSON settings
 *
 * @details
 * `SpectralSimulationSession` groups World, decomposition, CPU FFT, Time,
 * concrete Model, and Simulator construction in one place so `App::main()` can
 * focus on profiling, I/O registration, and the time loop. The session is
 * returned as `std::unique_ptr` so it is never moved after construction: the
 * Simulator holds references to the model and time members. `CpuFft` is
 * constructed in the member initializer list (it is not movable).
 */

#ifndef PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP
#define PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP

#include <memory>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/simulation_wiring.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::ui {

/**
 * @brief Heap-owned spectral (HeFFTe CPU FFT) simulation stack from settings
 *
 * @tparam ConcreteModel Physics model type (e.g. from an application target)
 */
template <class ConcreteModel> class SpectralSimulationSession {
public:
  SpectralSimulationSession(const SpectralSimulationSession &) = delete;
  SpectralSimulationSession &operator=(const SpectralSimulationSession &) = delete;
  SpectralSimulationSession(SpectralSimulationSession &&) = delete;
  SpectralSimulationSession &operator=(SpectralSimulationSession &&) = delete;

  /**
   * @brief Build world, decomposition, FFT, time, model, and simulator
   *
   * @param settings Parsed application JSON (world, time, plan_options, …)
   * @param comm MPI communicator for FFT and simulator modifier context
   * @param rank_id This rank index for FFT creation
   * @param num_ranks Number of ranks for decomposition
   */
  explicit SpectralSimulationSession(const nlohmann::json &settings, MPI_Comm comm,
                                     int rank_id, int num_ranks)
      : m_world(ui::from_json<World>(settings)),
        m_decomp(decomposition::create(m_world, num_ranks)),
        m_fft(fft::create(
            fft::layout::create(m_decomp, 0), rank_id,
            settings.contains("plan_options")
                ? ui::from_json<heffte::plan_options>(settings["plan_options"])
                : heffte::default_options<heffte::backend::fftw>(),
            comm)),
        m_time(ui::from_json<Time>(settings)), m_model(m_fft, m_world),
        m_simulator(m_model, m_time, comm) {}

  [[nodiscard]] static std::unique_ptr<SpectralSimulationSession>
  assemble(const nlohmann::json &settings, MPI_Comm comm, int rank_id,
           int num_ranks) {
    return std::make_unique<SpectralSimulationSession>(settings, comm, rank_id,
                                                       num_ranks);
  }

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

  [[nodiscard]] ConcreteModel &model() noexcept { return m_model; }
  [[nodiscard]] const ConcreteModel &model() const noexcept { return m_model; }

  [[nodiscard]] Simulator &simulator() noexcept { return m_simulator; }
  [[nodiscard]] const Simulator &simulator() const noexcept { return m_simulator; }

  /**
   * @brief Register results writers and field modifiers from application JSON
   */
  void wire_simulator_from_settings(const nlohmann::json &settings, MPI_Comm comm,
                                    int mpi_rank, bool rank0) {
    add_result_writers_from_json(m_simulator, settings, comm, mpi_rank, rank0);
    add_initial_conditions_from_json(m_simulator, settings, comm, mpi_rank, rank0);
    add_boundary_conditions_from_json(m_simulator, settings, comm, mpi_rank, rank0);
  }

private:
  World m_world;
  decomposition::Decomposition m_decomp;
  fft::CpuFft m_fft;
  Time m_time;
  ConcreteModel m_model;
  Simulator m_simulator;
};

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP
