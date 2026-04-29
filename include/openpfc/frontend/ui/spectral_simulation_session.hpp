// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_simulation_session.hpp
 * @brief Owns the spectral simulation object graph built from JSON settings
 *
 * @details
 * `SpectralSimulationSession` composes `SpectralCpuStack` (world, decomposition,
 * CPU FFT, time) with a concrete `Model` and `Simulator`. The session is
 * returned as `std::unique_ptr` so it is never moved after construction: the
 * simulator holds references to the model and time members. `CpuFft` is held
 * inside the stack and is not movable.
 */

#ifndef PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP
#define PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP

#include <memory>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/simulation_wiring.hpp>
#include <openpfc/frontend/ui/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>

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
      : m_stack(settings, comm, rank_id, num_ranks),
        m_model(m_stack.fft(), m_stack.world()),
        m_simulator(m_model, m_stack.time(), comm) {}

  [[nodiscard]] static std::unique_ptr<SpectralSimulationSession>
  assemble(const nlohmann::json &settings, MPI_Comm comm, int rank_id,
           int num_ranks) {
    return std::make_unique<SpectralSimulationSession>(settings, comm, rank_id,
                                                       num_ranks);
  }

  [[nodiscard]] World &world() noexcept { return m_stack.world(); }
  [[nodiscard]] const World &world() const noexcept { return m_stack.world(); }

  [[nodiscard]] decomposition::Decomposition &decomposition() noexcept {
    return m_stack.decomposition();
  }
  [[nodiscard]] const decomposition::Decomposition &decomposition() const noexcept {
    return m_stack.decomposition();
  }

  [[nodiscard]] fft::CpuFft &fft() noexcept { return m_stack.fft(); }
  [[nodiscard]] const fft::CpuFft &fft() const noexcept { return m_stack.fft(); }

  [[nodiscard]] Time &time() noexcept { return m_stack.time(); }
  [[nodiscard]] const Time &time() const noexcept { return m_stack.time(); }

  [[nodiscard]] ConcreteModel &model() noexcept { return m_model; }
  [[nodiscard]] const ConcreteModel &model() const noexcept { return m_model; }

  [[nodiscard]] Simulator &simulator() noexcept { return m_simulator; }
  [[nodiscard]] const Simulator &simulator() const noexcept { return m_simulator; }

  /**
   * @brief Register writers/modifiers and apply optional `"simulator"` JSON keys
   *
   * Uses the same MPI communicator as FFT construction (`SpectralCpuStack`).
   *
   * @param modifier_catalog Factories for JSON `type` strings (defaults to the
   *        process-wide catalog; inject a test catalog for unit tests).
   */
  void wire_simulator_from_settings(const nlohmann::json &settings, int mpi_rank,
                                    bool rank0,
                                    const FieldModifierCatalog &modifier_catalog =
                                        default_field_modifier_catalog()) {
    wire_simulator_and_runtime_from_json(m_simulator, m_stack.time(), settings,
                                         m_stack.mpi_comm(), mpi_rank, rank0,
                                         modifier_catalog);
  }

private:
  SpectralCpuStack m_stack;
  ConcreteModel m_model;
  Simulator m_simulator;
};

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP
