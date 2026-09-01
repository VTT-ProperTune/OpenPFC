// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_simulation_session.hpp
 * @brief Gen-1 App owner: `SimulationSession<SpectralCPUStack>` + Model + Simulator
 *
 * @details
 * JSON `App` still drives a virtual `Model` / `Simulator` until M12. This
 * session is the heap-owned bundle those types need: a
 * `pfc::sim::SimulationSession<pfc::sim::stacks::SpectralCPUStack>` (Domain,
 * decomposition, CPUFFT, Time, HeFFTe plan options) plus `ConcreteModel` and
 * `Simulator`. There is no frontend `SpectralCPUStack` twin.
 *
 * `World` is stored by value so `Model`'s `const World&` does not dangle
 * off `stack().world()` (which returns a temporary).
 *
 * Returned as `std::unique_ptr` so the object is never moved after
 * construction: the simulator holds references to the model and time.
 *
 * Non-member accessors (`pfc::ui::world(session)`, `pfc::ui::time(session)`, …)
 * mirror the member API for consistency with `pfc::get_model(sim)` on
 * `Simulator`.
 */

#ifndef PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP
#define PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP

#include <memory>

#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/frontend/ui/simulation_wiring.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/simulation_session.hpp>
#include <openpfc/kernel/simulation/simulator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

namespace pfc::ui {

/**
 * @brief Heap-owned spectral CPU simulation graph from JSON (Gen-1 Model path)
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
   * @brief Build `SimulationSession<SpectralCPUStack>`, model, and simulator
   *
   * @param settings Parsed application JSON (world, time, plan_options, …)
   * @param comm MPI communicator for FFT and simulator modifier context
   * @param rank_id This rank index for FFT creation
   * @param num_ranks Number of ranks for decomposition
   */
  explicit SpectralSimulationSession(const nlohmann::json &settings, MPI_Comm comm,
                                     int rank_id, int num_ranks)
      : m_session(make_simulation_session<pfc::sim::stacks::SpectralCPUStack>(
            settings, rank_id, num_ranks, comm)),
        m_world(m_session.stack().world()),
        m_model(m_session.stack().fft(), m_world, comm),
        m_simulator(m_model, m_session.time(), comm) {}

  [[nodiscard]] static std::unique_ptr<SpectralSimulationSession>
  assemble(const nlohmann::json &settings, MPI_Comm comm, int rank_id,
           int num_ranks) {
    return std::make_unique<SpectralSimulationSession>(settings, comm, rank_id,
                                                       num_ranks);
  }

  [[nodiscard]] pfc::sim::SimulationSession<pfc::sim::stacks::SpectralCPUStack> &
  session() noexcept {
    return m_session;
  }
  [[nodiscard]] const pfc::sim::SimulationSession<
      pfc::sim::stacks::SpectralCPUStack> &
  session() const noexcept {
    return m_session;
  }

  [[nodiscard]] World &world() noexcept { return m_world; }
  [[nodiscard]] const World &world() const noexcept { return m_world; }

  [[nodiscard]] decomposition::Decomposition &decomposition() noexcept {
    return m_session.stack().decomposition();
  }
  [[nodiscard]] const decomposition::Decomposition &decomposition() const noexcept {
    return m_session.stack().decomposition();
  }

  [[nodiscard]] fft::CPUFFT &fft() noexcept { return m_session.stack().fft(); }
  [[nodiscard]] const fft::CPUFFT &fft() const noexcept {
    return m_session.stack().fft();
  }

  [[nodiscard]] Time &time() noexcept { return m_session.time(); }
  [[nodiscard]] const Time &time() const noexcept { return m_session.time(); }

  [[nodiscard]] ConcreteModel &model() noexcept { return m_model; }
  [[nodiscard]] const ConcreteModel &model() const noexcept { return m_model; }

  [[nodiscard]] Simulator &simulator() noexcept { return m_simulator; }
  [[nodiscard]] const Simulator &simulator() const noexcept { return m_simulator; }

  /**
   * @brief Register writers/modifiers and apply optional `"simulator"` JSON keys
   *
   * Uses the same MPI communicator as FFT construction (`SpectralCPUStack`).
   *
   * @param modifier_catalog Factories for JSON `type` strings.
   * @param writer_catalog Factories for JSON `fields[].writer` (e.g.
   *        `default_results_writer_catalog()` for built-in writers).
   */
  void wire_simulator_from_settings(const nlohmann::json &settings, int mpi_rank,
                                    bool rank0,
                                    const FieldModifierCatalog &modifier_catalog,
                                    const ResultsWriterCatalog &writer_catalog) {
    wire_simulator_and_runtime_from_json(
        m_simulator, time(), settings,
        JsonWiringContext{m_session.stack().mpi_comm(), mpi_rank, rank0},
        modifier_catalog, writer_catalog);
  }

  /**
   * @brief Same as the overload taking `(mpi_rank, rank0, catalogs)`; uses the
   *        stack communicator with rank flags from `session.ctx`
   */
  void wire_simulator_from_settings(const nlohmann::json &settings,
                                    const JsonWiringSession &wiring) {
    wire_simulator_and_runtime_from_json(
        m_simulator, time(), settings,
        JsonWiringSession{JsonWiringContext{m_session.stack().mpi_comm(),
                                            wiring.ctx.mpi_rank, wiring.ctx.rank0},
                          wiring.modifier_catalog, wiring.writer_catalog});
  }

private:
  pfc::sim::SimulationSession<pfc::sim::stacks::SpectralCPUStack> m_session;
  World m_world;
  ConcreteModel m_model;
  Simulator m_simulator;
};

template <class M>
[[nodiscard]] inline World &world(SpectralSimulationSession<M> &session) noexcept {
  return session.world();
}
template <class M>
[[nodiscard]] inline const World &
world(const SpectralSimulationSession<M> &session) noexcept {
  return session.world();
}

template <class M>
[[nodiscard]] inline decomposition::Decomposition &
decomposition(SpectralSimulationSession<M> &session) noexcept {
  return session.decomposition();
}
template <class M>
[[nodiscard]] inline const decomposition::Decomposition &
decomposition(const SpectralSimulationSession<M> &session) noexcept {
  return session.decomposition();
}

template <class M>
[[nodiscard]] inline fft::CPUFFT &
fft(SpectralSimulationSession<M> &session) noexcept {
  return session.fft();
}
template <class M>
[[nodiscard]] inline const fft::CPUFFT &
fft(const SpectralSimulationSession<M> &session) noexcept {
  return session.fft();
}

template <class M>
[[nodiscard]] inline Time &time(SpectralSimulationSession<M> &session) noexcept {
  return session.time();
}
template <class M>
[[nodiscard]] inline const Time &
time(const SpectralSimulationSession<M> &session) noexcept {
  return session.time();
}

template <class M>
[[nodiscard]] inline M &model(SpectralSimulationSession<M> &session) noexcept {
  return session.model();
}
template <class M>
[[nodiscard]] inline const M &
model(const SpectralSimulationSession<M> &session) noexcept {
  return session.model();
}

template <class M>
[[nodiscard]] inline Simulator &
simulator(SpectralSimulationSession<M> &session) noexcept {
  return session.simulator();
}
template <class M>
[[nodiscard]] inline const Simulator &
simulator(const SpectralSimulationSession<M> &session) noexcept {
  return session.simulator();
}

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_SIMULATION_SESSION_HPP
