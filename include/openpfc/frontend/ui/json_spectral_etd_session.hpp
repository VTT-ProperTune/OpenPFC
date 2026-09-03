// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file json_spectral_etd_session.hpp
 * @brief JSON-driven session for any `SpectralETDPhysics` on any spectral stack.
 *
 * @details
 * `SpectralETDSession<Physics, Stack>` is the one production driver for
 * pseudo-spectral ETD models (tungsten, aluminum, toy models). From one JSON
 * document it assembles:
 *
 * - `Domain`, `Time`, and the spectral `Stack` (`SpectralCPUStack` or
 *   `GPUSpectralStack<MemorySpace>`) via `make_simulation_session`;
 * - the physics from `Physics::from_json(model_params, domain, inbox)` and its
 *   fields on an owning `SimulationState`;
 * - initial and boundary conditions from the `FieldModifier` catalog
 *   (`initial_conditions[]` / `boundary_conditions[]`, `default` target = the
 *   physics' primary field); apps register their own modifiers on the
 *   process-wide catalog before constructing the session;
 * - result writers from the `ResultsWriter` catalog (`fields[]`, `.vti`/`.vtk`
 *   paths default to the `vtk` writer), fed from the fields' own geometry;
 * - `CheckpointService` (`checkpoint.every` / `checkpoint.directory` /
 *   `restart_from`) and the optional `profiling` section;
 * - `SpectralETDSystem<Physics, MemorySpace>` driven by `SimulationDriver`.
 *
 * Device fields are handled by the framework: modifiers and writers run on the
 * tracked host mirror (`apply_field_modifier`, `with_host_view`), so no app
 * code touches residency.
 */

#include <complex>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/frontend/ui/json_checkpoint.hpp>
#include <openpfc/frontend/ui/json_step_profiler.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/frontend/ui/simulation_wiring_conditions.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/checkpoint_service.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>
#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/simulation_session.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>
#endif

namespace pfc::ui {

/// Memory space a spectral stack allocates its fields in.
template <class Stack> struct stack_memory_space {
  using type = typename Stack::field_type::memory_space;
};
template <> struct stack_memory_space<pfc::sim::stacks::SpectralCPUStack> {
  using type = pfc::HostSpace;
};
template <class Stack>
using stack_memory_space_t = typename stack_memory_space<Stack>::type;

/**
 * @brief Physics constructible from its JSON `model.params` plus geometry.
 */
template <class Physics>
concept JsonConstructiblePhysics =
    requires(const nlohmann::json &params, const pfc::Domain &domain,
             const pfc::Box3i &box) {
      { Physics::from_json(params, domain, box) } -> std::same_as<Physics>;
    };

/**
 * @brief JSON → spectral ETD run for one physics on one stack.
 *
 * @tparam Physics Models `pfc::sim::SpectralETDPhysics` and
 *                 @ref JsonConstructiblePhysics.
 * @tparam Stack   `pfc::sim::stacks::SpectralCPUStack` or
 *                 `pfc::sim::stacks::GPUSpectralStack<MemorySpace>`.
 */
template <class Physics, class Stack = pfc::sim::stacks::SpectralCPUStack>
  requires pfc::sim::SpectralETDPhysics<Physics> && JsonConstructiblePhysics<Physics>
class SpectralETDSession {
public:
  using memory_space = stack_memory_space_t<Stack>;
  using System = pfc::sim::SpectralETDSystem<Physics, memory_space>;
  using RealField = pfc::data::Field<double, memory_space>;
  static constexpr bool is_host = std::is_same_v<memory_space, pfc::HostSpace>;

  SpectralETDSession(const SpectralETDSession &) = delete;
  SpectralETDSession &operator=(const SpectralETDSession &) = delete;
  SpectralETDSession(SpectralETDSession &&) = delete;
  SpectralETDSession &operator=(SpectralETDSession &&) = delete;

  SpectralETDSession(const nlohmann::json &settings_in, int rank, int nproc,
                     MPI_Comm comm = MPI_COMM_WORLD,
                     pfc::sim::SpectralETDOptions system_options = {})
      : m_settings(with_backend_default(settings_in)),
        m_ctx{.comm = comm, .mpi_rank = rank, .rank0 = (rank == 0)},
        m_domain(from_json<pfc::Domain>(m_settings)),
        m_session(make_simulation_session<Stack>(m_settings, rank, nproc, comm)),
        m_ckpt(make_checkpoint_service(m_settings, comm)),
        m_profile(m_settings, rank, comm) {
    const pfc::Box3i inbox = fft().get_inbox_bounds();

    // Physics + fields.
    const nlohmann::json params =
        (m_settings.contains("model") && m_settings["model"].contains("params"))
            ? m_settings["model"]["params"]
            : nlohmann::json::object();
    Physics physics = Physics::from_json(params, m_domain, inbox);
    physics.declare_fields(m_state);
    if (!m_state.has_field(system_options.psi_name)) {
      throw std::invalid_argument("SpectralETDSession: physics did not declare "
                                  "the primary field '" +
                                  system_options.psi_name + "'");
    }
    m_psi_name = system_options.psi_name;

    // Conditions from the catalog; ICs applied once, BCs kept for the loop.
    auto &modifiers = default_field_modifier_catalog();
    for (auto &ic : parse_initial_conditions_from_json(m_settings, m_ctx, modifiers)) {
      apply_modifier(*ic, pfc::time::current(m_session.time()));
    }
    m_bcs = parse_boundary_conditions_from_json(m_settings, m_ctx, modifiers);

    // Writers, fed from field geometry.
    for (auto &nw : parse_result_writers_from_json(m_settings, m_ctx,
                                                   default_results_writer_catalog())) {
      const std::string &name = nw.field_name;
      if (!m_state.has_field(name)) {
        throw std::invalid_argument("SpectralETDSession: fields[] names '" + name +
                                    "' but the physics declares no such field");
      }
      pfc::apply_writer_domain(*nw.writer, real_field(name));
      m_writers.push_back(std::move(nw));
    }

    system_options.comm = comm;
    m_sys = std::make_unique<System>(std::move(physics), fft(), m_state,
                                     pfc::time::dt(m_session.time()),
                                     std::move(system_options));
    m_ckpt.template restore_from_config<memory_space>(m_state, m_session.time());
    m_result_counter = m_ckpt.result_counter();
  }

  /// Run to `t1`: BCs before every step, writers on `saveat`, checkpoints.
  void run() {
    pfc::sim::SimulationDriver driver(m_session.time(), &m_state);
    driver.run(
        [&](double t) {
          m_profile.timed_step(pfc::time::increment(m_session.time()), fft(),
                               [&] { m_sys->step(t); });
          m_ckpt.template maybe_save<memory_space>(m_state, m_session.time());
        },
        [&](pfc::Time &tm) { apply_bcs(tm); }, [&](pfc::Time &tm) { apply_bcs(tm); },
        [&](const pfc::Time &) { write_results(); });
    m_profile.finalize();
  }

  /// One physics step at the current time without the driver (benchmarks).
  void step_physics() { m_sys->step(pfc::time::current(m_session.time())); }

  // ---- accessors --------------------------------------------------------
  [[nodiscard]] RealField &psi() { return real_field(m_psi_name); }
  [[nodiscard]] const RealField &psi() const {
    return m_state.template get_field<double, memory_space>(m_psi_name);
  }
  [[nodiscard]] pfc::Time &time() noexcept { return m_session.time(); }
  [[nodiscard]] const pfc::Time &time() const noexcept { return m_session.time(); }
  [[nodiscard]] pfc::SimulationState &state() noexcept { return m_state; }
  [[nodiscard]] const pfc::SimulationState &state() const noexcept {
    return m_state;
  }
  [[nodiscard]] Stack &stack() noexcept { return m_session.stack(); }
  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }
  [[nodiscard]] auto &fft() noexcept { return m_session.stack().fft(); }
  [[nodiscard]] System &system() noexcept { return *m_sys; }
  [[nodiscard]] const System &system() const noexcept { return *m_sys; }
  [[nodiscard]] int dumps() const noexcept { return m_result_counter; }
  [[nodiscard]] bool writers_enabled() const noexcept { return !m_writers.empty(); }
  [[nodiscard]] const nlohmann::json &settings() const noexcept { return m_settings; }

  /// Communicator-wide free energy after the last step (0 if the physics has none).
  [[nodiscard]] double last_free_energy() const noexcept {
    return m_sys->last_free_energy();
  }
  [[nodiscard]] double last_free_energy_sum() const noexcept {
    return m_sys->last_free_energy_sum();
  }

private:
  static nlohmann::json with_backend_default(nlohmann::json settings) {
    // JSON may omit `backend`; a device binary then means its own backend.
    if (!settings.contains("backend")) {
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
      if constexpr (std::is_same_v<memory_space, pfc::CUDASpace>) {
        settings["backend"] = "cuda";
      }
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
      if constexpr (std::is_same_v<memory_space, pfc::HIPSpace>) {
        settings["backend"] = "hip";
      }
#endif
    }
    // `.vti` / `.vtk` output paths select the VTK writer unless told otherwise.
    if (settings.contains("fields") && settings["fields"].is_array()) {
      for (auto &field : settings["fields"]) {
        if (field.is_object() && !field.contains("writer") &&
            field.contains("data") && field["data"].is_string()) {
          const auto ext =
              std::filesystem::path(field["data"].get<std::string>()).extension();
          if (ext == ".vti" || ext == ".vtk") {
            field["writer"] = "vtk";
          }
        }
      }
    }
    return settings;
  }

  RealField &real_field(const std::string &name) {
    return m_state.template get_field<double, memory_space>(name);
  }

  /// Resolve a modifier's target: `default` means the primary field.
  RealField &target_field(const pfc::FieldModifier &m) {
    const std::string &name = m.get_field_name();
    if (name == "default" || name.empty()) {
      return real_field(m_psi_name);
    }
    if (!m_state.has_field(name)) {
      throw std::invalid_argument("SpectralETDSession: modifier '" +
                                  m.get_modifier_name() + "' targets field '" +
                                  name + "' which the physics does not declare");
    }
    return real_field(name);
  }

  void apply_modifier(pfc::FieldModifier &m, double t) {
    pfc::apply_field_modifier(m, target_field(m), t);
  }

  void apply_bcs(pfc::Time &tm) {
    const double t = pfc::time::current(tm);
    for (auto &bc : m_bcs) {
      apply_modifier(*bc, t);
    }
  }

  void write_results() {
    if (m_writers.empty()) {
      return;
    }
    for (auto &nw : m_writers) {
      auto &f = real_field(nw.field_name);
      const auto write_view = [&](const double *d, std::size_t n) {
        pfc::field::FieldView<double> view(d, n, f.box().size, f.spacing(),
                                           f.origin());
        nw.writer->write(m_result_counter, view);
      };
      if constexpr (is_host) {
        write_view(f.data(), f.size());
      } else {
        f.with_host_view([&](double *d, std::size_t n) { write_view(d, n); });
      }
    }
    ++m_result_counter;
    m_ckpt.set_result_counter(m_result_counter);
  }

  nlohmann::json m_settings;
  JsonWiringContext m_ctx{};
  pfc::Domain m_domain{};
  pfc::sim::SimulationSession<Stack> m_session;
  pfc::SimulationState m_state;
  pfc::sim::CheckpointService m_ckpt;
  JsonStepProfiler m_profile;
  std::string m_psi_name{"psi"};
  std::vector<std::unique_ptr<pfc::FieldModifier>> m_bcs;
  std::vector<NamedResultsWriter> m_writers;
  int m_result_counter{0};
  std::unique_ptr<System> m_sys;
};

} // namespace pfc::ui
