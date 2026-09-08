// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gradient_elasticity_session.hpp
 * @brief JSON one-shot spectral solve for Helmholtz–Navier gradient elasticity.
 *
 * Not ETD: equilibrium \(\mathbf{u}\) is obtained by one FFT, a host-side
 * \(2\times 2\) invert per \(\mathbf{k}\), and two inverse FFTs. Dummy
 * `timestepping` is still required by the shared JSON Time parser.
 */

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <gradient_elasticity/cosine_mode.hpp>
#include <gradient_elasticity/gaussian_inclusion.hpp>
#include <gradient_elasticity/gradient_elasticity_physics.hpp>
#include <gradient_elasticity/gradient_elasticity_solve.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/frontend/ui/from_json_simulation_session.hpp>
#include <openpfc/frontend/ui/results_writer_catalog.hpp>
#include <openpfc/frontend/ui/simulation_wiring_conditions.hpp>
#include <openpfc/frontend/ui/simulation_wiring_context.hpp>
#include <openpfc/frontend/ui/simulation_wiring_writers.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/results_writer.hpp>
#include <openpfc/kernel/simulation/results_writer_domain.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>
#include <openpfc/kernel/simulation/simulation_session.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc/runtime/gpu/spectral_etd_ops_gpu.hpp>
#endif

namespace gradient_elasticity {

inline void register_catalog() {
  pfc::ui::register_field_modifier<CosineMode>("cosine_mode");
  pfc::ui::register_field_modifier<GaussianInclusion>("gaussian_inclusion");
}

template <class Stack> struct stack_memory_space {
  using type = typename Stack::field_type::memory_space;
};
template <> struct stack_memory_space<pfc::sim::stacks::SpectralCPUStack> {
  using type = pfc::HostSpace;
};
template <class Stack>
using stack_memory_space_t = typename stack_memory_space<Stack>::type;

template <class Stack = pfc::sim::stacks::SpectralCPUStack>
class GradientElasticitySession {
public:
  using memory_space = stack_memory_space_t<Stack>;
  using RealField = pfc::data::Field<double, memory_space>;
  using Physics = GradientElasticityPhysics<double, memory_space>;
  static constexpr bool is_host = std::is_same_v<memory_space, pfc::HostSpace>;

  GradientElasticitySession(const GradientElasticitySession &) = delete;
  GradientElasticitySession &operator=(const GradientElasticitySession &) = delete;
  GradientElasticitySession(GradientElasticitySession &&) = delete;
  GradientElasticitySession &operator=(GradientElasticitySession &&) = delete;

  GradientElasticitySession(const nlohmann::json &settings_in, int rank, int nproc,
                            MPI_Comm comm = MPI_COMM_WORLD)
      : m_settings(with_backend_default(settings_in)),
        m_ctx{.comm = comm, .mpi_rank = rank, .rank0 = (rank == 0)},
        m_domain(pfc::ui::from_json<pfc::Domain>(m_settings)),
        m_session(
            pfc::ui::make_simulation_session<Stack>(m_settings, rank, nproc, comm)) {
    const pfc::Box3i inbox = fft().get_inbox_bounds();
    const nlohmann::json params =
        (m_settings.contains("model") && m_settings["model"].contains("params"))
            ? m_settings["model"]["params"]
            : nlohmann::json::object();
    m_physics = Physics::from_json(params, m_domain, inbox);
    m_physics.declare_fields(m_state);

    auto &modifiers = pfc::ui::default_field_modifier_catalog();
    for (auto &ic :
         pfc::ui::parse_initial_conditions_from_json(m_settings, m_ctx, modifiers)) {
      apply_modifier(*ic, pfc::time::current(m_session.time()));
    }

    for (auto &nw : pfc::ui::parse_result_writers_from_json(
             m_settings, m_ctx, pfc::ui::default_results_writer_catalog())) {
      const std::string &name = nw.field_name;
      if (!m_state.has_field(name)) {
        throw std::invalid_argument("GradientElasticitySession: fields[] names '" +
                                    name +
                                    "' but the physics declares no such field");
      }
      pfc::apply_writer_domain(*nw.writer, real_field(name));
      m_writers.push_back(std::move(nw));
    }
  }

  struct FieldChecksum {
    double sum{};
    double sumsq{};
  };

  [[nodiscard]] FieldChecksum field_checksum(const std::string &name) {
    double sum = 0.0;
    double sumsq = 0.0;
    auto &f = real_field(name);
    const auto accumulate = [&](double v) {
      sum += v;
      sumsq += v * v;
    };
    if constexpr (is_host) {
      f.for_each_owned([&](int i, int j, int k) { accumulate(f(i, j, k)); });
    } else {
      f.with_host_view([&](double *d, std::size_t) {
        const auto sz = f.local_size();
        for (int k = 0; k < sz[2]; ++k) {
          for (int j = 0; j < sz[1]; ++j) {
            for (int i = 0; i < sz[0]; ++i) {
              accumulate(d[f.idx(i, j, k)]);
            }
          }
        }
      });
      f.note_device_write();
    }
    FieldChecksum global{};
    MPI_Allreduce(&sum, &global.sum, 1, MPI_DOUBLE, MPI_SUM, m_ctx.comm);
    MPI_Allreduce(&sumsq, &global.sumsq, 1, MPI_DOUBLE, MPI_SUM, m_ctx.comm);
    return global;
  }

  void run() {
    solve_displacement(fft(), m_physics, g(), ux(), uy());
    write_results();
    const FieldChecksum cs_x = field_checksum("ux");
    const FieldChecksum cs_y = field_checksum("uy");
    if (m_ctx.rank0) {
      std::cout << std::setprecision(17)
                << "SPECTRAL_CHECKSUM field=ux sum=" << cs_x.sum
                << " sumsq=" << cs_x.sumsq << " l2=" << std::sqrt(cs_x.sumsq)
                << '\n';
      std::cout << std::setprecision(17)
                << "SPECTRAL_CHECKSUM field=uy sum=" << cs_y.sum
                << " sumsq=" << cs_y.sumsq << " l2=" << std::sqrt(cs_y.sumsq)
                << '\n';
      std::cout << "SPECTRAL_CHECKSUM_HEX sum=" << std::hexfloat << cs_x.sum
                << std::defaultfloat << " sumsq=" << std::hexfloat << cs_x.sumsq
                << '\n';
    }
  }

  [[nodiscard]] RealField &g() { return real_field("g"); }
  [[nodiscard]] RealField &ux() { return real_field("ux"); }
  [[nodiscard]] RealField &uy() { return real_field("uy"); }
  [[nodiscard]] pfc::Time &time() noexcept { return m_session.time(); }
  [[nodiscard]] const pfc::Time &time() const noexcept { return m_session.time(); }
  [[nodiscard]] pfc::SimulationState &state() noexcept { return m_state; }
  [[nodiscard]] Physics &physics() noexcept { return m_physics; }
  [[nodiscard]] const Physics &physics() const noexcept { return m_physics; }
  [[nodiscard]] auto &fft() noexcept { return m_session.stack().fft(); }
  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }

private:
  static nlohmann::json with_backend_default(nlohmann::json settings) {
    if (!settings.contains("backend")) {
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
      if constexpr (std::is_same_v<memory_space, pfc::HIPSpace>) {
        settings["backend"] = "hip";
      }
#endif
    }
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

  RealField &target_field(const pfc::FieldModifier &m) {
    const std::string &name = m.get_field_name();
    if (name == "default" || name.empty()) {
      return real_field("g");
    }
    if (!m_state.has_field(name)) {
      throw std::invalid_argument("GradientElasticitySession: modifier '" +
                                  m.get_modifier_name() + "' targets field '" +
                                  name + "' which the physics does not declare");
    }
    return real_field(name);
  }

  void apply_modifier(pfc::FieldModifier &m, double t) {
    const pfc::SimulationContext context(m_ctx.comm);
    pfc::apply_field_modifier(m, target_field(m), t, &context);
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
        nw.writer->write(0, view);
      };
      if constexpr (is_host) {
        write_view(f.data(), f.size());
      } else {
        f.with_host_view([&](double *d, std::size_t n) { write_view(d, n); });
      }
    }
  }

  nlohmann::json m_settings;
  pfc::ui::JsonWiringContext m_ctx{};
  pfc::Domain m_domain{};
  pfc::sim::SimulationSession<Stack> m_session;
  pfc::SimulationState m_state;
  Physics m_physics{};
  std::vector<pfc::ui::NamedResultsWriter> m_writers;
};

using GradientElasticityCPUSession =
    GradientElasticitySession<pfc::sim::stacks::SpectralCPUStack>;

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using GradientElasticityHIPSession =
    GradientElasticitySession<pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>;
#endif

} // namespace gradient_elasticity
