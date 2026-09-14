// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_elasticity_hip.hpp
 * @brief Equations (5)-(7) on the device Green operator, for a GPU
 *        `DeviceStepper`. Same physics as `elasticity.hpp`.
 *
 * Assemble `h`/`a` from padded `phi`/`U`/`theta` on the device, solve with
 * rocFFT HeFFTe, write `d f_el/d phi` back into the padded driving-force
 * field. No tensor-field host round-trip in the timed path.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "device_elasticity_hip.hpp requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <stdexcept>
#include <string>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/gpu/fft_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc_apps/microelasticity.hpp>
#include <openpfc_apps/microelasticity_hip.hpp>
#include <openpfc_apps/microelasticity_hip_kernels.hpp>

#include <alloy_dendrite/device_step_hip.hpp>
#include <alloy_dendrite/elasticity.hpp>

namespace alloy_dendrite {

class DeviceElasticCoupling {
public:
  using DevField = pfc::data::Field<double, pfc::HIPSpace>;

  DeviceElasticCoupling(const DeviceElasticCoupling &) = delete;
  DeviceElasticCoupling &operator=(const DeviceElasticCoupling &) = delete;

  DeviceElasticCoupling(const pfc::Domain &domain,
                         const pfc::decomposition::Decomposition &decomp, int rank,
                         MPI_Comm comm, const ElasticParams &params,
                         const hip::DeviceGeom &geom)
      : m_params(params), m_geom(geom),
        m_fft(pfc::fft::create_hip(decomp, rank, comm, 0)),
        m_dfel(domain, pfc::decomposition::local_box(decomp, rank), geom.hw),
        m_solver(domain, m_fft, make_solver_params_(params, comm)) {
    if (params.n_el_substep < 1) {
      throw std::invalid_argument(
          "DeviceElasticCoupling: n_el_substep must be >= 1");
    }
    const auto inbox = m_fft.get_inbox_bounds();
    const auto fd = pfc::decomposition::local_box(decomp, rank);
    for (int d = 0; d < 3; ++d) {
      if (fd.low[d] != inbox.low[d] || fd.high[d] != inbox.high[d]) {
        throw std::runtime_error(
            "DeviceElasticCoupling: FD owned box and FFT inbox differ on axis " +
            std::to_string(d));
      }
    }
    if (static_cast<int>(m_solver.n_local()) != geom.nx * geom.ny * geom.nz) {
      throw std::runtime_error(
          "DeviceElasticCoupling: inbox cell count does not match DeviceGeom");
    }
    pfc::apps::hip_detail::me_fill(m_dfel.data(), 0.0,
                        static_cast<long long>(m_dfel.size()));
    m_dfel.note_device_write();
  }

  [[nodiscard]] const DevField &driving_force() const noexcept { return m_dfel; }
  [[nodiscard]] DevField &driving_force() noexcept { return m_dfel; }
  [[nodiscard]] pfc::apps::DeviceEigenstrainMicroelasticity &solver() noexcept {
    return m_solver;
  }

  [[nodiscard]] bool due(int step) const noexcept {
    return (step % m_params.n_el_substep) == 0;
  }

  ElasticReport solve(const DevField &phi, const DevField &U,
                      const DevField &theta) {
    pfc::apps::hip_detail::me_assemble_from_padded(
        phi.data(), U.data(), theta.data(), m_solver.h_device(),
        m_solver.amp_device(), m_solver.damp_device(), m_geom.nx, m_geom.ny,
        m_geom.nz, m_geom.hw, m_geom.sy, m_geom.sz, m_params.eps_c, m_params.eps_T,
        m_params.U_ref, m_params.theta_ref, m_solver.block_device(),
        m_solver.n_blocks());
    const double amp_sum = m_solver.finish_block_sum();
    if (m_params.macro_strain == MacroStrainMode::ZeroMeanStress) {
      const auto g = phi.global_size();
      const double ncells = static_cast<double>(g[0]) * g[1] * g[2];
      const double mean_amp = amp_sum / ncells;
      pfc::apps::Sym3 bar;
      for (int c = 0; c < pfc::apps::kSymComponents; ++c) {
        bar[c] = mean_amp * m_solver.params().eigenstrain_pattern[c];
      }
      m_solver.params().applied_strain = bar;
    }
    const auto rep = m_solver.solve_resident(true);
    pfc::apps::hip_detail::me_copy_owned_to_padded(
        m_solver.dfel_device(), m_dfel.data(), m_geom.nx, m_geom.ny, m_geom.nz,
        m_geom.hw, m_geom.sy, m_geom.sz);
    m_dfel.note_device_write();

    ElasticReport out;
    out.iterations = rep.iterations;
    out.residual = rep.residual;
    out.converged = rep.converged;
    out.total_energy = m_solver.total_elastic_energy();
    m_solver.stress_invariants(out.mean_stress_trace, out.max_dfel_dphi,
                               m_max_vm);
    return out;
  }

  [[nodiscard]] double max_von_mises() const noexcept { return m_max_vm; }

private:
  static pfc::apps::MicroelasticityParams
  make_solver_params_(const ElasticParams &p, MPI_Comm comm) {
    pfc::apps::MicroelasticityParams q;
    q.c_solid = p.c_solid;
    q.c_liquid =
        pfc::apps::soft_liquid(p.c_solid, p.mu_liquid_fraction, p.bulk_liquid_fraction);
    q.eigenstrain_pattern = pfc::apps::Sym3::identity();
    q.applied_strain = pfc::apps::Sym3{};
    q.scheme = p.scheme;
    q.tol_el = p.tol_el;
    q.n_el_iter = p.n_el_iter;
    q.warm_start = p.warm_start;
    q.comm = comm;
    return q;
  }

  ElasticParams m_params;
  hip::DeviceGeom m_geom{};
  pfc::fft::FFT_HIP m_fft;
  DevField m_dfel;
  pfc::apps::DeviceEigenstrainMicroelasticity m_solver;
  double m_max_vm{0.0};
};

} // namespace alloy_dendrite
