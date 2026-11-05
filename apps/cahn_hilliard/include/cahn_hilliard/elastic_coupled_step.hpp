// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file elastic_coupled_step.hpp
 * @brief One coherent equal-stiffness Cahn–Hilliard step, host and HIP.
 *
 * The HIP function calls `DeviceEigenstrainMicroelasticity::solve_resident`.
 * Strain and the elastic chemical potential stay on the device between
 * steps. The solver's own residual reduction is the only device-to-host
 * traffic inside that iteration. CUDA is not provided here. The periodic
 * representative volume and the equal-stiffness model are unchanged.
 */

#include <cmath>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include <cahn_hilliard/cahn_hilliard_physics.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/solvers/microelasticity/microelasticity.hpp>

namespace cahn_hilliard {

struct ElasticCoupledReport {
  double mass0{0.0};
  double mass{0.0};
  double elastic_energy{0.0};
  double max_abs_mu_el{0.0};
  int iterations{0};
  double residual{0.0};
  bool converged{false};
  std::vector<double> composition;
};

/// Owned-cell sum times the cell volume. `data` is inbox order, halo 0.
inline double composition_mass(const double *data, std::size_t n, double cell,
                               MPI_Comm comm) {
  double local = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    local += data[i];
  }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global * cell;
}

/// Cosine along x on the owned cells of @p grid. Mode 0 is the constant @p c0.
inline std::vector<double> cosine_composition(const pfc::data::Field<double> &grid,
                                              double c0, double amplitude,
                                              int mode_x) {
  std::vector<double> out(grid.size(), c0);
  const auto n = grid.global_size();
  const auto sp = grid.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double twopi = 2.0 * std::acos(-1.0);
  const auto ls = grid.local_size();
  for (int k = 0; k < ls[2]; ++k) {
    for (int j = 0; j < ls[1]; ++j) {
      for (int i = 0; i < ls[0]; ++i) {
        const auto x = grid.coords(i, j, k);
        const double phase = twopi * static_cast<double>(mode_x) * x[0] / Lx;
        out[grid.idx(i, j, k)] = c0 + amplitude * std::cos(phase);
      }
    }
  }
  return out;
}

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
ElasticCoupledReport advance_coherent_elastic_hip(
    const pfc::Domain &domain, const CahnHilliardParams &chemistry,
    const pfc::solvers::Stiffness &stiffness, double eps0, double c_ref,
    const std::vector<double> &initial, double dt, int steps, int rank, int nproc,
    MPI_Comm comm);
#endif

} // namespace cahn_hilliard
