// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gradient_elasticity_diagnostics.hpp
 * @brief Real-space stress/energy post-processing and reduced diagnostics
 * for the gradient-elasticity size-effect study (`#117`).
 *
 * @details
 * `compute_stress_fields` turns the compatible strain (`exx`/`eyy`/`exy`,
 * from `solve_displacement_and_strain`) and the eigenstrain field `g` into
 * the Cauchy stress components, hydrostatic/von-Mises invariants, and the
 * elastic energy density, via `GradientElasticityPhysics::stress_state` at
 * every owned grid point. `summarize_stress` then reduces those fields to
 * the three scalar science-case observables: peak \(|\sigma_h|\), peak
 * \(\sigma_{vm}\), and total elastic energy \(\int w\,dA\) (MPI-collective).
 *
 * `write_line_profile` dumps a straight line cut through a chosen centre
 * (displacement, stress, energy density vs. distance) to CSV -- the "radial
 * line profile" required by the issue. A straight cut through the inclusion
 * centre is used instead of a polar/angular average: the circular inclusion
 * is not exactly axisymmetric on a periodic square grid, but a cut is
 * simple, exact (no interpolation), and sufficient to see the near-field
 * decay and any high-\(k\) ringing at the interface.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <gradient_elasticity/gradient_elasticity_physics.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

namespace gradient_elasticity {

/// Reduced size-effect observables (`#117` science case A): peak stresses
/// and total stored elastic energy over the whole periodic cell.
struct StressSummary {
  double peak_abs_hydrostatic{}; ///< \(\max|\sigma_h|\) over the domain.
  double peak_von_mises{};       ///< \(\max\sigma_{vm}\) over the domain.
  double total_elastic_energy{}; ///< \(\int w\,dA\) (2-D; per unit thickness).
};

/// Fill `sxx`/`syy`/`sxy`/`stress_hydro`/`stress_vm`/`energy_density` from
/// the strain fields `exx`/`eyy`/`exy` and the eigenstrain field `g`, using
/// `phys.stress_state` pointwise. Works uniformly for a host or a
/// device-backed field via `with_host_view` (identity for `HostSpace`).
template <class Physics, class MemorySpace>
void compute_stress_fields(const Physics &phys,
                           pfc::data::Field<double, MemorySpace> &g,
                           pfc::data::Field<double, MemorySpace> &exx,
                           pfc::data::Field<double, MemorySpace> &eyy,
                           pfc::data::Field<double, MemorySpace> &exy,
                           pfc::data::Field<double, MemorySpace> &sxx,
                           pfc::data::Field<double, MemorySpace> &syy,
                           pfc::data::Field<double, MemorySpace> &sxy,
                           pfc::data::Field<double, MemorySpace> &stress_hydro,
                           pfc::data::Field<double, MemorySpace> &stress_vm,
                           pfc::data::Field<double, MemorySpace> &energy_density) {
  g.with_host_view([&](double *gp, std::size_t n) {
    exx.with_host_view([&](double *exxp, std::size_t) {
      eyy.with_host_view([&](double *eyyp, std::size_t) {
        exy.with_host_view([&](double *exyp, std::size_t) {
          sxx.with_host_view([&](double *sxxp, std::size_t) {
            syy.with_host_view([&](double *syyp, std::size_t) {
              sxy.with_host_view([&](double *sxyp, std::size_t) {
                stress_hydro.with_host_view([&](double *pp, std::size_t) {
                  stress_vm.with_host_view([&](double *vmp, std::size_t) {
                    energy_density.with_host_view([&](double *wp, std::size_t) {
                      for (std::size_t idx = 0; idx < n; ++idx) {
                        const auto s =
                            phys.stress_state(exxp[idx], eyyp[idx], exyp[idx], gp[idx]);
                        sxxp[idx] = s.sxx;
                        syyp[idx] = s.syy;
                        sxyp[idx] = s.sxy;
                        pp[idx] = s.hydrostatic;
                        vmp[idx] = s.von_mises;
                        wp[idx] = s.energy_density;
                      }
                    });
                  });
                });
              });
            });
          });
        });
      });
    });
  });
}

/// MPI-collective reduction of the derived stress/energy fields to the
/// scalar size-effect observables (peak stresses, total energy).
template <class MemorySpace>
StressSummary summarize_stress(pfc::data::Field<double, MemorySpace> &stress_hydro,
                               pfc::data::Field<double, MemorySpace> &stress_vm,
                               pfc::data::Field<double, MemorySpace> &energy_density,
                               const pfc::Domain &domain, MPI_Comm comm) {
  double local_hydro = 0.0;
  double local_vm = 0.0;
  double local_energy = 0.0;
  stress_hydro.with_host_view([&](double *p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      local_hydro = std::max(local_hydro, std::abs(p[i]));
    }
  });
  stress_vm.with_host_view([&](double *p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      local_vm = std::max(local_vm, p[i]);
    }
  });
  energy_density.with_host_view([&](double *p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      local_energy += p[i];
    }
  });
  const auto dx = pfc::domain::get_spacing(domain);
  const double cell_volume = dx[0] * dx[1] * dx[2];
  StressSummary global{};
  MPI_Allreduce(&local_hydro, &global.peak_abs_hydrostatic, 1, MPI_DOUBLE, MPI_MAX,
               comm);
  MPI_Allreduce(&local_vm, &global.peak_von_mises, 1, MPI_DOUBLE, MPI_MAX, comm);
  double global_energy_sum = 0.0;
  MPI_Allreduce(&local_energy, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  global.total_elastic_energy = global_energy_sum * cell_volume;
  return global;
}

/**
 * @brief Write a straight-line CSV cut through `(x0, y0)` along `+x`
 * (`r, g, ux, uy, stress_hydro, stress_vm, energy_density`).
 *
 * Single-rank only (the science sweeps that use this run on one rank); throws
 * if `nproc > 1` so a multi-rank misuse fails loudly instead of writing a
 * partial line.
 */
template <class MemorySpace>
void write_line_profile(const std::filesystem::path &path,
                        pfc::data::Field<double, MemorySpace> &g,
                        pfc::data::Field<double, MemorySpace> &ux,
                        pfc::data::Field<double, MemorySpace> &uy,
                        pfc::data::Field<double, MemorySpace> &stress_hydro,
                        pfc::data::Field<double, MemorySpace> &stress_vm,
                        pfc::data::Field<double, MemorySpace> &energy_density,
                        double x0, double y0, int nproc) {
  if (nproc != 1) {
    throw std::runtime_error(
        "gradient_elasticity: write_line_profile requires a single rank");
  }
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("gradient_elasticity: cannot open line profile CSV: " +
                             path.string());
  }
  out << std::setprecision(17);
  out << "r,x,y,g,ux,uy,stress_hydro,stress_vm,energy_density\n";
  const auto size = g.local_size();
  const int j0 = std::clamp(
      static_cast<int>(std::lround((y0 - g.origin()[1]) / g.spacing()[1])), 0,
      size[1] - 1);
  g.with_host_view([&](double *gp, std::size_t) {
    ux.with_host_view([&](double *uxp, std::size_t) {
      uy.with_host_view([&](double *uyp, std::size_t) {
        stress_hydro.with_host_view([&](double *pp, std::size_t) {
          stress_vm.with_host_view([&](double *vmp, std::size_t) {
            energy_density.with_host_view([&](double *wp, std::size_t) {
              for (int i = 0; i < size[0]; ++i) {
                const auto c = g.coords(i, j0, 0);
                const std::size_t idx = g.idx(i, j0, 0);
                out << (c[0] - x0) << ',' << c[0] << ',' << c[1] << ',' << gp[idx]
                    << ',' << uxp[idx] << ',' << uyp[idx] << ',' << pp[idx] << ','
                    << vmp[idx] << ',' << wp[idx] << '\n';
              }
            });
          });
        });
      });
    });
  });
}

} // namespace gradient_elasticity
