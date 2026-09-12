// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "openpfc_inverse_homogenize_hip requires HIP spectral (rocFFT HeFFTe)"
#endif

/**
 * HIP inverse: device Green/FFT elasticity, host Allen–Cahn on the
 * downloaded C_H sensitivity. The inner loop (six unit-cell solves) is GPU.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc_apps/homogenization_hip.hpp>
#include <inverse_homogenization/auxetic_geometry.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int nx = 16, ny = 16, nz = 16, steps = 4;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    auto eq = a.find('=');
    if (eq == std::string::npos) continue;
    auto k = a.substr(2, eq - 2), v = a.substr(eq + 1);
    if (k == "nx") nx = std::stoi(v);
    else if (k == "ny") ny = std::stoi(v);
    else if (k == "nz") nz = std::stoi(v);
    else if (k == "steps") steps = std::stoi(v);
  }
  int rc = 0;
  {
    const pfc::Domain domain = pfc::domain::create(
        pfc::GridSize({nx, ny, nz}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
        pfc::GridSpacing({1.0, 1.0, 1.0}));
    pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace> stack(domain, rank, nproc,
                                                            MPI_COMM_WORLD);
    auto h = pfc::data::field_from_inbox<double>(domain,
                                                 stack.fft().get_inbox_bounds());
    std::fill(h.vec().begin(), h.vec().end(), 0.55);
    h.note_host_write();
    pfc::apps::MicroelasticityParams p;
    p.c_solid = pfc::apps::Stiffness::isotropic(1.0, 0.3);
    p.c_liquid = pfc::apps::Stiffness::isotropic(0.25, 0.3);
    p.tol_el = 1.0e-8;
    p.n_el_iter = 80;
    p.comm = MPI_COMM_WORLD;
    pfc::apps::PeriodicHomogenizerHIP hom(domain, stack.fft(), p);
    const auto Cstar =
        pfc::apps::voigt_from_stiffness(pfc::apps::Stiffness::isotropic(0.6, 0.3));
    const auto W = pfc::apps::Voigt6::ones();
    auto dJ = pfc::data::field_from_inbox<double>(domain,
                                                  stack.fft().get_inbox_bounds());
    if (rank == 0)
      std::cout << "backend hip ranks " << nproc << " grid " << nx << 'x' << ny
                << 'x' << nz << " steps " << steps << '\n';
    pfc::apps::HomogenizationResult last{};
    for (int s = 0; s < steps; ++s) {
      last = hom.compute(h);
      hom.objective_sensitivity(h, Cstar, W, dJ);
      double g2 = 0.0;
      for (std::size_t i = 0; i < h.size(); ++i) g2 += dJ.data()[i] * dJ.data()[i];
      double g2g = 0.0;
      MPI_Allreduce(&g2, &g2g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double rms = std::sqrt(g2g / std::max(1.0, static_cast<double>(nx) * ny * nz));
      const double scale = (rms > 1e-30) ? 0.05 / rms : 0.0;
      for (std::size_t i = 0; i < h.size(); ++i) {
        double hn = h.data()[i] - scale * dJ.data()[i];
        h.data()[i] = std::min(1.0, std::max(0.0, hn));
      }
      h.note_host_write();
      if (rank == 0)
        std::cout << s << " J " << last.stiffness.frobenius_norm() << " vf "
                  << last.volume_fraction << " conv "
                  << (last.all_converged() ? "yes" : "no") << '\n';
      if (!last.all_converged()) rc = 1;
    }
    if (rank == 0)
      std::cout << std::setprecision(16) << "INVERSE_CHECKSUM "
                << last.stiffness.frobenius_norm() << '\n';
  }
  MPI_Finalize();
  return rc;
}
