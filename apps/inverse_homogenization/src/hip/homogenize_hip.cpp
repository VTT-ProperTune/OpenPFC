// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "openpfc_homogenize_hip requires HIP spectral (rocFFT HeFFTe)"
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <openpfc_apps/homogenization_hip.hpp>
#include <inverse_homogenization/auxetic_geometry.hpp>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int rc = 0;
  {
    int nx = 16, ny = 16, nz = 16;
    double E_s = 1.0, nu_s = 0.3, E_v = 0.25, nu_v = 0.3, vol = 1.0;
    std::string shape = "homogeneous";
    for (int i = 1; i < argc; ++i) {
      const std::string_view tok(argv[i]);
      const auto eq = tok.find('=');
      if (!tok.starts_with("--") || eq == std::string_view::npos) continue;
      const auto key = tok.substr(2, eq - 2);
      const auto val = std::string(tok.substr(eq + 1));
      if (key == "nx") nx = std::stoi(val);
      else if (key == "ny") ny = std::stoi(val);
      else if (key == "nz") nz = std::stoi(val);
      else if (key == "E-solid") E_s = std::stod(val);
      else if (key == "nu-solid") nu_s = std::stod(val);
      else if (key == "E-void") E_v = std::stod(val);
      else if (key == "nu-void") nu_v = std::stod(val);
      else if (key == "volume") vol = std::stod(val);
      else if (key == "shape") shape = val;
    }
    const pfc::Domain domain = pfc::domain::create(
        pfc::GridSize({nx, ny, nz}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
        pfc::GridSpacing({1.0, 1.0, 1.0}));
    pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace> stack(domain, rank, nproc,
                                                            MPI_COMM_WORLD);
    auto h = pfc::data::field_from_inbox<double>(domain, stack.fft().get_inbox_bounds());
    const auto n = h.local_size();
    const double Lz = nz * 1.0;
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const auto x = h.coords(i, j, k);
          double hv = vol;
          if (shape == "laminate-z") hv = (x[2] < 0.5 * Lz) ? 1.0 : 0.0;
          else if (shape == "rotating-squares") hv = 0.0;
          h(i, j, k) = hv;
        }
    if (shape == "rotating-squares")
      pfc::apps::inverse::fill_rotating_squares(h, nx, ny, 0.200, 0.45);
    h.note_host_write();

    pfc::apps::MicroelasticityParams p;
    p.c_solid = pfc::apps::Stiffness::isotropic(E_s, nu_s);
    p.c_liquid = pfc::apps::Stiffness::isotropic(E_v, nu_v);
    p.tol_el = 1.0e-8;
    p.n_el_iter = 80;
    p.comm = MPI_COMM_WORLD;
    pfc::apps::PeriodicHomogenizerHIP hom(domain, stack.fft(), p);
    const auto r = hom.compute(h);
    if (rank == 0) {
      std::cout << "backend hip ranks " << nproc << " grid " << nx << 'x' << ny
                << 'x' << nz << " shape " << shape << '\n';
      std::cout << std::setprecision(10) << "volume_fraction " << r.volume_fraction
                << " converged " << (r.all_converged() ? "yes" : "no") << '\n';
      std::cout << "C_H (engineering Voigt)\n";
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          if (j) std::cout << ' ';
          std::cout << std::setw(16) << r.stiffness(i, j);
        }
        std::cout << '\n';
      }
      std::cout << std::setprecision(16)
                << "HOMOGENIZATION_CHECKSUM " << r.stiffness.frobenius_norm()
                << '\n';
      const double den = r.stiffness(0, 0) + r.stiffness(0, 1);
      const double nu = (std::abs(den) > 1e-30) ? r.stiffness(0, 1) / den : 0.0;
      std::cout << "C11 " << r.stiffness(0, 0) << " C12 " << r.stiffness(0, 1)
                << " nu_eff " << nu << '\n';
    }
    rc = r.all_converged() ? 0 : 1;
  }
  MPI_Finalize();
  return rc;
}
