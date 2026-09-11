// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file homogenize.cpp
 * @brief Forward periodic homogenization driver (issue #161, Stage 1).
 *
 * Prints the engineering Voigt \f$C_H\f$ of a two-phase unit cell. Inverse
 * design is a later binary on the same homogenizer; this executable is the
 * oracle the inverse loop will call six times per iteration.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/homogenization.hpp>

namespace {

struct Config {
  int nx{16};
  int ny{16};
  int nz{16};
  double dx{1.0};
  double E_solid{1.0};
  double nu_solid{0.3};
  double E_void{0.25};
  double nu_void{0.3};
  std::string shape{"homogeneous"};
  double volume{1.0};
};

void usage(std::ostream &os, const char *exe) {
  os << "Usage: " << exe << " [--key=value]...\n"
     << "  Forward FFT homogenization of a periodic two-phase unit cell.\n"
     << "  Issue #161 Stage 1; not a compliance topology-optimization demo.\n\n"
     << "  --nx --ny --nz     grid (default 16^3)\n"
     << "  --dx               spacing (default 1)\n"
     << "  --E-solid --nu-solid --E-void --nu-void\n"
     << "  --shape            homogeneous | laminate-z | sphere\n"
     << "  --volume           h for homogeneous; ignored otherwise\n";
}

bool parse_double(std::string_view v, double &out) {
  try {
    std::size_t n = 0;
    out = std::stod(std::string(v), &n);
    return n == v.size();
  } catch (...) {
    return false;
  }
}
bool parse_int(std::string_view v, int &out) {
  try {
    std::size_t n = 0;
    out = std::stoi(std::string(v), &n);
    return n == v.size() && out > 0;
  } catch (...) {
    return false;
  }
}

bool parse(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view tok(argv[i]);
    if (tok == "--help" || tok == "-h") return false;
    const auto eq = tok.find('=');
    if (!tok.starts_with("--") || eq == std::string_view::npos) return false;
    const auto key = tok.substr(2, eq - 2);
    const auto val = tok.substr(eq + 1);
    if (key == "nx") {
      if (!parse_int(val, cfg.nx)) return false;
    } else if (key == "ny") {
      if (!parse_int(val, cfg.ny)) return false;
    } else if (key == "nz") {
      if (!parse_int(val, cfg.nz)) return false;
    } else if (key == "dx") {
      if (!parse_double(val, cfg.dx) || cfg.dx <= 0.0) return false;
    } else if (key == "E-solid") {
      if (!parse_double(val, cfg.E_solid) || cfg.E_solid <= 0.0) return false;
    } else if (key == "nu-solid") {
      if (!parse_double(val, cfg.nu_solid)) return false;
    } else if (key == "E-void") {
      if (!parse_double(val, cfg.E_void) || cfg.E_void <= 0.0) return false;
    } else if (key == "nu-void") {
      if (!parse_double(val, cfg.nu_void)) return false;
    } else if (key == "shape") {
      cfg.shape = std::string(val);
    } else if (key == "volume") {
      if (!parse_double(val, cfg.volume)) return false;
    } else {
      return false;
    }
  }
  return cfg.shape == "homogeneous" || cfg.shape == "laminate-z" ||
         cfg.shape == "sphere";
}

void print_matrix(std::ostream &os, const pfc::apps::Voigt6 &C) {
  os << std::setprecision(10) << std::scientific;
  for (int i = 0; i < pfc::apps::kVoigtDim; ++i) {
    for (int j = 0; j < pfc::apps::kVoigtDim; ++j) {
      if (j) os << ' ';
      os << std::setw(16) << C(i, j);
    }
    os << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  Config cfg;
  if (!parse(argc, argv, cfg)) {
    if (rank == 0) usage(std::cerr, argc >= 1 ? argv[0] : "openpfc_homogenize");
    MPI_Finalize();
    return 2;
  }

  const pfc::Domain domain = pfc::domain::create(
      pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}),
      pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, MPI_COMM_WORLD);
  auto h = pfc::data::field_from_inbox<double>(domain,
                                               stack.fft().get_inbox_bounds());

  const auto n = h.local_size();
  const double Lx = cfg.nx * cfg.dx;
  const double Ly = cfg.ny * cfg.dx;
  const double Lz = cfg.nz * cfg.dx;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = h.coords(i, j, k);
        double hv = cfg.volume;
        if (cfg.shape == "laminate-z") {
          hv = (x[2] < 0.5 * Lz) ? 1.0 : 0.0;
        } else if (cfg.shape == "sphere") {
          const double cx = 0.5 * Lx, cy = 0.5 * Ly, cz = 0.5 * Lz;
          const double R = 0.25 * std::min({Lx, Ly, Lz});
          const double w = 1.5 * cfg.dx;
          const double r = std::sqrt((x[0] - cx) * (x[0] - cx) +
                                     (x[1] - cy) * (x[1] - cy) +
                                     (x[2] - cz) * (x[2] - cz));
          hv = 0.5 * (1.0 - std::tanh((r - R) / w));
        }
        h(i, j, k) = hv;
      }
    }
  }
  h.note_host_write();

  pfc::apps::MicroelasticityParams p;
  p.c_solid = pfc::apps::Stiffness::isotropic(cfg.E_solid, cfg.nu_solid);
  p.c_liquid = pfc::apps::Stiffness::isotropic(cfg.E_void, cfg.nu_void);
  p.tol_el = 1.0e-8;
  p.n_el_iter = 80;
  p.warm_start = false;
  p.comm = MPI_COMM_WORLD;

  pfc::apps::PeriodicHomogenizer hom(domain, stack.fft(), p);
  const auto r = hom.compute(h);

  if (rank == 0) {
    std::cout << "shape " << cfg.shape << " grid " << cfg.nx << 'x' << cfg.ny
              << 'x' << cfg.nz << " ranks " << nproc << '\n';
    std::cout << std::setprecision(10) << "volume_fraction " << r.volume_fraction
              << " spd " << (pfc::apps::is_spd(r.stiffness) ? "yes" : "no")
              << " converged " << (r.all_converged() ? "yes" : "no") << '\n';
    std::cout << "C_H (engineering Voigt, order 11 22 33 23 13 12)\n";
    print_matrix(std::cout, r.stiffness);
    std::cout << std::setprecision(16)
              << "HOMOGENIZATION_CHECKSUM " << r.stiffness.frobenius_norm()
              << '\n';
  }

  const int rc = r.all_converged() ? 0 : 1;
  MPI_Finalize();
  return rc;
}
