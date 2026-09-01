// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_pf_directional_engine.cpp
 * @brief Explicit Euler FTA Al-Cu on a regular grid (2D Nz=1 or 3D brick).
 *
 * Persistent fields: φ (and ψ if Glasner) per grain, c, ∂tφ (antitrapping),
 * e^u and u by default. Iso solute fills nodal α/β once per cell.
 * Recomputed: anisotropy fluxes unless STORE_AUX (see recompute.hpp).
 * Nz=1 with n_dim=2 is the 2D-equivalent path. Classic AMR is not implemented.
 */

#include <alloy_pf_directional/engine.hpp>

#include <alloy_pf_directional/isotropic_fd.hpp>
#include <alloy_pf_directional/noise.hpp>
#include <alloy_pf_directional/pad_field.hpp>
#include <alloy_pf_directional/recompute.hpp>
#include <alloy_pf_directional/window.hpp>
#include <openpfc/frontend/io/png_writer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

namespace {

using alloy_pf_directional::FieldAs3;
using alloy_pf_directional::Mat3;
using alloy_pf_directional::PadField;
using alloy_pf_directional::Physics;
using alloy_pf_directional::a_at_nodal;
using alloy_pf_directional::beta_glasner_from_phi;
using alloy_pf_directional::eu_at;
using alloy_pf_directional::eu_from_phi_c;
using alloy_pf_directional::fill_ghosts;
using alloy_pf_directional::flux_aniso_x;
using alloy_pf_directional::flux_aniso_y;
using alloy_pf_directional::flux_aniso_z;
using alloy_pf_directional::phi_at;
using alloy_pf_directional::q_of;
using alloy_pf_directional::u_from_eu;
using alloy_pf_directional::x_of;
using alloy_pf_directional::y_of;
using alloy_pf_directional::z_of;

double interior_sum(const PadField &f) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) collapse(2) schedule(static)
  for (int j = 1; j <= f.Ny; ++j) {
    for (int i = 1; i <= f.Nx; ++i) {
      for (int k = 1; k <= f.Nz; ++k) {
        s += f(i, j, k);
      }
    }
  }
  return s;
}

void write_c_phi_png(const char *path, const PadField &p1, const PadField &p2,
                     const PadField &c, double clo) {
  const int Nx = p1.Nx;
  const int Ny = p1.Ny;
  const int ksl = (p1.Nz + 1) / 2;
  const int H = 2 * Ny;
  std::vector<double> img(static_cast<std::size_t>(Nx) * static_cast<std::size_t>(H), 0.0);
  const double cmax = std::max(2.0 * clo, 1.0e-12);
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      const double ph = alloy_pf_directional::phi_eff_two(p1(i, j, ksl), p2(i, j, ksl));
      const double cv = std::max(0.0, std::min(1.0, c(i, j, ksl) / cmax));
      const double pv = std::max(0.0, std::min(1.0, 0.5 * (ph + 1.0)));
      img[static_cast<std::size_t>(i - 1 + (j - 1) * Nx)] = cv;
      img[static_cast<std::size_t>(i - 1 + (Ny + j - 1) * Nx)] = pv;
    }
  }
  pfc::io::write_png_grayscale_from_doubles(path, Nx, H, img.data(), 0.0, 1.0);
}

void write_c_png(const char *path, const PadField &c, double clo) {
  const int Nx = c.Nx;
  const int Ny = c.Ny;
  const int ksl = (c.Nz + 1) / 2;
  std::vector<double> img(static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny), 0.0);
  const double cmax = std::max(2.0 * clo, 1.0e-12);
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      img[static_cast<std::size_t>(i - 1 + (j - 1) * Nx)] =
          std::max(0.0, std::min(1.0, c(i, j, ksl) / cmax));
    }
  }
  pfc::io::write_png_grayscale_from_doubles(path, Nx, Ny, img.data(), 0.0, 1.0);
}

void write_c_phi_vti(const std::string &path, const PadField &p1, const PadField &p2,
                     const PadField &c, double dx) {
  const int Nx = p1.Nx;
  const int Ny = p1.Ny;
  const int Nz = p1.Nz;
  const std::size_t n = static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny) *
                        static_cast<std::size_t>(Nz);
  std::vector<double> buf_c(n), buf_phi(n);
  for (int k = 1; k <= Nz; ++k) {
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        const std::size_t p = static_cast<std::size_t>(i - 1) +
                              static_cast<std::size_t>(j - 1) * static_cast<std::size_t>(Nx) +
                              static_cast<std::size_t>(k - 1) * static_cast<std::size_t>(Nx) *
                                  static_cast<std::size_t>(Ny);
        buf_c[p] = c(i, j, k);
        buf_phi[p] = alloy_pf_directional::phi_eff_two(p1(i, j, k), p2(i, j, k));
      }
    }
  }
  const double ox = 0.5 * dx;
  const double oy = 0.5 * dx;
  const double oz = Nz > 1 ? 0.5 * dx : 0.0;
  const int z1 = Nz > 1 ? (Nz - 1) : 0;
  const std::uint64_t nbytes = static_cast<std::uint64_t>(n) * sizeof(double);
  const std::uint64_t off_phi = sizeof(std::uint64_t) + nbytes;

  std::ofstream file(path, std::ios::binary);
  file << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
  file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" "
          "header_type=\"UInt64\">\n";
  file << "  <ImageData WholeExtent=\"0 " << (Nx - 1) << " 0 " << (Ny - 1) << " 0 " << z1
       << "\" Origin=\"" << ox << " " << oy << " " << oz << "\" Spacing=\"" << dx << " "
       << dx << " " << dx << "\">\n";
  file << "    <Piece Extent=\"0 " << (Nx - 1) << " 0 " << (Ny - 1) << " 0 " << z1
       << "\">\n";
  file << "      <PointData>\n";
  file << "        <DataArray type=\"Float64\" Name=\"c\" NumberOfComponents=\"1\" "
          "format=\"appended\" offset=\"0\"/>\n";
  file << "        <DataArray type=\"Float64\" Name=\"phi\" NumberOfComponents=\"1\" "
          "format=\"appended\" offset=\""
       << off_phi << "\"/>\n";
  file << "      </PointData>\n";
  file << "    </Piece>\n";
  file << "  </ImageData>\n";
  file << "  <AppendedData encoding=\"raw\">\n";
  file << "_";
  file.write(reinterpret_cast<const char *>(&nbytes), sizeof(nbytes));
  file.write(reinterpret_cast<const char *>(buf_c.data()), static_cast<std::streamsize>(nbytes));
  file.write(reinterpret_cast<const char *>(&nbytes), sizeof(nbytes));
  file.write(reinterpret_cast<const char *>(buf_phi.data()), static_cast<std::streamsize>(nbytes));
  file << "\n  </AppendedData>\n";
  file << "</VTKFile>\n";
}

struct FieldStats {
  double avg_phi = 0.0;
  double min_phi = 0.0;
  double max_phi = 0.0;
  double avg_c = 0.0;
  double min_c = 0.0;
  double max_c = 0.0;
  int n_bad_phi = 0;
  int n_bad_c = 0;
  double max_overlap = 0.0;
  int n_both_solid = 0;
};

inline bool value_blown(double v) noexcept {
  return !std::isfinite(v) || std::abs(v) > alloy_pf_directional::kFieldBlowAbs;
}

inline const char *value_blown_why(double v) noexcept {
  if (std::isnan(v)) {
    return "NaN";
  }
  if (!std::isfinite(v)) {
    return "Inf";
  }
  return "|value| exceeds 1e6";
}

FieldStats reduce_field_stats(const PadField &p1, const PadField &p2, const PadField &c,
                              int n_grains) {
  const int Nx = p1.Nx;
  const int Ny = p1.Ny;
  const int Nz = p1.Nz;
  const double ncell =
      static_cast<double>(Nx) * static_cast<double>(Ny) * static_cast<double>(Nz);
  double sum_phi = 0.0;
  double sum_c = 0.0;
  double min_phi = std::numeric_limits<double>::infinity();
  double max_phi = -std::numeric_limits<double>::infinity();
  double min_c = std::numeric_limits<double>::infinity();
  double max_c = -std::numeric_limits<double>::infinity();
  int n_bad_phi = 0;
  int n_bad_c = 0;
  double max_overlap = 0.0;
  int n_both_solid = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : sum_phi, sum_c, n_bad_phi, n_bad_c, n_both_solid) \
    reduction(min : min_phi, min_c) reduction(max : max_phi, max_c, max_overlap)
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      for (int k = 1; k <= Nz; ++k) {
        const double ph =
            (n_grains == 1) ? p1(i, j, k) : alloy_pf_directional::phi_eff_two(p1(i, j, k), p2(i, j, k));
        const double cv = c(i, j, k);
        sum_phi += ph;
        sum_c += cv;
        min_phi = std::min(min_phi, ph);
        max_phi = std::max(max_phi, ph);
        min_c = std::min(min_c, cv);
        max_c = std::max(max_c, cv);
        if (n_grains == 2) {
          const double ov = alloy_pf_directional::phi_hat(p1(i, j, k)) * alloy_pf_directional::phi_hat(p2(i, j, k));
          max_overlap = std::max(max_overlap, ov);
          if (p1(i, j, k) > 0.0 && p2(i, j, k) > 0.0) {
            ++n_both_solid;
          }
        }
        if (value_blown(ph) ||
            (n_grains == 2 && (value_blown(p1(i, j, k)) || value_blown(p2(i, j, k))))) {
          ++n_bad_phi;
        }
        if (value_blown(cv)) {
          ++n_bad_c;
        }
      }
    }
  }
  FieldStats s;
  s.avg_phi = sum_phi / ncell;
  s.min_phi = min_phi;
  s.max_phi = max_phi;
  s.avg_c = sum_c / ncell;
  s.min_c = min_c;
  s.max_c = max_c;
  s.n_bad_phi = n_bad_phi;
  s.n_bad_c = n_bad_c;
  s.max_overlap = max_overlap;
  s.n_both_solid = n_both_solid;
  return s;
}

std::string describe_blowup(const PadField &p1, const PadField &p2, const PadField &c,
                            int n_grains, int euler_step, double dt) {
  auto line = [&](const char *name, double v, int i, int j, int k) {
    std::ostringstream os;
    os << std::setprecision(16);
    os << "field blow-up at step " << euler_step << " t=" << (dt * static_cast<double>(euler_step))
       << " s: " << name << " is " << value_blown_why(v) << " (" << v << ") at i=" << i
       << " j=" << j << " k=" << k;
    return os.str();
  };
  for (int k = 1; k <= p1.Nz; ++k) {
    for (int j = 1; j <= p1.Ny; ++j) {
      for (int i = 1; i <= p1.Nx; ++i) {
        if (value_blown(p1(i, j, k))) {
          return line("phi1", p1(i, j, k), i, j, k);
        }
        if (n_grains == 2 && value_blown(p2(i, j, k))) {
          return line("phi2", p2(i, j, k), i, j, k);
        }
        if (value_blown(c(i, j, k))) {
          return line("c", c(i, j, k), i, j, k);
        }
      }
    }
  }
  std::ostringstream os;
  os << "field blow-up at step " << euler_step << " t=" << (dt * static_cast<double>(euler_step))
     << " s: phi or c is NaN/Inf or |value| > " << alloy_pf_directional::kFieldBlowAbs
     << " (no interior cell located)";
  return os.str();
}

void write_grain_png(const char *path, const PadField &p1, const PadField &p2) {
  const int ksl = (p1.Nz + 1) / 2;
  std::vector<double> img(static_cast<std::size_t>(p1.Nx) * static_cast<std::size_t>(p1.Ny));
  for (int j = 1; j <= p1.Ny; ++j) {
    for (int i = 1; i <= p1.Nx; ++i) {
      img[static_cast<std::size_t>(i - 1 + (j - 1) * p1.Nx)] = p1(i, j, ksl) - p2(i, j, ksl);
    }
  }
  pfc::io::write_png_grayscale_from_doubles(path, p1.Nx, p1.Ny, img.data(), -2.0, 2.0);
}

void write_field_raw(const std::string &path, const PadField &f) {
  std::ofstream out(path, std::ios::binary);
  for (int k = 1; k <= f.Nz; ++k) {
    for (int j = 1; j <= f.Ny; ++j) {
      for (int i = 1; i <= f.Nx; ++i) {
        const double v = f(i, j, k);
        out.write(reinterpret_cast<const char *>(&v), sizeof(double));
      }
    }
  }
}

void refresh_eu_u(PadField &eu, PadField &u, const PadField &p1, const PadField &p2,
                  const PadField &c, int n_grains, double ke, double clo, bool periodic_y,
                  bool periodic_z) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int kk = 1; kk <= eu.Nz; ++kk) {
    for (int j = 1; j <= eu.Ny; ++j) {
      for (int i = 1; i <= eu.Nx; ++i) {
        const double euv = eu_at(p1, p2, c, n_grains, ke, clo, i, j, kk);
        eu(i, j, kk) = euv;
        u(i, j, kk) = u_from_eu(euv);
      }
    }
  }
  fill_ghosts(eu, periodic_y, periodic_z);
  fill_ghosts(u, periodic_y, periodic_z);
}

void fill_solute_nodal(PadField &a_diff, PadField &a_at, PadField &beta, PadField *u_opt,
                       const PadField &phi1, const PadField &phi2, const PadField &c,
                       const PadField &dphi1, const PadField &dphi2, const PadField *eu_opt,
                       int n_grains, bool use_glasner, const Physics &phys) {
  const int Nx = a_diff.Nx;
  const int Ny = a_diff.Ny;
  const int Nz = a_diff.Nz;
  const double ke = phys.ke;
  const double clo = phys.clo;
  const int k_lo = (Nz > 1) ? 0 : 1;
  const int k_hi = (Nz > 1) ? Nz + 1 : Nz;
#pragma omp parallel for collapse(2) schedule(static)
  for (int kk = k_lo; kk <= k_hi; ++kk) {
    for (int j = 0; j <= Ny + 1; ++j) {
      for (int i = 0; i <= Nx + 1; ++i) {
        const double ph = phi_at(phi1, phi2, n_grains, i, j, kk);
        const double euv =
            eu_opt ? (*eu_opt)(i, j, kk) : eu_from_phi_c(ph, c(i, j, kk), ke, clo);
        a_diff(i, j, kk) = phys.DL * c(i, j, kk) * q_of(ph, ke);
        const double dte = dphi1(i, j, kk) + dphi2(i, j, kk);
        a_at(i, j, kk) =
            a_at_nodal(ph, euv, dte, phys.a_at, phys.A_trap, phys.W0, clo, ke, use_glasner);
        if (use_glasner) {
          beta(i, j, kk) = beta_glasner_from_phi(ph);
        } else {
          beta(i, j, kk) = ph;
        }
        if (u_opt) {
          (*u_opt)(i, j, kk) = u_from_eu(euv);
        }
      }
    }
  }
}

void fill_aniso_fluxes(PadField &jx, PadField &jy, PadField &jz, const PadField &pf, bool dim3,
                       bool periodic_y, bool periodic_z, double inv_dx, const Physics &phys,
                       const Mat3 &R) {
  const int Nx = pf.Nx;
  const int Ny = pf.Ny;
  const int Nz = pf.Nz;
#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 0; i <= Nx; ++i) {
      for (int kk = 1; kk <= Nz; ++kk) {
        jx(i, j, kk) = flux_aniso_x(pf, i, j, kk, Nx, inv_dx, dim3, phys, R);
      }
    }
  }
#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 0; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      for (int kk = 1; kk <= Nz; ++kk) {
        jy(i, j, kk) = flux_aniso_y(pf, i, j, kk, Ny, inv_dx, dim3, periodic_y, phys, R);
      }
    }
  }
  if (dim3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        for (int kk = 0; kk <= Nz; ++kk) {
          jz(i, j, kk) = flux_aniso_z(pf, i, j, kk, Nz, inv_dx, periodic_z, phys, R);
        }
      }
    }
  }
}

std::size_t pad_bytes(const PadField &f) noexcept {
  return f.a.size() * sizeof(double);
}

struct FarWallC {
  double mean = 0.0;
  double max_abs_dev = 0.0;
  bool hit = false;
};

/** Every liquid-side far-face pixel (x = L), not a line average. */
FarWallC scan_right_wall_c(const PadField &c, double c_inf) {
  FarWallC out;
  const double tol = alloy_pf_directional::kWallCRel * std::max(std::abs(c_inf), 1.0e-12);
  double s = 0.0;
  int n = 0;
  for (int kk = 1; kk <= c.Nz; ++kk) {
    for (int j = 1; j <= c.Ny; ++j) {
      const double cc = c(c.Nx, j, kk);
      s += cc;
      ++n;
      const double dev = std::abs(cc - c_inf);
      if (dev > out.max_abs_dev) {
        out.max_abs_dev = dev;
      }
      if (dev > tol) {
        out.hit = true;
      }
    }
  }
  out.mean = (n > 0) ? s / static_cast<double>(n) : 0.0;
  return out;
}

double interpolate_leading_tip(const PadField &phi, double dx, int shift_cells) {
  double xt = 0.0;
  for (int k = 1; k <= phi.Nz; ++k) {
    for (int j = 1; j <= phi.Ny; ++j) {
      for (int i = 1; i < phi.Nx; ++i) {
        const double p0 = phi(i, j, k);
        const double p1 = phi(i + 1, j, k);
        if (p0 >= 0.0 && p1 < 0.0) {
          const double a = p0 / (p0 - p1 + 1.0e-30);
          xt = std::max(xt, x_of(i, dx, shift_cells) + a * dx);
        }
      }
    }
  }
  return xt;
}

/** Instantaneous partition on a 1D line: k = c_s(φ≈+0.9) / max c in the liquid spike. */
struct Partition1D {
  double cs = std::numeric_limits<double>::quiet_NaN();
  double cl = std::numeric_limits<double>::quiet_NaN();
  double k = std::numeric_limits<double>::quiet_NaN();
};

Partition1D measure_partition_1d(const PadField &phi, const PadField &c, double dx, double W0) {
  Partition1D out;
  if (phi.Ny != 1 || phi.Nz != 1) {
    return out;
  }
  constexpr int j = 1;
  constexpr int kk = 1;
  int i_if = -1;
  for (int i = 1; i < phi.Nx; ++i) {
    if (phi(i, j, kk) >= 0.0 && phi(i + 1, j, kk) < 0.0) {
      i_if = i;
    }
  }
  if (i_if < 0) {
    return out;
  }
  const int nW = std::max(4, static_cast<int>(std::ceil(6.0 * W0 / dx)));
  int i_s = -1;
  for (int i = i_if; i >= std::max(1, i_if - nW); --i) {
    if (phi(i, j, kk) >= 0.9) {
      i_s = i;
      break;
    }
  }
  const int i1 = std::min(phi.Nx, i_if + nW);
  double cl = -1.0;
  for (int i = i_if + 1; i <= i1; ++i) {
    cl = std::max(cl, c(i, j, kk));
  }
  if (i_s < 0 || !(cl > 0.0)) {
    return out;
  }
  out.cs = c(i_s, j, kk);
  out.cl = cl;
  out.k = out.cs / out.cl;
  return out;
}

} // namespace

namespace alloy_pf_directional::engine {

RunResult run(const RunConfig &cfg, bool skip_png, bool quiet) {
  const Physics phys = cfg.phys;
  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int Nz = cfg.Nz < 1 ? 1 : cfg.Nz;
  const bool dim3 = (phys.n_dim >= 3 && Nz > 1);
  const double dx = phys.dx;
  const double dt = phys.dt;
  const double inv_dx = 1.0 / dx;
  const double k = phys.ke;
  const double clo = phys.clo;
  const Mat3 R1 = bunge_crystal_to_lab(phys.phi1_g1, phys.Phi_g1, phys.phi2_g1);
  const Mat3 R2 = bunge_crystal_to_lab(phys.phi1_g2, phys.Phi_g2, phys.phi2_g2);

  if (cfg.num_threads > 0) {
    omp_set_num_threads(cfg.num_threads);
  }
  const int nthr = omp_get_max_threads();

  PadField phi1(Nx, Ny, Nz), phi2(Nx, Ny, Nz), psi1, psi2;
  PadField c(Nx, Ny, Nz);
  PadField dphi1(Nx, Ny, Nz), dphi2(Nx, Ny, Nz);
  PadField dc(Nx, Ny, Nz); // Jacobi scratch for solute (c frozen while forming RHS)
  const bool store_eu = cfg.store_eu;
  const bool store_aux = cfg.store_aux;
  PadField eu_f, u_f, jx, jy, jz;
  PadField a_diff_f, a_at_f, beta_at_f;
  if (store_eu) {
    eu_f = PadField(Nx, Ny, Nz);
    u_f = PadField(Nx, Ny, Nz);
  }
  if (store_aux) {
    jx = PadField(Nx, Ny, Nz);
    jy = PadField(Nx, Ny, Nz);
    if (dim3) {
      jz = PadField(Nx, Ny, Nz);
    }
  }
  const bool use_glasner = cfg.use_glasner;
  const bool use_iso = cfg.use_isotropic;
  if (use_iso) {
    a_diff_f = PadField(Nx, Ny, Nz);
    a_at_f = PadField(Nx, Ny, Nz);
    beta_at_f = PadField(Nx, Ny, Nz);
    if (!store_eu) {
      u_f = PadField(Nx, Ny, Nz);
    }
  }
  const bool periodic_y = cfg.periodic_y;
  const bool periodic_z = dim3 && cfg.periodic_z;
  const int n_grains = cfg.n_grains;
  auto fill_bc = [&](PadField &fld) { fill_ghosts(fld, periodic_y, periodic_z); };
  auto fill_bc_g2 = [&](PadField &fld) {
    if (n_grains == 2) {
      fill_bc(fld);
    }
  };
  if (use_glasner) {
    psi1 = PadField(Nx, Ny, Nz);
    psi2 = PadField(Nx, Ny, Nz);
  }

  WindowState win;
  win.lab_Nx = cfg.lab_Lx > 0.0 ? std::max(Nx, static_cast<int>(std::ceil(cfg.lab_Lx / dx)))
                                : Nx;
  win.lab_Lx = cfg.lab_Lx > 0.0 ? cfg.lab_Lx : x_of(Nx, dx);
  BlockSkip skip =
      make_block_skip(Nx, Ny, Nz, cfg.block_skip, cfg.block_skip_tol_phi,
                      cfg.block_skip_tol_c, cfg.block_skip_refresh);
  const BlockSkip *skip_ptr = skip.enabled() ? &skip : nullptr;

  const double Ly = y_of(Ny, dx);
  const double Lz = dim3 ? z_of(Nz, dx) : 0.0;
  double y1 = 0.0;
  double y2 = 0.0;
  two_grain_seed_ys(Ny, dx, y1, y2);
  const double ymid = 0.5 * (Ly + 0.5 * dx);
  const double zmid = dim3 ? 0.5 * (Lz + 0.5 * dx) : 0.0;
  const double Rseed = phys.r_seed;

#pragma omp parallel for collapse(2) schedule(static)
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      for (int kk = 1; kk <= Nz; ++kk) {
        const double x = x_of(i, dx);
        const double y = y_of(j, dx);
        const double z = dim3 ? z_of(kk, dx) : 0.0;
        if (n_grains == 1) {
          const double dy = y - ymid;
          const double dz = dim3 ? (z - zmid) : 0.0;
          const double xint =
              cfg.seed_depth +
              cfg.seed_bump * std::exp(-0.5 * (dy * dy + dz * dz) /
                                       (cfg.seed_bump_sigma * cfg.seed_bump_sigma));
          const double s = -(x - xint) / phys.W0;
          if (use_glasner) {
            psi1(i, j, kk) = std::max(-8.0, std::min(8.0, s));
            psi2(i, j, kk) = -8.0;
            phi1(i, j, kk) = phi_from_psi(psi1(i, j, kk));
            phi2(i, j, kk) = -1.0;
          } else {
            phi1(i, j, kk) = -std::tanh((x - xint) / (std::sqrt(2.0) * phys.W0));
            phi2(i, j, kk) = -1.0;
          }
        } else {
          double s1 = 0.0;
          double s2 = 0.0;
          two_grain_seed_s(x, y, z, y1, y2, zmid, Rseed, phys.W0, dim3, s1, s2);
          if (use_glasner) {
            apply_two_grain_seed(s1, s2, true, phi1(i, j, kk), phi2(i, j, kk), &psi1(i, j, kk),
                                 &psi2(i, j, kk));
          } else {
            apply_two_grain_seed(s1, s2, false, phi1(i, j, kk), phi2(i, j, kk), nullptr,
                                 nullptr);
          }
        }
        const double ph =
            (n_grains == 1) ? phi1(i, j, kk) : phi_eff_two(phi1(i, j, kk), phi2(i, j, kk));
        c(i, j, kk) = c_eq(ph, k, clo);
      }
    }
  }
  fill_bc(phi1);
  fill_bc_g2(phi2);
  fill_bc(c);
  if (use_glasner) {
    fill_bc(psi1);
    fill_bc_g2(psi2);
  }

  const double dV = dim3 ? dx * dx * dx : dx * dx;
  const double mass0 = interior_sum(c) * dV;
  if (skip.enabled()) {
    refresh_block_skip(skip, phi1, phi2, c, phys, n_grains);
  }

  std::ofstream hist(cfg.output_dir + "/history.tsv");
  hist << std::setprecision(16);
  hist << "# t mass x_tip min_phi max_phi min_c max_c k_part c_s c_l c_wall c_wall_maxdev\n";

  std::ofstream flog(cfg.output_dir + "/fields.log");
  flog << std::setprecision(16);
  flog << "# step t avg_phi min_phi max_phi avg_c min_c max_c x_tip mass max_overlap n_both_solid\n";

  {
    std::ofstream meta(cfg.output_dir + "/meta.txt");
    meta << std::setprecision(16);
    meta << "ke " << phys.ke << "\n";
    meta << "mle " << phys.mle << "\n";
    meta << "clo " << phys.clo << "\n";
    meta << "W0 " << phys.W0 << "\n";
    meta << "d0 " << phys.d0 << "\n";
    meta << "DL " << phys.DL << "\n";
    meta << "lambda " << phys.lambda << "\n";
    meta << "tau0 " << phys.tau0 << "\n";
    meta << "tau_beta " << phys.tau_beta << "\n";
    meta << "tau_a2 " << phys.tau_a2 << "\n";
    meta << "u_corr " << 1 << "\n";
    meta << "dx " << dx << "\n";
    meta << "dt " << dt << "\n";
    meta << "n_dim " << phys.n_dim << "\n";
    meta << "dt_cfl_c " << phys.dt_cfl_c << "\n";
    meta << "dt_cfl_phi " << phys.dt_cfl_phi << "\n";
    meta << "dt_cfl_iface " << phys.dt_cfl_iface << "\n";
    meta << "dt_tau " << phys.dt_tau << "\n";
    meta << "dt_over_tau " << (phys.tau0 > 0.0 ? dt / phys.tau0 : 0.0) << "\n";
    meta << "G " << phys.G << "\n";
    meta << "Vp " << phys.Vp << "\n";
    meta << "delta_iso " << phys.delta_iso << "\n";
    meta << "x_tl " << phys.x_tl << "\n";
    meta << "omega_zhong " << (phys.omega_zhong ? 1 : 0) << "\n";
    meta << "omega " << phys.omega << "\n";
    meta << "omega_solidus " << omega_at_solidus(phys) << "\n";
    meta << "A_trap " << phys.A_trap << "\n";
    meta << "a2 " << phys.a2 << "\n";
    meta << "alpha_drag " << phys.alpha_drag << "\n";
    meta << "VD_pf " << phys.VD_pf << "\n";
    meta << "eps_c " << phys.eps_c << "\n";
    meta << "eps_k " << phys.eps_k << "\n";
    meta << "beta0 " << phys.beta0 << "\n";
    meta << "Nx " << Nx << "\n";
    meta << "Ny " << Ny << "\n";
    meta << "Nz " << Nz << "\n";
    meta << "n_steps " << cfg.n_steps << "\n";
    meta << "nsave " << cfg.nsave << "\n";
    meta << "n_hist " << cfg.n_hist << "\n";
    meta << "t_end " << cfg.t_end << "\n";
    meta << "n_grains " << n_grains << "\n";
    meta << "r_seed " << phys.r_seed << "\n";
    meta << "y_seed1 " << y1 << "\n";
    meta << "y_seed2 " << y2 << "\n";
    meta << "seed_gap " << ((y2 - y1) - 2.0 * phys.r_seed) << "\n";
    meta << "seed_depth " << cfg.seed_depth << "\n";
    meta << "seed_bump " << cfg.seed_bump << "\n";
    meta << "seed_bump_sigma " << cfg.seed_bump_sigma << "\n";
    meta << "use_glasner " << (use_glasner ? 1 : 0) << "\n";
    meta << "use_isotropic " << (use_iso ? 1 : 0) << "\n";
    meta << "periodic_y " << (cfg.periodic_y ? 1 : 0) << "\n";
    meta << "periodic_z " << (periodic_z ? 1 : 0) << "\n";
    meta << "vtk_every " << cfg.vtk_every << "\n";
    meta << "skip_vtk " << (cfg.skip_vtk ? 1 : 0) << "\n";
    meta << "field_blow_abs " << kFieldBlowAbs << "\n";
    meta << "noise_F0 " << cfg.noise_F0 << "\n";
    meta << "noise_seed " << cfg.noise_seed << "\n";
    meta << "window_enable " << (cfg.window_enable ? 1 : 0) << "\n";
    meta << "window_nx " << cfg.window_nx << "\n";
    meta << "block_skip " << cfg.block_skip << "\n";
    meta << "amr_deferred 1\n";
    meta << "store_eu " << (store_eu ? 1 : 0) << "\n";
    meta << "store_aux " << (store_aux ? 1 : 0) << "\n";
    meta << "stop_on_right " << (cfg.stop_on_right ? 1 : 0) << "\n";
    meta << "stop_on_far_c " << (cfg.stop_on_far_c ? 1 : 0) << "\n";
    meta << "wall_c_rel " << kWallCRel << "\n";
    meta << "c_inf " << clo << "\n";
    meta << "phi1_g1_deg " << (phys.phi1_g1 * 180.0 / std::acos(-1.0)) << "\n";
    meta << "phi1_g2_deg " << (phys.phi1_g2 * 180.0 / std::acos(-1.0)) << "\n";
  }

  const bool scale_io_off = cfg.timed_steps > 0;
  auto dump_c_phi_png = [&](int euler_step) {
    if (skip_png || scale_io_off) {
      return;
    }
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/output_c_phi_%d.png", cfg.output_dir.c_str(),
                  euler_step);
    write_c_phi_png(path, phi1, phi2, c, clo);
    if (n_grains == 2) {
      std::snprintf(path, sizeof(path), "%s/grains_%d.png", cfg.output_dir.c_str(),
                    euler_step);
      write_grain_png(path, phi1, phi2);
    }
  };

  auto dump_vtk = [&](int euler_step, bool force = false) {
    if (cfg.skip_vtk || cfg.vtk_every <= 0 || scale_io_off) {
      return;
    }
    if (!force && euler_step != 0 && euler_step % cfg.vtk_every != 0) {
      return;
    }
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/output_c_phi_%d.vti", cfg.output_dir.c_str(),
                  euler_step);
    write_c_phi_vti(path, phi1, phi2, c, dx);
  };

  auto dump_fields_log = [&](int euler_step, bool flush) {
    const FieldStats st = reduce_field_stats(phi1, phi2, c, n_grains);
    const double tnow = dt * static_cast<double>(euler_step);
    const double mass = interior_sum(c) * dV;
    const double xtip = interpolate_leading_tip(phi1, dx, win.shift_cells);
    flog << euler_step << ' ' << tnow << ' ' << st.avg_phi << ' ' << st.min_phi << ' '
         << st.max_phi << ' ' << st.avg_c << ' ' << st.min_c << ' ' << st.max_c << ' '
         << xtip << ' ' << mass << ' ' << st.max_overlap << ' ' << st.n_both_solid << '\n';
    const Partition1D part = measure_partition_1d(phi1, c, dx, phys.W0);
    const FarWallC wall = scan_right_wall_c(c, clo);
    hist << tnow << ' ' << mass << ' ' << xtip << ' ' << st.min_phi << ' ' << st.max_phi
         << ' ' << st.min_c << ' ' << st.max_c << ' ' << part.k << ' ' << part.cs << ' '
         << part.cl << ' ' << wall.mean << ' ' << wall.max_abs_dev << '\n';
    if (flush) {
      flog.flush();
      hist.flush();
    }
    return st;
  };

  const FieldStats st0 = dump_fields_log(0, true);
  dump_c_phi_png(0);
  dump_vtk(0);

  const int nprint_eff = quiet ? 0 : cfg.nprint;
  const int n_health = cfg.n_hist > 0 ? cfg.n_hist : kIoEveryLog;
  const int n_loop = cfg.timed_steps > 0
                         ? (cfg.warmup_steps + cfg.timed_steps)
                         : cfg.n_steps;
  const int warmup = cfg.timed_steps > 0 ? cfg.warmup_steps : 0;
  double t_halo = 0.0;
  double t_kern = 0.0;
  double t_ghost = 0.0, t_eu = 0.0, t_flux = 0.0, t_grain = 0.0, t_euler = 0.0, t_solute = 0.0,
         t_reduce = 0.0, t_io = 0.0;
  double t_timed0 = 0.0;
  double t_timed1 = 0.0;
  const double t_loop0 = omp_get_wtime();
  int n_done = 0;
  bool hit_right = false;
  bool hit_far_c = false;
  bool blew_up = false;
  std::string abort_reason;

  auto check_blowup = [&](int euler_step, const FieldStats *st_opt) -> bool {
    const FieldStats st = st_opt ? *st_opt : reduce_field_stats(phi1, phi2, c, n_grains);
    if (st.n_bad_phi == 0 && st.n_bad_c == 0) {
      return false;
    }
    abort_reason = describe_blowup(phi1, phi2, c, n_grains, euler_step, dt);
    blew_up = true;
    std::cerr << "ALCU_ABORT " << abort_reason << " (n_bad_phi=" << st.n_bad_phi
              << " n_bad_c=" << st.n_bad_c << " max_overlap=" << st.max_overlap
              << " n_both_solid=" << st.n_both_solid << ")\n";
    std::cout << "ALCU_ABORT " << abort_reason << " (n_bad_phi=" << st.n_bad_phi
              << " n_bad_c=" << st.n_bad_c << " max_overlap=" << st.max_overlap
              << " n_both_solid=" << st.n_both_solid << ")\n";
    {
      std::ofstream af(cfg.output_dir + "/abort.txt");
      af << std::setprecision(16);
      af << abort_reason << "\n";
      af << "max_overlap " << st.max_overlap << "\n";
      af << "n_both_solid " << st.n_both_solid << "\n";
    }
    flog << "# ABORT " << abort_reason << "\n";
    flog.flush();
    hist.flush();
    dump_c_phi_png(euler_step);
    dump_vtk(euler_step, true);
    return true;
  };

  if (check_blowup(0, &st0)) {
    n_done = 0;
  }

  for (int istep = 1; istep <= n_loop && !blew_up; ++istep) {
    if (cfg.timed_steps > 0 && istep == warmup + 1) {
      t_timed0 = omp_get_wtime();
      t_halo = 0.0;
      t_kern = 0.0;
      t_ghost = t_eu = t_flux = t_grain = t_euler = t_solute = t_reduce = t_io = 0.0;
    }
    const double t = dt * static_cast<double>(istep - 1);
    const double t_h0 = omp_get_wtime();
    fill_bc(phi1);
    fill_bc_g2(phi2);
    fill_bc(c);
    if (use_glasner) {
      fill_bc(psi1);
      fill_bc_g2(psi2);
    }
    const double t_h1 = omp_get_wtime();
    t_halo += t_h1 - t_h0;
    t_ghost += t_h1 - t_h0;
    const PadField &pf1 = use_glasner ? psi1 : phi1;
    const PadField &pf2 = use_glasner ? psi2 : phi2;
    const double t_k0 = omp_get_wtime();
    if (store_eu) {
      const double t_e0 = omp_get_wtime();
      refresh_eu_u(eu_f, u_f, phi1, phi2, c, n_grains, k, clo, periodic_y, periodic_z);
      t_eu += omp_get_wtime() - t_e0;
    }

    if (skip.enabled() && istep % skip.refresh == 0) {
      refresh_block_skip(skip, phi1, phi2, c, phys, n_grains);
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        for (int kk = 1; kk <= Nz; ++kk) {
          double ci = c(i, j, kk);
          if (std::isfinite(ci) && ci < kCMin && ci > -kFieldBlowAbs) {
            c(i, j, kk) = kCMin;
          }
        }
      }
    }

    auto step_grain = [&](PadField &dphi, const PadField &pf, const PadField &phi_self,
                          const PadField &phi_other, const Mat3 &R, int grain_id) {
#pragma omp parallel for collapse(2) schedule(static)
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          for (int kk = 1; kk <= Nz; ++kk) {
            if (skip_ptr && !skip_ptr->is_active(i, j, kk)) {
              dphi(i, j, kk) = 0.0;
              continue;
            }
            const double gx = 0.5 * inv_dx * (pf(i + 1, j, kk) - pf(i - 1, j, kk));
            const double gy = 0.5 * inv_dx * (pf(i, j + 1, kk) - pf(i, j - 1, kk));
            const double gz = dim3 ? 0.5 * inv_dx * (pf(i, j, kk + 1) - pf(i, j, kk - 1)) : 0.0;
            double jxv = 0.0, jyv = 0.0, jzv = 0.0, tau = 0.0, A = 0.0;
            cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.W0, phys.tau0, R, jxv, jyv,
                                  jzv, tau, A);
            (void)jxv;
            (void)jyv;
            (void)jzv;
            const double euv =
                store_eu ? eu_f(i, j, kk) : eu_at(phi1, phi2, c, n_grains, k, clo, i, j, kk);
            tau = tau_with_u_corr(tau, phys, euv);
            double aniso = 0.0;
            if (store_aux) {
              aniso = (jx(i, j, kk) - jx(i - 1, j, kk) + jy(i, j, kk) - jy(i, j - 1, kk)) *
                      inv_dx;
              if (dim3) {
                aniso += (jz(i, j, kk) - jz(i, j, kk - 1)) * inv_dx;
              }
            } else {
              aniso = (flux_aniso_x(pf, i, j, kk, Nx, inv_dx, dim3, phys, R) -
                       flux_aniso_x(pf, i - 1, j, kk, Nx, inv_dx, dim3, phys, R)) *
                      inv_dx;
              aniso += (flux_aniso_y(pf, i, j, kk, Ny, inv_dx, dim3, periodic_y, phys, R) -
                        flux_aniso_y(pf, i, j - 1, kk, Ny, inv_dx, dim3, periodic_y, phys, R)) *
                       inv_dx;
              if (dim3) {
                aniso += (flux_aniso_z(pf, i, j, kk, Nz, inv_dx, periodic_z, phys, R) -
                          flux_aniso_z(pf, i, j, kk - 1, Nz, inv_dx, periodic_z, phys, R)) *
                         inv_dx;
              }
            }
            const double ph = phi_self(i, j, kk);
            const double x = x_of(i, dx, win.shift_cells);
            const double therm = thermal_drive(phys, x, t);
            const double grain =
                grain_repulsion(ph, phi_other(i, j, kk), omega_used(phys, therm));
            double mag2 = gx * gx + gy * gy + gz * gz;
            if (use_iso) {
              const FieldAs3 pf3{pf};
              const double L_iso = alloy_pf_directional::iso::laplacian_iso(pf3, i, j, kk, dx, dim3);
              const double L_std = alloy_pf_directional::iso::laplacian_std(pf3, i, j, kk, dx, dim3);
              mag2 = alloy_pf_directional::iso::grad2_iso(pf3, i, j, kk, dx, dim3);
              aniso += phys.W0 * phys.W0 * A * A * (L_iso - L_std);
            }
            if (use_glasner) {
              const double W = phys.W0 * A;
              const double sqrt2 = std::sqrt(2.0);
              const double bulk = sqrt2 * ph * (1.0 - W * W * mag2) -
                                  sqrt2 * (1.0 - ph * ph) * (phys.lambda / (1.0 - k)) *
                                      (euv - 1.0 - therm);
              dphi(i, j, kk) =
                  (aniso + bulk) / tau + grain_dpsi_dt(grain, tau, ph, dt);
            } else {
              const double bulk = -f_prime(ph) - (phys.lambda / (1.0 - k)) * g_prime(ph) *
                                                     (euv - 1.0 - therm);
              dphi(i, j, kk) = (aniso + bulk + grain) / tau;
            }
            if (cfg.noise_F0 > 0.0) {
              const int knoise = (Nz == 1) ? 0 : kk;
              const double xi = gaussian_n01(cfg.noise_seed, istep, i, j, knoise, grain_id);
              dphi(i, j, kk) +=
                  use_glasner
                      ? fdt_psi_noise_rate(cfg.noise_F0, phys.W0, phys.n_dim, tau, dt, dV, ph, xi)
                      : fdt_phi_noise_rate(cfg.noise_F0, phys.W0, phys.n_dim, tau, dt, dV, ph, xi);
            }
          }
        }
      }
    };

    {
      if (store_aux) {
        const double t_f0 = omp_get_wtime();
        fill_aniso_fluxes(jx, jy, jz, pf1, dim3, periodic_y, periodic_z, inv_dx, phys, R1);
        t_flux += omp_get_wtime() - t_f0;
      }
      const double t_g0 = omp_get_wtime();
      step_grain(dphi1, pf1, phi1, phi2, R1, 1);
      t_grain += omp_get_wtime() - t_g0;
      if (n_grains == 2) {
        if (store_aux) {
          const double t_f0 = omp_get_wtime();
          fill_aniso_fluxes(jx, jy, jz, pf2, dim3, periodic_y, periodic_z, inv_dx, phys, R2);
          t_flux += omp_get_wtime() - t_f0;
        }
        const double t_g1 = omp_get_wtime();
        step_grain(dphi2, pf2, phi2, phi1, R2, 2);
        t_grain += omp_get_wtime() - t_g1;
      }
    }

    const double t_el0 = omp_get_wtime();
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        for (int kk = 1; kk <= Nz; ++kk) {
          if (skip_ptr && !skip_ptr->is_active(i, j, kk)) {
            continue;
          }
          if (use_glasner) {
            const double p1_old = phi1(i, j, kk);
            psi1(i, j, kk) += dt * dphi1(i, j, kk);
            psi1(i, j, kk) = std::max(-8.0, std::min(8.0, psi1(i, j, kk)));
            phi1(i, j, kk) = phi_from_psi(psi1(i, j, kk));
            if (n_grains == 2) {
              dphi1(i, j, kk) = (phi1(i, j, kk) - p1_old) / dt;
              const double p2_old = phi2(i, j, kk);
              psi2(i, j, kk) += dt * dphi2(i, j, kk);
              psi2(i, j, kk) = std::max(-8.0, std::min(8.0, psi2(i, j, kk)));
              phi2(i, j, kk) = phi_from_psi(psi2(i, j, kk));
              dphi2(i, j, kk) = (phi2(i, j, kk) - p2_old) / dt;
            } else {
              dphi1(i, j, kk) = dphi_dpsi_from_phi(p1_old) * dphi1(i, j, kk);
            }
          } else {
            phi1(i, j, kk) += dt * dphi1(i, j, kk);
            if (n_grains == 2) {
              phi2(i, j, kk) += dt * dphi2(i, j, kk);
            }
          }
        }
      }
    }
    t_euler += omp_get_wtime() - t_el0;
    const double t_h2 = omp_get_wtime();
    fill_bc(phi1);
    fill_bc_g2(phi2);
    fill_bc(dphi1);
    fill_bc_g2(dphi2);
    if (use_glasner) {
      fill_bc(psi1);
      fill_bc_g2(psi2);
    }
    {
      const double t_h3 = omp_get_wtime();
      t_ghost += t_h3 - t_h2;
      t_halo += t_h3 - t_h2;
    }
    if (store_eu) {
      const double t_e1 = omp_get_wtime();
      refresh_eu_u(eu_f, u_f, phi1, phi2, c, n_grains, k, clo, periodic_y, periodic_z);
      t_eu += omp_get_wtime() - t_e1;
    }

    const double t_s0 = omp_get_wtime();
    if (use_iso) {
      fill_solute_nodal(a_diff_f, a_at_f, beta_at_f, store_eu ? nullptr : &u_f, phi1, phi2, c,
                        dphi1, dphi2, store_eu ? &eu_f : nullptr, n_grains, use_glasner, phys);
#pragma omp parallel for collapse(2) schedule(static)
      for (int kk = 1; kk <= Nz; ++kk) {
        for (int j = 1; j <= Ny; ++j) {
          for (int i = 1; i <= Nx; ++i) {
            if (skip_ptr && !skip_ptr->is_active(i, j, kk)) {
              continue;
            }
            const double Dd =
                alloy_pf_directional::iso::div_alpha_grad(a_diff_f, u_f, i, j, kk, dx, dim3);
            const double Dat =
                alloy_pf_directional::iso::div_alpha_grad(a_at_f, beta_at_f, i, j, kk, dx, dim3);
            dc(i, j, kk) = Dd + Dat;
          }
        }
      }
#pragma omp parallel for collapse(2) schedule(static)
      for (int kk = 1; kk <= Nz; ++kk) {
        for (int j = 1; j <= Ny; ++j) {
          for (int i = 1; i <= Nx; ++i) {
            if (skip_ptr && !skip_ptr->is_active(i, j, kk)) {
              continue;
            }
            c(i, j, kk) += dt * dc(i, j, kk);
            if (std::isfinite(c(i, j, kk)) && c(i, j, kk) < kCMin &&
                c(i, j, kk) > -kFieldBlowAbs) {
              c(i, j, kk) = kCMin;
            }
          }
        }
      }
    } else {
      auto ph_at = [&](int ii, int jj, int kkz) {
        return phi_eff_two(phi1(ii, jj, kkz), phi2(ii, jj, kkz));
      };
      auto solute_face = [&](int i0, int j0, int k0, int i1, int j1, int k1, double ncomp,
                             bool noflux) -> double {
        if (noflux) {
          return 0.0;
        }
        const double p0 = ph_at(i0, j0, k0);
        const double p1 = ph_at(i1, j1, k1);
        const double dt0 = dphi1(i0, j0, k0) + dphi2(i0, j0, k0);
        const double dt1 = dphi1(i1, j1, k1) + dphi2(i1, j1, k1);
        const double phf = 0.5 * (p0 + p1);
        const double cf = 0.5 * (c(i0, j0, k0) + c(i1, j1, k1));
        const double euf = 0.5 * (eu_from_phi_c(p0, c(i0, j0, k0), k, clo) +
                                  eu_from_phi_c(p1, c(i1, j1, k1), k, clo));
        const double dtf = 0.5 * (dt0 + dt1);
        const double u0 = u_from_eu(eu_from_phi_c(p0, c(i0, j0, k0), k, clo));
        const double u1 = u_from_eu(eu_from_phi_c(p1, c(i1, j1, k1), k, clo));
        const double du = inv_dx * (u1 - u0);
        const double qf = q_of(phf, k);
        const double at = a_prime_trap(phf, phys.a_at, phys.A_trap);
        return -phys.DL * cf * qf * du - at * phys.W0 * clo * (1.0 - k) * euf * dtf * ncomp;
      };
#pragma omp parallel for collapse(2) schedule(static)
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          for (int kk = 1; kk <= Nz; ++kk) {
            if (skip_ptr && !skip_ptr->is_active(i, j, kk)) {
              continue;
            }
            auto n_of = [&](int ia, int ja, int ka, int ib, int jb, int kb) {
              const double p0 = ph_at(ia, ja, ka);
              const double p1 = ph_at(ib, jb, kb);
              const double px = inv_dx * ((ib != ia) ? (p1 - p0) : 0.0);
              const double py = inv_dx * ((jb != ja) ? (p1 - p0) : 0.0);
              const double pz = inv_dx * ((kb != ka) ? (p1 - p0) : 0.0);
              const double mag = std::sqrt(px * px + py * py + pz * pz + kGradEps * kGradEps);
              return std::array<double, 3>{px / mag, py / mag, pz / mag};
            };
            const auto nx = n_of(i, j, kk, i + 1, j, kk);
            const auto nxm = n_of(i - 1, j, kk, i, j, kk);
            const auto ny = n_of(i, j, kk, i, j + 1, kk);
            const auto nym = n_of(i, j - 1, kk, i, j, kk);
            double divj =
                (solute_face(i, j, kk, i + 1, j, kk, nx[0], i == Nx) -
                 solute_face(i - 1, j, kk, i, j, kk, nxm[0], i == 1)) *
                    inv_dx +
                (solute_face(i, j, kk, i, j + 1, kk, ny[1], !periodic_y && j == Ny) -
                 solute_face(i, j - 1, kk, i, j, kk, nym[1], !periodic_y && j == 1)) *
                    inv_dx;
            if (dim3) {
              const auto nz = n_of(i, j, kk, i, j, kk + 1);
              const auto nzm = n_of(i, j, kk - 1, i, j, kk);
              divj += (solute_face(i, j, kk, i, j, kk + 1, nz[2], !periodic_z && kk == Nz) -
                       solute_face(i, j, kk - 1, i, j, kk, nzm[2], !periodic_z && kk == 1)) *
                      inv_dx;
            }
            dc(i, j, kk) = -divj;
          }
        }
      }
#pragma omp parallel for collapse(2) schedule(static)
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          for (int kk = 1; kk <= Nz; ++kk) {
            if (skip_ptr && !skip_ptr->is_active(i, j, kk)) {
              continue;
            }
            c(i, j, kk) += dt * dc(i, j, kk);
            if (std::isfinite(c(i, j, kk)) && c(i, j, kk) < kCMin &&
                c(i, j, kk) > -kFieldBlowAbs) {
              c(i, j, kk) = kCMin;
            }
          }
        }
      }
    }
    t_solute += omp_get_wtime() - t_s0;
    t_kern += omp_get_wtime() - t_k0;

    if (cfg.stop_on_far_c) {
      const FarWallC wall = scan_right_wall_c(c, clo);
      if (wall.hit) {
        hit_far_c = true;
        abort_reason = "wall_c";
        n_done = istep;
        break;
      }
    }

    if (cfg.window_enable) {
      const double xtip = interpolate_leading_tip(phi1, dx, win.shift_cells);
      const int nsh = window_shift_count(xtip, win.shift_cells, dx, cfg.window_margin_left, Nx,
                                         cfg.window_margin_right);
      if (nsh > 0) {
        apply_window_shift(phi1, phi2, c, use_glasner ? &psi1 : nullptr,
                           use_glasner ? &psi2 : nullptr, phys, n_grains, use_glasner, nsh);
        win.shift_cells += nsh;
        fill_bc(phi1);
        fill_bc_g2(phi2);
        fill_bc(c);
        if (use_glasner) {
          fill_bc(psi1);
          fill_bc_g2(psi2);
        }
      }
    }

    if (nprint_eff > 0 && istep % nprint_eff == 0) {
      std::cout << "step " << istep << "/" << n_loop << " done\n";
    }

    const bool in_timed = cfg.timed_steps > 0 && istep > warmup;
    if (!in_timed || !scale_io_off) {
      const double t_r0 = omp_get_wtime();
      if (cfg.n_hist > 0 && istep % cfg.n_hist == 0) {
        const FieldStats st = dump_fields_log(istep, true);
        t_reduce += omp_get_wtime() - t_r0;
        if (check_blowup(istep, &st)) {
          n_done = istep;
          break;
        }
      } else if (n_health > 0 && istep % n_health == 0) {
        if (check_blowup(istep, nullptr)) {
          n_done = istep;
          break;
        }
        t_reduce += omp_get_wtime() - t_r0;
      } else {
        t_reduce += omp_get_wtime() - t_r0;
      }
      const double t_i0 = omp_get_wtime();
      dump_vtk(istep);
      if (cfg.nsave > 0 && istep % cfg.nsave == 0) {
        dump_c_phi_png(istep);
      }
      t_io += omp_get_wtime() - t_i0;
    }

    n_done = istep;
    if (cfg.timed_steps > 0 && istep == warmup + cfg.timed_steps) {
      t_timed1 = omp_get_wtime();
      break;
    }
    if (cfg.stop_on_right && !cfg.window_enable) {
      for (int kk = 1; kk <= Nz && !hit_right; ++kk) {
        for (int j = 1; j <= Ny; ++j) {
          const double ph =
              (n_grains == 1) ? phi1(Nx, j, kk) : phi_eff_two(phi1(Nx, j, kk), phi2(Nx, j, kk));
          if (ph > 0.0) {
            hit_right = true;
            break;
          }
        }
      }
      if (hit_right) {
        break;
      }
    } else if (cfg.stop_on_right && cfg.window_enable) {
      const double xtip = interpolate_leading_tip(phi1, dx, win.shift_cells);
      if (xtip >= win.lab_Lx - 2.0 * phys.W0) {
        hit_right = true;
        break;
      }
    }
  }

  if (cfg.timed_steps > 0 && t_timed1 <= t_timed0) {
    t_timed1 = omp_get_wtime();
  }
  const double t_loop1 = omp_get_wtime();
  flog.flush();
  hist.flush();

  if (n_done > 0 && !blew_up && !scale_io_off) {
    if (cfg.n_hist > 0 && n_done % cfg.n_hist != 0) {
      dump_fields_log(n_done, true);
    }
    if (cfg.vtk_every > 0 && n_done % cfg.vtk_every != 0) {
      dump_vtk(n_done, true);
    }
    if (cfg.nsave <= 0 || n_done % cfg.nsave != 0) {
      dump_c_phi_png(n_done);
    }
  }
  write_field_raw(cfg.output_dir + "/phi_final.raw", phi1);
  if (n_grains == 2) {
    write_field_raw(cfg.output_dir + "/phi2_final.raw", phi2);
  }
  write_field_raw(cfg.output_dir + "/c_final.raw", c);
  if (!skip_png && !scale_io_off) {
    write_c_png((cfg.output_dir + "/c_late.png").c_str(), c, clo);
  }

  double min_phi = phi_eff_two(phi1(1, 1, 1), phi2(1, 1, 1));
  double max_phi = min_phi;
  double min_c = c(1, 1, 1);
  double max_c = min_c;
  double sum_phi = 0.0;
  double sum_c = 0.0;
  for (int kk = 1; kk <= Nz; ++kk) {
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        const double ph =
            (n_grains == 1) ? phi1(i, j, kk) : phi_eff_two(phi1(i, j, kk), phi2(i, j, kk));
        min_phi = std::min(min_phi, ph);
        max_phi = std::max(max_phi, ph);
        min_c = std::min(min_c, c(i, j, kk));
        max_c = std::max(max_c, c(i, j, kk));
        sum_phi += ph;
        sum_c += c(i, j, kk);
      }
    }
  }

  RunResult out;
  out.wall_loop_s = t_loop1 - t_loop0;
  out.halo_s = t_halo;
  out.kernel_s = t_kern;
  const int n_timed = cfg.timed_steps > 0 ? std::min(cfg.timed_steps, std::max(0, n_done - warmup))
                                          : n_done;
  out.n_timed = n_timed;
  out.time_per_step_s = n_timed > 0
                            ? ((cfg.timed_steps > 0 ? (t_timed1 - t_timed0) : out.wall_loop_s) /
                               static_cast<double>(n_timed))
                            : 0.0;
  out.nthreads = nthr;
  out.nproc = 1;
  out.mass0 = mass0;
  out.mass1 = interior_sum(c) * dV;
  out.min_phi = min_phi;
  out.max_phi = max_phi;
  out.min_c = min_c;
  out.max_c = max_c;
  out.x_tip = interpolate_leading_tip(phi1, dx, win.shift_cells);
  out.sum_phi = sum_phi;
  out.sum_c = sum_c;
  out.n_steps_done = n_done;
  out.window_shift_cells = win.shift_cells;
  out.hit_right = hit_right;
  out.hit_far_c = hit_far_c;
  out.blew_up = blew_up;
  out.abort_reason = abort_reason;
  {
    std::size_t nbytes = pad_bytes(phi1) + pad_bytes(phi2) + pad_bytes(c) + pad_bytes(dphi1) +
                         pad_bytes(dphi2) + pad_bytes(dc);
    int nfields = 6; // allocated bricks (phi1/2, c, dphi1/2, dc)
    if (use_glasner) {
      nbytes += pad_bytes(psi1) + pad_bytes(psi2);
      nfields += 2;
    }
    if (store_eu) {
      nbytes += pad_bytes(eu_f) + pad_bytes(u_f);
      nfields += 2;
    }
    if (store_aux) {
      nbytes += pad_bytes(jx) + pad_bytes(jy);
      nfields += 2;
      if (dim3) {
        nbytes += pad_bytes(jz);
        nfields += 1;
      }
    }
    if (use_iso) {
      nbytes += pad_bytes(a_diff_f) + pad_bytes(a_at_f) + pad_bytes(beta_at_f);
      nfields += 3;
      if (!store_eu) {
        nbytes += pad_bytes(u_f);
        nfields += 1;
      }
    }
    out.perf.ghost_s = t_ghost;
    out.perf.eu_s = t_eu;
    out.perf.flux_s = t_flux;
    out.perf.grain_s = t_grain;
    out.perf.euler_s = t_euler;
    out.perf.solute_s = t_solute;
    out.perf.reduce_s = t_reduce;
    out.perf.io_s = t_io;
    out.perf.store_eu = store_eu ? 1 : 0;
    out.perf.store_aux = store_aux ? 1 : 0;
    out.perf.n_persistent_fields = nfields;
    out.perf.alloc_bytes = nbytes;
    out.perf.bytes_per_cell = nfields * sizeof(double);
  }
  {
    std::ofstream meta(cfg.output_dir + "/meta.txt", std::ios::app);
    meta << std::setprecision(16);
    meta << "n_steps_done " << n_done << "\n";
    meta << "t_stop " << (dt * static_cast<double>(n_done)) << "\n";
    meta << "hit_right " << (hit_right ? 1 : 0) << "\n";
    meta << "hit_far_c " << (hit_far_c ? 1 : 0) << "\n";
    meta << "blew_up " << (blew_up ? 1 : 0) << "\n";
    if (!abort_reason.empty()) {
      meta << "abort_reason " << abort_reason << "\n";
    }
    meta << "x_tip " << out.x_tip << "\n";
    meta << "window_shift_cells " << win.shift_cells << "\n";
    meta << "time_per_step_s " << out.time_per_step_s << "\n";
    meta << "halo_s " << out.halo_s << "\n";
    meta << "kernel_s " << out.kernel_s << "\n";
    meta << "sum_phi " << out.sum_phi << "\n";
    meta << "sum_c " << out.sum_c << "\n";
    meta << "store_eu " << (store_eu ? 1 : 0) << "\n";
    meta << "store_aux " << (store_aux ? 1 : 0) << "\n";
    meta << "stored_fields phi[,psi] c dphi dc_scratch"
         << (store_eu ? " eu u" : "") << (store_aux ? " jx jy" : "")
         << (use_iso ? " a_diff a_at beta_at" : "") << "\n";
    meta << "recomputed_fields "
         << (store_aux ? "gradients\n" : "aniso_fluxes gradients\n");
  }
  return out;
}

} // namespace alloy_pf_directional::engine
