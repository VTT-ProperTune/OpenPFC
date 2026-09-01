// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_pf_karma2001_benchmark_engine.cpp
 * @brief Explicit Euler FD for Karma (2001) present model, 2D or 3D, Neumann BCs.
 *
 * Cubic a_s(n), a_k(n); τ from Pinomaa (2020) eq. (7) at W_s, β_k. Optional Glasner ψ.
 */

#include <alloy_pf_karma2001_benchmark/engine.hpp>
#include <alloy_pf_karma2001_benchmark/isotropic_fd.hpp>
#include <alloy_pf_karma2001_benchmark/noise.hpp>

#include <openpfc/frontend/io/png_writer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <omp.h>

namespace {

struct PadField {
  int Nx = 0;
  int Ny = 0;
  int Nz = 1;
  int sx = 0;
  int sxy = 0;
  std::vector<double> a;

  PadField() = default;
  PadField(int nx, int ny, int nz = 1)
      : Nx(nx), Ny(ny), Nz(nz), sx(nx + 2), sxy((nx + 2) * (ny + 2)),
        a(static_cast<std::size_t>(sx) * static_cast<std::size_t>(ny + 2) *
              static_cast<std::size_t>(nz + 2),
          0.0) {}

  double &operator()(int i, int j, int k) noexcept {
    return a[static_cast<std::size_t>(i + j * sx + k * sxy)];
  }
  double operator()(int i, int j, int k) const noexcept {
    return a[static_cast<std::size_t>(i + j * sx + k * sxy)];
  }
  double &operator()(int i, int j) noexcept { return (*this)(i, j, 1); }
  double operator()(int i, int j) const noexcept { return (*this)(i, j, 1); }
};

void fill_neumann(PadField &f) {
  const int Nx = f.Nx;
  const int Ny = f.Ny;
  const int Nz = f.Nz;
  for (int k = 0; k <= Nz + 1; ++k) {
    for (int j = 1; j <= Ny; ++j) {
      f(0, j, k) = f(1, j, k);
      f(Nx + 1, j, k) = f(Nx, j, k);
    }
  }
  for (int k = 0; k <= Nz + 1; ++k) {
    for (int i = 0; i <= Nx + 1; ++i) {
      f(i, 0, k) = f(i, 1, k);
      f(i, Ny + 1, k) = f(i, Ny, k);
    }
  }
  for (int j = 0; j <= Ny + 1; ++j) {
    for (int i = 0; i <= Nx + 1; ++i) {
      f(i, j, 0) = f(i, j, 1);
      f(i, j, Nz + 1) = f(i, j, Nz);
    }
  }
}

inline double x_of(int i, double dx) noexcept { return (static_cast<double>(i) - 0.5) * dx; }
inline double y_of(int j, double dx) noexcept { return (static_cast<double>(j) - 0.5) * dx; }
inline double z_of(int k, double dx) noexcept { return (static_cast<double>(k) - 0.5) * dx; }

alloy_pf_karma2001_benchmark::Vec3 growth_ray(const alloy_pf_karma2001_benchmark::Mat3 &R) {
  alloy_pf_karma2001_benchmark::Vec3 best{1.0, 0.0, 0.0};
  double best_s = -1.0e9;
  for (int ax = 0; ax < 3; ++ax) {
    for (int sgn = 0; sgn < 2; ++sgn) {
      const double s = (sgn == 0) ? 1.0 : -1.0;
      alloy_pf_karma2001_benchmark::Vec3 w{s * R[0][ax], s * R[1][ax], s * R[2][ax]};
      if (w[0] < -1.0e-8 || w[1] < -1.0e-8 || w[2] < -1.0e-8) {
        continue;
      }
      const double sc = w[0] + w[1] + w[2];
      if (sc > best_s) {
        best_s = sc;
        best = w;
      }
    }
  }
  const double n = std::sqrt(best[0] * best[0] + best[1] * best[1] + best[2] * best[2]);
  if (n > 0.0) {
    best[0] /= n;
    best[1] /= n;
    best[2] /= n;
  }
  return best;
}

double sample_field(const PadField &f, double x, double y, double z, double dx) {
  const double fi = x / dx + 0.5;
  const double fj = y / dx + 0.5;
  const double fk = z / dx + 0.5;
  int i0 = std::max(0, std::min(f.Nx, static_cast<int>(std::floor(fi))));
  int j0 = std::max(0, std::min(f.Ny, static_cast<int>(std::floor(fj))));
  int k0 = std::max(0, std::min(f.Nz, static_cast<int>(std::floor(fk))));
  const int i1 = std::min(f.Nx + 1, i0 + 1);
  const int j1 = std::min(f.Ny + 1, j0 + 1);
  const int k1 = std::min(f.Nz + 1, k0 + 1);
  const double ax = std::min(1.0, std::max(0.0, fi - static_cast<double>(i0)));
  const double ay = std::min(1.0, std::max(0.0, fj - static_cast<double>(j0)));
  const double az = std::min(1.0, std::max(0.0, fk - static_cast<double>(k0)));
  const double v000 = f(i0, j0, k0);
  const double v100 = f(i1, j0, k0);
  const double v010 = f(i0, j1, k0);
  const double v110 = f(i1, j1, k0);
  const double v001 = f(i0, j0, k1);
  const double v101 = f(i1, j0, k1);
  const double v011 = f(i0, j1, k1);
  const double v111 = f(i1, j1, k1);
  const double v00 = (1.0 - az) * v000 + az * v001;
  const double v10 = (1.0 - az) * v100 + az * v101;
  const double v01 = (1.0 - az) * v010 + az * v011;
  const double v11 = (1.0 - az) * v110 + az * v111;
  return (1.0 - ax) * (1.0 - ay) * v00 + ax * (1.0 - ay) * v10 + (1.0 - ax) * ay * v01 +
         ax * ay * v11;
}

double quadratic_zero(double r0, double p0, double r1, double p1, double r2, double p2) {
  const double den = (r0 - r1) * (r0 - r2) * (r1 - r2);
  if (std::abs(den) < 1.0e-30) {
    const double d = p0 - p1;
    return (d > 0.0) ? (r0 + (p0 / d) * (r1 - r0)) : r0;
  }
  const double a = (p0 * (r1 - r2) + p1 * (r2 - r0) + p2 * (r0 - r1)) / den;
  const double b = (p0 * (r2 * r2 - r1 * r1) + p1 * (r0 * r0 - r2 * r2) + p2 * (r1 * r1 - r0 * r0)) /
                   den;
  const double c = (p0 * r1 * r2 * (r1 - r2) + p1 * r0 * r2 * (r2 - r0) + p2 * r0 * r1 * (r0 - r1)) /
                   den;
  if (std::abs(a) < 1.0e-18) {
    if (std::abs(b) < 1.0e-18) {
      return r1;
    }
    return -c / b;
  }
  const double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) {
    const double d = p0 - p1;
    return (d > 0.0) ? (r0 + (p0 / d) * (r1 - r0)) : r0;
  }
  const double s = std::sqrt(disc);
  const double q1 = (-b - s) / (2.0 * a);
  const double q2 = (-b + s) / (2.0 * a);
  const double lo = std::min(r0, r2);
  const double hi = std::max(r0, r2);
  const bool in1 = (q1 >= lo - 1.0e-12 && q1 <= hi + 1.0e-12);
  const bool in2 = (q2 >= lo - 1.0e-12 && q2 <= hi + 1.0e-12);
  if (in1 && !in2) {
    return q1;
  }
  if (in2 && !in1) {
    return q2;
  }
  if (in1 && in2) {
    return (std::abs(q1 - r1) < std::abs(q2 - r1)) ? q1 : q2;
  }
  const double d = p0 - p1;
  return (d > 0.0) ? (r0 + (p0 / d) * (r1 - r0)) : r0;
}

double interpolate_tip_ray(const PadField &phi, double dx, const alloy_pf_karma2001_benchmark::Vec3 &dir) {
  const double xmax = x_of(phi.Nx, dx);
  const double ymax = y_of(phi.Ny, dx);
  const double zmax = (phi.Nz > 1) ? z_of(phi.Nz, dx) : 0.0;
  double rmax = xmax + ymax + zmax;
  if (std::abs(dir[0]) > 1.0e-12) {
    rmax = std::min(rmax, xmax / std::max(dir[0], 1.0e-12));
  }
  if (std::abs(dir[1]) > 1.0e-12) {
    rmax = std::min(rmax, ymax / std::max(dir[1], 1.0e-12));
  }
  if (phi.Nz > 1 && std::abs(dir[2]) > 1.0e-12) {
    rmax = std::min(rmax, zmax / std::max(dir[2], 1.0e-12));
  }
  rmax = std::max(dx, rmax - dx);
  const double dr = 0.5 * dx;
  std::vector<double> rs;
  std::vector<double> ps;
  rs.reserve(static_cast<std::size_t>(rmax / dr) + 4);
  for (double r = 0.0; r <= rmax; r += dr) {
    const double z = (phi.Nz > 1) ? r * dir[2] : 0.0;
    rs.push_back(r);
    ps.push_back(sample_field(phi, r * dir[0], r * dir[1], z, dx));
  }
  for (std::size_t i = 1; i < ps.size(); ++i) {
    if (ps[i - 1] >= 0.0 && ps[i] < 0.0) {
      const std::size_t i0 = (i >= 2) ? i - 2 : 0;
      const std::size_t i1 = i - 1;
      const std::size_t i2 = (i + 1 < ps.size()) ? i + 1 : i;
      return quadratic_zero(rs[i0], ps[i0], rs[i1], ps[i1], rs[i2], ps[i2]);
    }
  }
  return (ps.back() >= 0.0) ? rs.back() : 0.0;
}

double ls_slope(const std::vector<double> &t, const std::vector<double> &r, int min_points,
                double min_dt) {
  const int ntot = static_cast<int>(t.size());
  if (ntot < 2) {
    return 0.0;
  }
  int i0 = ntot - 1;
  while (i0 > 0) {
    const int n = ntot - i0;
    const double span = t[static_cast<std::size_t>(ntot - 1)] - t[static_cast<std::size_t>(i0)];
    if (n >= min_points && span >= min_dt) {
      break;
    }
    --i0;
  }
  const int n = ntot - i0;
  if (n < 2) {
    return 0.0;
  }
  double st = 0.0;
  double sr = 0.0;
  double stt = 0.0;
  double str = 0.0;
  for (int i = i0; i < ntot; ++i) {
    st += t[static_cast<std::size_t>(i)];
    sr += r[static_cast<std::size_t>(i)];
    stt += t[static_cast<std::size_t>(i)] * t[static_cast<std::size_t>(i)];
    str += t[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
  }
  const double nd = static_cast<double>(n);
  const double den = nd * stt - st * st;
  if (std::abs(den) < 1.0e-30) {
    return 0.0;
  }
  return (nd * str - st * sr) / den;
}

double fit_tip_radius_oriented(const PadField &phi, double dx, const alloy_pf_karma2001_benchmark::Vec3 &dir,
                              double r_tip, double W0) {
  const double s_max = 8.0 * W0;
  double num = 0.0;
  double den = 0.0;
  int count = 0;
  auto consider = [&](double x, double y, double z) {
    const double xp = x * dir[0] + y * dir[1] + z * dir[2];
    const double s = r_tip - xp;
    if (s <= 0.2 * dx || s > s_max) {
      return;
    }
    const double y2 = (x - xp * dir[0]) * (x - xp * dir[0]) + (y - xp * dir[1]) * (y - xp * dir[1]) +
                      (z - xp * dir[2]) * (z - xp * dir[2]);
    num += y2 * s;
    den += 2.0 * s * s;
    ++count;
  };
  const int k1 = (phi.Nz > 1) ? phi.Nz : 1;
  for (int k = 1; k <= k1; ++k) {
    const double z = (phi.Nz > 1) ? z_of(k, dx) : 0.0;
    for (int j = 1; j <= phi.Ny; ++j) {
      for (int i = 1; i < phi.Nx; ++i) {
        const double p0 = phi(i, j, k);
        const double p1 = phi(i + 1, j, k);
        if (p0 >= 0.0 && p1 < 0.0) {
          const double denp = p0 - p1;
          const double frac = (denp > 0.0) ? (p0 / denp) : 0.0;
          consider(x_of(i, dx) + frac * dx, y_of(j, dx), z);
        }
      }
    }
  }
  if (count < 4 || den < 1.0e-18) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return num / den;
}

double interior_sum(const PadField &f) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) collapse(3) schedule(static)
  for (int k = 1; k <= f.Nz; ++k) {
    for (int j = 1; j <= f.Ny; ++j) {
      for (int i = 1; i <= f.Nx; ++i) {
        s += f(i, j, k);
      }
    }
  }
  return s;
}

/** Mean concentration on the far Neumann faces (x = L and y = L). */
double mean_far_wall_c(const PadField &c) {
  double s = 0.0;
  int n = 0;
  for (int k = 1; k <= c.Nz; ++k) {
    for (int j = 1; j <= c.Ny; ++j) {
      s += c(c.Nx, j, k);
      ++n;
    }
    for (int i = 1; i <= c.Nx; ++i) {
      s += c(i, c.Ny, k);
      ++n;
    }
  }
  return (n > 0) ? s / static_cast<double>(n) : 0.0;
}

struct BcHit {
  const char *reason = nullptr;
  double value = 0.0;
};

/**
 * Origin faces are the quarter-seed symmetry planes. Stop on the far walls:
 * solid past stop_frac of L, diffuse φ on a far face, or solute departing c_∞.
 */
BcHit detect_bc_hit(const PadField &phi, const PadField &c, double dx, double stop_frac,
                    double c_inf, double Lx, double Ly, double Lz, bool dim3) {
  if (!(stop_frac > 0.0)) {
    return {};
  }
  const double c_tol = alloy_pf_karma2001_benchmark::kWallCRel * std::max(std::abs(c_inf), 1.0e-12);
  auto face = [&](double ph, double cc) -> BcHit {
    if (ph > alloy_pf_karma2001_benchmark::kWallPhiLiq) {
      return {"wall_phi", ph};
    }
    if (std::abs(cc - c_inf) > c_tol) {
      return {"wall_c", cc};
    }
    return {};
  };
  for (int k = 1; k <= phi.Nz; ++k) {
    for (int j = 1; j <= phi.Ny; ++j) {
      if (const auto h = face(phi(phi.Nx, j, k), c(c.Nx, j, k)); h.reason) {
        return h;
      }
    }
    for (int i = 1; i <= phi.Nx; ++i) {
      if (const auto h = face(phi(i, phi.Ny, k), c(i, c.Ny, k)); h.reason) {
        return h;
      }
    }
  }
  if (dim3) {
    for (int j = 1; j <= phi.Ny; ++j) {
      for (int i = 1; i <= phi.Nx; ++i) {
        if (const auto h = face(phi(i, j, phi.Nz), c(i, j, c.Nz)); h.reason) {
          return h;
        }
      }
    }
  }
  const double x_lim = stop_frac * Lx;
  const double y_lim = stop_frac * Ly;
  const double z_lim = stop_frac * Lz;
  for (int k = 1; k <= phi.Nz; ++k) {
    const double z = dim3 ? z_of(k, dx) : 0.0;
    for (int j = 1; j <= phi.Ny; ++j) {
      const double y = y_of(j, dx);
      for (int i = 1; i <= phi.Nx; ++i) {
        if (phi(i, j, k) < 0.0) {
          continue;
        }
        const double x = x_of(i, dx);
        if (x > x_lim || y > y_lim || (dim3 && z > z_lim)) {
          const double fx = x / Lx;
          const double fy = y / Ly;
          const double fz = dim3 ? z / Lz : 0.0;
          return {"wall", std::max(fx, std::max(fy, fz))};
        }
      }
    }
  }
  return {};
}

double liquid_fraction(const PadField &phi) {
  int nliq = 0;
  int n = 0;
  for (int k = 1; k <= phi.Nz; ++k) {
    for (int j = 1; j <= phi.Ny; ++j) {
      for (int i = 1; i <= phi.Nx; ++i) {
        ++n;
        if (phi(i, j, k) < 0.0) {
          ++nliq;
        }
      }
    }
  }
  return (n > 0) ? static_cast<double>(nliq) / static_cast<double>(n) : 0.0;
}

inline void lerp_zero(double x0, double y0, double v0, double x1, double y1, double v1, double &x,
                      double &y) {
  const double den = v1 - v0;
  const double s = (std::abs(den) > 1.0e-30) ? (-v0 / den) : 0.5;
  x = x0 + s * (x1 - x0);
  y = y0 + s * (y1 - y0);
}

/** Marching-squares φ=0 segments on the k=1 slice. Units: t in μs, x,y in μm. */
void append_zero_contour(std::ofstream &os, const PadField &phi, double dx, double t_us, int id) {
  const int k = 1;
  auto emit = [&](double xa, double ya, double xb, double yb) {
    os << t_us << ' ' << id << ' ' << (xa * 1.0e6) << ' ' << (ya * 1.0e6) << '\n';
    os << t_us << ' ' << id << ' ' << (xb * 1.0e6) << ' ' << (yb * 1.0e6) << '\n';
    os << '\n';
  };
  for (int j = 1; j < phi.Ny; ++j) {
    for (int i = 1; i < phi.Nx; ++i) {
      const double v00 = phi(i, j, k);
      const double v10 = phi(i + 1, j, k);
      const double v01 = phi(i, j + 1, k);
      const double v11 = phi(i + 1, j + 1, k);
      const int m = (v00 >= 0.0 ? 1 : 0) | (v10 >= 0.0 ? 2 : 0) | (v11 >= 0.0 ? 4 : 0) |
                    (v01 >= 0.0 ? 8 : 0);
      if (m == 0 || m == 15) {
        continue;
      }
      const double x0 = x_of(i, dx);
      const double y0 = y_of(j, dx);
      const double x1 = x_of(i + 1, dx);
      const double y1 = y_of(j + 1, dx);
      double xb = 0.0, yb = 0.0, xl = 0.0, yl = 0.0, xr = 0.0, yr = 0.0, xt = 0.0, yt = 0.0;
      lerp_zero(x0, y0, v00, x1, y0, v10, xb, yb);
      lerp_zero(x1, y0, v10, x1, y1, v11, xr, yr);
      lerp_zero(x0, y1, v01, x1, y1, v11, xt, yt);
      lerp_zero(x0, y0, v00, x0, y1, v01, xl, yl);
      switch (m) {
      case 1:
      case 14:
        emit(xb, yb, xl, yl);
        break;
      case 2:
      case 13:
        emit(xb, yb, xr, yr);
        break;
      case 3:
      case 12:
        emit(xl, yl, xr, yr);
        break;
      case 4:
      case 11:
        emit(xr, yr, xt, yt);
        break;
      case 6:
      case 9:
        emit(xb, yb, xt, yt);
        break;
      case 7:
      case 8:
        emit(xl, yl, xt, yt);
        break;
      case 5:
        emit(xb, yb, xl, yl);
        emit(xr, yr, xt, yt);
        break;
      case 10:
        emit(xb, yb, xr, yr);
        emit(xl, yl, xt, yt);
        break;
      default:
        break;
      }
    }
  }
}

void write_slice_png(const char *path, const PadField &phi, int k) {
  std::vector<double> img(static_cast<std::size_t>(phi.Nx) * static_cast<std::size_t>(phi.Ny));
  for (int j = 1; j <= phi.Ny; ++j) {
    for (int i = 1; i <= phi.Nx; ++i) {
      img[static_cast<std::size_t>(i - 1 + (j - 1) * phi.Nx)] = phi(i, j, k);
    }
  }
  pfc::io::write_png_grayscale_from_doubles(path, phi.Nx, phi.Ny, img.data(), -1.0, 1.0);
}

void write_ray_profile(const std::string &path, const PadField &phi, const PadField &c,
                       const alloy_pf_karma2001_benchmark::Physics &phys, const alloy_pf_karma2001_benchmark::Vec3 &dir) {
  std::ofstream os(path);
  os << std::setprecision(16);
  os << "# r r_over_d0 r_um phi c c_over_cl0 c_atpct\n";
  const double xmax = x_of(phi.Nx, phys.dx);
  const double ymax = y_of(phi.Ny, phys.dx);
  const double zmax = (phi.Nz > 1) ? z_of(phi.Nz, phys.dx) : 0.0;
  double rmax = xmax;
  if (std::abs(dir[0]) > 1.0e-12) {
    rmax = std::min(rmax, xmax / std::max(dir[0], 1.0e-12));
  }
  if (std::abs(dir[1]) > 1.0e-12) {
    rmax = std::min(rmax, ymax / std::max(dir[1], 1.0e-12));
  }
  if (phi.Nz > 1 && std::abs(dir[2]) > 1.0e-12) {
    rmax = std::min(rmax, zmax / std::max(dir[2], 1.0e-12));
  }
  rmax = std::max(phys.dx, rmax - phys.dx);
  const double dr = 0.5 * phys.dx;
  for (double r = 0.5 * phys.dx; r <= rmax; r += dr) {
    const double z = (phi.Nz > 1) ? r * dir[2] : 0.0;
    const double ph = sample_field(phi, r * dir[0], r * dir[1], z, phys.dx);
    const double cv = sample_field(c, r * dir[0], r * dir[1], z, phys.dx);
    os << r << ' ' << (r / phys.d0) << ' ' << (r * 1.0e6) << ' ' << ph << ' ' << cv << ' '
       << (cv / phys.cl0) << ' ' << (cv / phys.cl0 * phys.clo_phys) << '\n';
  }
}

void write_axis_profile(const std::string &path, const PadField &phi, const PadField &c,
                        const alloy_pf_karma2001_benchmark::Physics &phys) {
  std::ofstream os(path);
  os << std::setprecision(16);
  os << "# x x_over_d0 phi c c_over_cl0\n";
  const int j = 1;
  const int kk = 1;
  for (int i = 1; i <= phi.Nx; ++i) {
    const double x = x_of(i, phys.dx);
    os << x << ' ' << (x / phys.d0) << ' ' << phi(i, j, kk) << ' ' << c(i, j, kk) << ' '
       << (c(i, j, kk) / phys.cl0) << '\n';
  }
}

void sample_k_eff(const PadField &phi, const PadField &c, double dx, const alloy_pf_karma2001_benchmark::Vec3 &dir,
                  double &c_s, double &c_l) {
  c_s = std::numeric_limits<double>::quiet_NaN();
  c_l = std::numeric_limits<double>::quiet_NaN();
  const double xmax = x_of(phi.Nx, dx);
  const double ymax = y_of(phi.Ny, dx);
  const double zmax = (phi.Nz > 1) ? z_of(phi.Nz, dx) : 0.0;
  double rmax = xmax;
  if (std::abs(dir[0]) > 1.0e-12) {
    rmax = std::min(rmax, xmax / std::max(dir[0], 1.0e-12));
  }
  if (std::abs(dir[1]) > 1.0e-12) {
    rmax = std::min(rmax, ymax / std::max(dir[1], 1.0e-12));
  }
  if (phi.Nz > 1 && std::abs(dir[2]) > 1.0e-12) {
    rmax = std::min(rmax, zmax / std::max(dir[2], 1.0e-12));
  }
  rmax = std::max(dx, rmax - dx);
  const double dr = 0.5 * dx;
  for (double r = 0.5 * dx; r <= rmax; r += dr) {
    const double z = (phi.Nz > 1) ? r * dir[2] : 0.0;
    const double ph = sample_field(phi, r * dir[0], r * dir[1], z, dx);
    const double cv = sample_field(c, r * dir[0], r * dir[1], z, dx);
    if (ph > 0.9) {
      c_s = cv;
    } else if (ph < -0.9 && std::isfinite(c_s)) {
      c_l = cv;
      return;
    }
  }
}

} // namespace

namespace alloy_pf_karma2001_benchmark::engine {

RunResult run(const RunConfig &cfg, bool skip_png, bool quiet) {
  const Physics phys = cfg.phys;
  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int Nz = std::max(1, cfg.Nz);
  const double dx = phys.dx;
  const double dt = phys.dt;
  const double inv_dx = 1.0 / dx;
  const double k = phys.k;
  const double cl0 = phys.cl0;
  const Mat3 R = rotation_of(phys);
  const Vec3 ray = growth_ray(R);
  const bool dim3 = (Nz > 1);

  if (cfg.num_threads > 0) {
    omp_set_num_threads(cfg.num_threads);
  }
  const int nthr = omp_get_max_threads();
  const bool use_team = (nthr > 1);

  PadField phi(Nx, Ny, Nz);
  PadField psi;
  PadField c(Nx, Ny, Nz);
  PadField u(Nx, Ny, Nz);
  PadField eu(Nx, Ny, Nz);
  PadField dphidt(Nx, Ny, Nz);
  PadField jx(Nx, Ny, Nz);
  PadField jy(Nx, Ny, Nz);
  PadField jz(Nx, Ny, Nz);
  const bool use_glasner = cfg.use_glasner;
  if (use_glasner) {
    psi = PadField(Nx, Ny, Nz);
  }

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
  for (int kk = 1; kk <= Nz; ++kk) {
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        const double x = x_of(i, dx);
        const double y = y_of(j, dx);
        const double z = dim3 ? z_of(kk, dx) : 0.0;
        const double r = std::sqrt(x * x + y * y + z * z);
        double ph = 0.0;
        if (use_glasner) {
          psi(i, j, kk) = -(r - phys.r_seed) / phys.W0;
          ph = phi_from_psi(psi(i, j, kk));
        } else {
          const double eta = (r - phys.r_seed) / (std::sqrt(2.0) * phys.W0);
          ph = -std::tanh(eta);
          if (!std::isfinite(ph)) {
            ph = (r < phys.r_seed) ? 1.0 : -1.0;
          }
        }
        phi(i, j, kk) = ph;
        const double den = std::max(denom_c(ph, k), 1.0e-12);
        c(i, j, kk) = 0.5 * cl0 * std::exp(phys.u_inf) * den;
      }
    }
  }
  fill_neumann(phi);
  fill_neumann(c);
  if (use_glasner) {
    fill_neumann(psi);
  }

  const double dV = dim3 ? (dx * dx * dx) : (dx * dx);
  const int n_dim = dim3 ? 3 : 2;
  const double mass0 = interior_sum(c) * dV;

  std::ofstream hist(cfg.output_dir + "/tip_history.tsv");
  hist << std::setprecision(16);
  hist << "# t t_star t_us r_tip r_over_d0 r_um V V_star V_mps rho rho_um mass k_CGM k_eff "
          "dT_k_K dT_r_K dT_th_K c_wall c_wall_over liquid_frac tip_x tip_y c_s c_l dT_c_K "
          "dT_tip_K\n";

  {
    std::ofstream meta(cfg.output_dir + "/meta.txt");
    meta << std::setprecision(16);
    meta << "d0_over_W " << phys.d0_over_W << "\n";
    meta << "d0 " << phys.d0 << "\n";
    meta << "D " << phys.D << "\n";
    meta << "lambda " << phys.lambda << "\n";
    meta << "W0 " << phys.W0 << "\n";
    meta << "tau0 " << phys.tau0 << "\n";
    meta << "dx " << dx << "\n";
    meta << "dt " << dt << "\n";
    meta << "dt_over_tau " << alloy_pf_karma2001_benchmark::dt_over_tau_of(phys) << "\n";
    meta << "fourier " << (phys.D * dt / (dx * dx)) << "\n";
    meta << "eps_c " << phys.eps_c << "\n";
    meta << "k " << k << "\n";
    meta << "Omega " << phys.Omega << "\n";
    meta << "cl0 " << cl0 << "\n";
    meta << "c_inf " << phys.c_inf << "\n";
    meta << "u_inf " << phys.u_inf << "\n";
    meta << "a_at " << phys.a_at << "\n";
    meta << "A_trap " << phys.A_trap << "\n";
    meta << "a2 " << phys.a2 << "\n";
    meta << "alpha_drag " << phys.alpha_drag << "\n";
    meta << "VD_pf " << phys.VD_pf << "\n";
    meta << "beta0 " << phys.beta0 << "\n";
    meta << "eps_k " << phys.eps_k << "\n";
    meta << "Gamma " << phys.Gamma << "\n";
    meta << "mle " << phys.mle << "\n";
    meta << "clo_phys " << phys.clo_phys << "\n";
    meta << "dT_scale " << phys.dT_scale << "\n";
    meta << "W0_nm " << (phys.W0 * 1.0e9) << "\n";
    meta << "d0_nm " << (phys.d0 * 1.0e9) << "\n";
    meta << "r_seed " << phys.r_seed << "\n";
    meta << "Tdot " << phys.Tdot << "\n";
    meta << "t_decay " << phys.t_decay << "\n";
    meta << "dT_sat " << (phys.t_decay > 0.0 ? phys.Tdot * phys.t_decay : 0.0) << "\n";
    meta << "dT_gt " << phys.dT_gt << "\n";
    meta << "dT_extra " << phys.dT_extra << "\n";
    meta << "n_contour " << cfg.n_contour << "\n";
    meta << "phi1 " << phys.phi1 << "\n";
    meta << "Phi " << phys.Phi << "\n";
    meta << "phi2 " << phys.phi2 << "\n";
    meta << "theta0 " << phys.phi1 << "\n";
    meta << "use_glasner " << (use_glasner ? 1 : 0) << "\n";
    meta << "use_isotropic " << (cfg.use_isotropic ? 1 : 0) << "\n";
    meta << "Nx " << Nx << "\n";
    meta << "Ny " << Ny << "\n";
    meta << "Nz " << Nz << "\n";
    meta << "n_steps " << cfg.n_steps << "\n";
    meta << "L_over_d0 " << cfg.L_over_d0 << "\n";
    meta << "L " << (static_cast<double>(Nx) * dx) << "\n";
    meta << "stop_frac " << cfg.stop_frac << "\n";
    meta << "bc neumann_zero_gradient\n";
    meta << "a1 " << kA1 << "\n";
    meta << "a2 " << kA2 << "\n";
    meta << "noise_F0 " << cfg.noise_F0 << "\n";
    meta << "noise_seed " << cfg.noise_seed << "\n";
  }

  int filenum = 0;
  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(), filenum);
    std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    write_slice_png(path, phi, 1);
    ++filenum;
  }

  const int nprint_eff = quiet ? 0 : cfg.nprint;
  std::vector<double> t_hist;
  std::vector<double> r_hist;
  t_hist.push_back(0.0);
  r_hist.push_back(interpolate_tip_ray(phi, dx, ray));

  std::ofstream contours;
  int next_contour = 1;
  if (cfg.n_contour > 0) {
    contours.open(cfg.output_dir + "/interface_contours.tsv");
    contours << std::setprecision(10);
    contours << "# t_us id x_um y_um\n";
    append_zero_contour(contours, phi, dx, 0.0, 0);
  }

  const double t_loop0 = omp_get_wtime();
  const double Lx = static_cast<double>(Nx) * dx;
  const double Ly = static_cast<double>(Ny) * dx;
  const double Lz = static_cast<double>(Nz) * dx;
  const char *stop_reason = "time";

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {
    const double t_mid = dt * (static_cast<double>(istep) - 0.5);
    const double therm = therm_drive(phys, t_mid);
    fill_neumann(phi);
    fill_neumann(c);
    if (use_glasner) {
      fill_neumann(psi);
    }
    const PadField &pf = use_glasner ? psi : phi;

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          const double ph = phi(i, j, kk);
          double ci = c(i, j, kk);
          if (ci < kCMin) {
            ci = kCMin;
            c(i, j, kk) = ci;
          }
          const double den = std::max(denom_c(ph, k), 1.0e-12);
          const double euv = (2.0 * ci / cl0) / den;
          eu(i, j, kk) = euv;
          u(i, j, kk) = std::log(std::max(euv, 1.0e-30));
        }
      }
    }
    fill_neumann(u);
    fill_neumann(eu);

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
          if (i == 0 || i == Nx) {
            jx(i, j, kk) = 0.0;
            continue;
          }
          const double gx = inv_dx * (pf(i + 1, j, kk) - pf(i, j, kk));
          const double gy =
              0.25 * inv_dx *
              (pf(i + 1, j + 1, kk) + pf(i, j + 1, kk) - pf(i + 1, j - 1, kk) - pf(i, j - 1, kk));
          const double gz =
              dim3 ? (0.25 * inv_dx *
                      (pf(i + 1, j, kk + 1) + pf(i, j, kk + 1) - pf(i + 1, j, kk - 1) -
                       pf(i, j, kk - 1)))
                   : 0.0;
          const auto an =
              cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.eps_k, phys.W0, R);
          jx(i, j, kk) = an.jx;
        }
      }
    }

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 0; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          if (j == 0 || j == Ny) {
            jy(i, j, kk) = 0.0;
            continue;
          }
          const double gy = inv_dx * (pf(i, j + 1, kk) - pf(i, j, kk));
          const double gx =
              0.25 * inv_dx *
              (pf(i + 1, j + 1, kk) + pf(i + 1, j, kk) - pf(i - 1, j + 1, kk) - pf(i - 1, j, kk));
          const double gz =
              dim3 ? (0.25 * inv_dx *
                      (pf(i, j + 1, kk + 1) + pf(i, j, kk + 1) - pf(i, j + 1, kk - 1) -
                       pf(i, j, kk - 1)))
                   : 0.0;
          const auto an =
              cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.eps_k, phys.W0, R);
          jy(i, j, kk) = an.jy;
        }
      }
    }

    if (dim3) {
#pragma omp parallel for collapse(3) schedule(static) if (use_team)
      for (int kk = 0; kk <= Nz; ++kk) {
        for (int j = 1; j <= Ny; ++j) {
          for (int i = 1; i <= Nx; ++i) {
            if (kk == 0 || kk == Nz) {
              jz(i, j, kk) = 0.0;
              continue;
            }
            const double gz = inv_dx * (pf(i, j, kk + 1) - pf(i, j, kk));
            const double gx =
                0.25 * inv_dx *
                (pf(i + 1, j, kk + 1) + pf(i + 1, j, kk) - pf(i - 1, j, kk + 1) - pf(i - 1, j, kk));
            const double gy =
                0.25 * inv_dx *
                (pf(i, j + 1, kk + 1) + pf(i, j, kk + 1) - pf(i, j - 1, kk + 1) - pf(i, j - 1, kk));
            const auto an =
                cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.eps_k, phys.W0, R);
            jz(i, j, kk) = an.jz;
          }
        }
      }
    }

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          const double gx = 0.5 * inv_dx * (pf(i + 1, j, kk) - pf(i - 1, j, kk));
          const double gy = 0.5 * inv_dx * (pf(i, j + 1, kk) - pf(i, j - 1, kk));
          const double gz = dim3 ? (0.5 * inv_dx * (pf(i, j, kk + 1) - pf(i, j, kk - 1))) : 0.0;
          const auto an =
              cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.eps_k, phys.W0, R);
          const double W_s = phys.W0 * an.a_s;
          const double beta_k = phys.beta0 * an.a_k;
          const double tau = tau_aniso(phys, W_s, beta_k, eu(i, j, kk));
          double aniso =
              (jx(i, j, kk) - jx(i - 1, j, kk)) * inv_dx + (jy(i, j, kk) - jy(i, j - 1, kk)) * inv_dx;
          if (dim3) {
            aniso += (jz(i, j, kk) - jz(i, j, kk - 1)) * inv_dx;
          }
          double mag2 = gx * gx + gy * gy + gz * gz;
          if (cfg.use_isotropic) {
            const double L_iso = iso::laplacian_iso(pf, i, j, kk, dx, dim3);
            const double L_std = iso::laplacian_std(pf, i, j, kk, dx, dim3);
            mag2 = iso::grad2_iso(pf, i, j, kk, dx, dim3);
            aniso += W_s * W_s * (L_iso - L_std);
          }
          const double ph = phi(i, j, kk);
          if (use_glasner) {
            const double W = W_s;
            const double sqrt2 = std::sqrt(2.0);
            const double bulk = sqrt2 * ph * (1.0 - W * W * mag2) -
                                sqrt2 * (1.0 - ph * ph) * (phys.lambda / (1.0 - k)) *
                                    (eu(i, j, kk) - 1.0 - therm);
            dphidt(i, j, kk) = (aniso + bulk) / tau;
          } else {
            const double bulk = -f_prime(ph) -
                                (phys.lambda / (1.0 - k)) * g_prime(ph) *
                                    (eu(i, j, kk) - 1.0 - therm);
            dphidt(i, j, kk) = (aniso + bulk) / tau;
          }
          if (cfg.noise_F0 > 0.0) {
            const int knoise = (Nz == 1) ? 0 : kk;
            const double xi = gaussian_n01(cfg.noise_seed, istep, i, j, knoise, 0);
            const double nrate =
                fdt_phi_noise_rate(cfg.noise_F0, phys.W0, n_dim, tau, dt, dV, ph, xi);
            if (use_glasner) {
              dphidt(i, j, kk) += nrate / std::max(dphi_dpsi_from_phi(ph), 1.0e-12);
            } else {
              dphidt(i, j, kk) += nrate;
            }
          }
        }
      }
    }

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          if (use_glasner) {
            const double dpsidt = dphidt(i, j, kk);
            psi(i, j, kk) += dt * dpsidt;
            dphidt(i, j, kk) = dphi_dpsi_from_phi(phi(i, j, kk)) * dpsidt;
            phi(i, j, kk) = phi_from_psi(psi(i, j, kk));
          } else {
            phi(i, j, kk) += dt * dphidt(i, j, kk);
          }
        }
      }
    }
    fill_neumann(phi);
    fill_neumann(dphidt);
    if (use_glasner) {
      fill_neumann(psi);
    }

    if (cfg.use_isotropic) {
      PadField &a_diff = jx;
      PadField &a_at = jy;
#pragma omp parallel for collapse(3) schedule(static) if (use_team)
      for (int kk = 0; kk <= Nz + 1; ++kk) {
        for (int j = 0; j <= Ny + 1; ++j) {
          for (int i = 0; i <= Nx + 1; ++i) {
            a_diff(i, j, kk) = phys.D * c(i, j, kk) * q_of(phi(i, j, kk), k);
            const double pref = a_prime_trap(phi(i, j, kk), phys.a_at, phys.A_trap) * phys.W0 *
                                cl0 * (1.0 - k) * eu(i, j, kk) * dphidt(i, j, kk);
            if (use_glasner) {
              a_at(i, j, kk) = pref * phys.W0;
            } else {
              const double ph = phi(i, j, kk);
              const double om = std::max(1.0 - ph * ph, 1.0e-8);
              a_at(i, j, kk) = pref * std::sqrt(2.0) * phys.W0 / om;
            }
          }
        }
      }
      const PadField &beta_at = use_glasner ? psi : phi;
#pragma omp parallel for collapse(3) schedule(static) if (use_team)
      for (int kk = 1; kk <= Nz; ++kk) {
        for (int j = 1; j <= Ny; ++j) {
          for (int i = 1; i <= Nx; ++i) {
            const double Dd = iso::div_alpha_grad(a_diff, u, i, j, kk, dx, dim3);
            const double Dat = iso::div_alpha_grad(a_at, beta_at, i, j, kk, dx, dim3);
            c(i, j, kk) += dt * (Dd + Dat);
          }
        }
      }
    } else {
#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
          if (i == 0 || i == Nx) {
            jx(i, j, kk) = 0.0;
            continue;
          }
          const double px = inv_dx * (phi(i + 1, j, kk) - phi(i, j, kk));
          const double py =
              0.25 * inv_dx *
              (phi(i + 1, j + 1, kk) + phi(i, j + 1, kk) - phi(i + 1, j - 1, kk) -
               phi(i, j - 1, kk));
          const double pz =
              dim3 ? (0.25 * inv_dx *
                      (phi(i + 1, j, kk + 1) + phi(i, j, kk + 1) - phi(i + 1, j, kk - 1) -
                       phi(i, j, kk - 1)))
                   : 0.0;
          const double mag = std::sqrt(px * px + py * py + pz * pz + kGradEps * kGradEps);
          const double nx = px / mag;
          const double phf = 0.5 * (phi(i, j, kk) + phi(i + 1, j, kk));
          const double cf = 0.5 * (c(i, j, kk) + c(i + 1, j, kk));
          const double euf = 0.5 * (eu(i, j, kk) + eu(i + 1, j, kk));
          const double dtf = 0.5 * (dphidt(i, j, kk) + dphidt(i + 1, j, kk));
          const double ux = inv_dx * (u(i + 1, j, kk) - u(i, j, kk));
          const double qf = q_of(phf, k);
          jx(i, j, kk) = -phys.D * cf * qf * ux -
                         a_prime_trap(phf, phys.a_at, phys.A_trap) * phys.W0 * cl0 * (1.0 - k) *
                             euf * dtf * nx;
        }
      }
    }

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 0; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          if (j == 0 || j == Ny) {
            jy(i, j, kk) = 0.0;
            continue;
          }
          const double py = inv_dx * (phi(i, j + 1, kk) - phi(i, j, kk));
          const double px =
              0.25 * inv_dx *
              (phi(i + 1, j + 1, kk) + phi(i + 1, j, kk) - phi(i - 1, j + 1, kk) -
               phi(i - 1, j, kk));
          const double pz =
              dim3 ? (0.25 * inv_dx *
                      (phi(i, j + 1, kk + 1) + phi(i, j, kk + 1) - phi(i, j + 1, kk - 1) -
                       phi(i, j, kk - 1)))
                   : 0.0;
          const double mag = std::sqrt(px * px + py * py + pz * pz + kGradEps * kGradEps);
          const double ny = py / mag;
          const double phf = 0.5 * (phi(i, j, kk) + phi(i, j + 1, kk));
          const double cf = 0.5 * (c(i, j, kk) + c(i, j + 1, kk));
          const double euf = 0.5 * (eu(i, j, kk) + eu(i, j + 1, kk));
          const double dtf = 0.5 * (dphidt(i, j, kk) + dphidt(i, j + 1, kk));
          const double uy = inv_dx * (u(i, j + 1, kk) - u(i, j, kk));
          const double qf = q_of(phf, k);
          jy(i, j, kk) = -phys.D * cf * qf * uy -
                         a_prime_trap(phf, phys.a_at, phys.A_trap) * phys.W0 * cl0 * (1.0 - k) *
                             euf * dtf * ny;
        }
      }
    }

    if (dim3) {
#pragma omp parallel for collapse(3) schedule(static) if (use_team)
      for (int kk = 0; kk <= Nz; ++kk) {
        for (int j = 1; j <= Ny; ++j) {
          for (int i = 1; i <= Nx; ++i) {
            if (kk == 0 || kk == Nz) {
              jz(i, j, kk) = 0.0;
              continue;
            }
            const double pz = inv_dx * (phi(i, j, kk + 1) - phi(i, j, kk));
            const double px =
                0.25 * inv_dx *
                (phi(i + 1, j, kk + 1) + phi(i + 1, j, kk) - phi(i - 1, j, kk + 1) -
                 phi(i - 1, j, kk));
            const double py =
                0.25 * inv_dx *
                (phi(i, j + 1, kk + 1) + phi(i, j, kk + 1) - phi(i, j - 1, kk + 1) -
                 phi(i, j - 1, kk));
            const double mag = std::sqrt(px * px + py * py + pz * pz + kGradEps * kGradEps);
            const double nzv = pz / mag;
            const double phf = 0.5 * (phi(i, j, kk) + phi(i, j, kk + 1));
            const double cf = 0.5 * (c(i, j, kk) + c(i, j, kk + 1));
            const double euf = 0.5 * (eu(i, j, kk) + eu(i, j, kk + 1));
            const double dtf = 0.5 * (dphidt(i, j, kk) + dphidt(i, j, kk + 1));
            const double uz = inv_dx * (u(i, j, kk + 1) - u(i, j, kk));
            const double qf = q_of(phf, k);
            jz(i, j, kk) = -phys.D * cf * qf * uz -
                           a_prime_trap(phf, phys.a_at, phys.A_trap) * phys.W0 * cl0 * (1.0 - k) *
                               euf * dtf * nzv;
          }
        }
      }
    }

#pragma omp parallel for collapse(3) schedule(static) if (use_team)
    for (int kk = 1; kk <= Nz; ++kk) {
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          double divj =
              (jx(i, j, kk) - jx(i - 1, j, kk)) * inv_dx + (jy(i, j, kk) - jy(i, j - 1, kk)) * inv_dx;
          if (dim3) {
            divj += (jz(i, j, kk) - jz(i, j, kk - 1)) * inv_dx;
          }
          c(i, j, kk) -= dt * divj;
        }
      }
    }
    } // !use_isotropic

    if (nprint_eff > 0 && istep % nprint_eff == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    const auto hit = detect_bc_hit(phi, c, dx, cfg.stop_frac, phys.c_inf, Lx, Ly, Lz, dim3);
    if (cfg.n_hist > 0 && (istep % cfg.n_hist == 0 || istep == cfg.n_steps || hit.reason)) {
      const double t = dt * static_cast<double>(istep);
      const double t_star = t * phys.D / (phys.d0 * phys.d0);
      const double r_tip = interpolate_tip_ray(phi, dx, ray);
      t_hist.push_back(t);
      r_hist.push_back(r_tip);
      const double min_dt_v = 50.0 * phys.d0 * phys.d0 / phys.D;
      const double V = ls_slope(t_hist, r_hist, 10, min_dt_v);
      const double V_star = V * phys.d0 / phys.D;
      const double rho = fit_tip_radius_oriented(phi, dx, ray, r_tip, phys.W0);
      const double mass = interior_sum(c) * dV;
      double c_s = 0.0;
      double c_l = 0.0;
      sample_k_eff(phi, c, dx, ray, c_s, c_l);
      const double k_eff = (c_l > 0.0 && std::isfinite(c_s) && std::isfinite(c_l)) ? (c_s / c_l)
                                                                                  : k;
      const double k_mod = k_cgm(k, V, phys.VD_pf);
      const double dT_k = phys.dT_scale * phys.beta0 * V;
      const double dT_r = (rho > 0.0 && std::isfinite(rho)) ? (phys.Gamma / rho) : 0.0;
      const double dT_th = dT_thermal(phys, t);
      const double dT_c =
          (std::isfinite(c_l) && cl0 > 0.0)
              ? (std::abs(phys.mle) * phys.clo_phys * (c_l / cl0 - phys.c_inf / cl0))
              : (dT_th - dT_r - dT_k);
      const double dT_tip = dT_c + dT_r + dT_k;
      const double c_wall = mean_far_wall_c(c);
      const double c_wall_over = (cl0 > 0.0) ? (c_wall / cl0) : 0.0;
      const double liq = liquid_fraction(phi);
      const double tip_x = r_tip * ray[0];
      const double tip_y = r_tip * ray[1];
      hist << t << ' ' << t_star << ' ' << (t * 1.0e6) << ' ' << r_tip << ' ' << (r_tip / phys.d0)
           << ' ' << (r_tip * 1.0e6) << ' ' << V << ' ' << V_star << ' ' << V << ' ' << rho << ' '
           << (rho * 1.0e6) << ' ' << mass << ' ' << k_mod << ' ' << k_eff << ' ' << dT_k << ' '
           << dT_r << ' ' << dT_th << ' ' << c_wall << ' ' << c_wall_over << ' ' << liq << ' '
           << tip_x << ' ' << tip_y << ' ' << c_s << ' ' << c_l << ' ' << dT_c << ' ' << dT_tip
           << '\n';
      hist.flush();
    }

    if (hit.reason) {
      stop_reason = hit.reason;
      std::cout << "abort: " << hit.reason << " value=" << hit.value << " stop_frac="
                << cfg.stop_frac << " L=(" << Lx << ", " << Ly << ")\n";
      if (contours.is_open()) {
        append_zero_contour(contours, phi, dx, dt * static_cast<double>(istep) * 1.0e6,
                            next_contour);
      }
      break;
    }

    if (contours.is_open() && cfg.n_contour > 1 && next_contour < cfg.n_contour) {
      const int tgt = (next_contour * cfg.n_steps) / (cfg.n_contour - 1);
      if (istep >= tgt) {
        append_zero_contour(contours, phi, dx, dt * static_cast<double>(istep) * 1.0e6,
                            next_contour);
        ++next_contour;
      }
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(), filenum);
      std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file " << path << "\n";
      write_slice_png(path, phi, 1);
      ++filenum;
    }
  }

  const double t_loop1 = omp_get_wtime();
  hist.flush();
  {
    std::ofstream meta(cfg.output_dir + "/meta.txt", std::ios::app);
    meta << std::setprecision(16);
    meta << "stop_reason " << stop_reason << "\n";
    meta << "wall_loop_s " << (t_loop1 - t_loop0) << "\n";
    meta << "c_wall " << mean_far_wall_c(c) << "\n";
    meta << "liquid_frac " << liquid_fraction(phi) << "\n";
  }

  fill_neumann(phi);
  fill_neumann(c);
  write_axis_profile(cfg.output_dir + "/axis_profile.tsv", phi, c, phys);
  write_ray_profile(cfg.output_dir + "/ray_profile.tsv", phi, c, phys, ray);

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    std::cout << "saving final field to " << path << "\n";
    write_slice_png(path, phi, 1);
    if (dim3) {
      std::snprintf(path, sizeof(path), "%s/phi_final_xz.png", cfg.output_dir.c_str());
      std::vector<double> img(static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Nz));
      for (int kk = 1; kk <= Nz; ++kk) {
        for (int i = 1; i <= Nx; ++i) {
          img[static_cast<std::size_t>(i - 1 + (kk - 1) * Nx)] = phi(i, 1, kk);
        }
      }
      pfc::io::write_png_grayscale_from_doubles(path, Nx, Nz, img.data(), -1.0, 1.0);
    }
  }

  double min_phi = phi(1, 1, 1);
  double max_phi = phi(1, 1, 1);
  double min_c = c(1, 1, 1);
  double max_c = c(1, 1, 1);
  for (int kk = 1; kk <= Nz; ++kk) {
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx; ++i) {
        min_phi = std::min(min_phi, phi(i, j, kk));
        max_phi = std::max(max_phi, phi(i, j, kk));
        min_c = std::min(min_c, c(i, j, kk));
        max_c = std::max(max_c, c(i, j, kk));
      }
    }
  }

  RunResult out;
  out.Nx = Nx;
  out.Ny = Ny;
  out.Nz = Nz;
  out.wall_loop_s = t_loop1 - t_loop0;
  out.nthreads = nthr;
  out.mass0 = mass0;
  out.mass1 = interior_sum(c) * dV;
  out.x_tip = interpolate_tip_ray(phi, dx, ray);
  out.rho_tip = fit_tip_radius_oriented(phi, dx, ray, out.x_tip, phys.W0);
  out.min_phi = min_phi;
  out.max_phi = max_phi;
  out.min_c = min_c;
  out.max_c = max_c;
  out.phi_xy.resize(static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny));
  out.c_xy.resize(out.phi_xy.size());
  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      const std::size_t idx = static_cast<std::size_t>(i - 1 + (j - 1) * Nx);
      out.phi_xy[idx] = phi(i, j, 1);
      out.c_xy[idx] = c(i, j, 1);
    }
  }
  return out;
}

} // namespace alloy_pf_karma2001_benchmark::engine
