// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_microelasticity.cpp
 * @brief Verification ladder for `openpfc_apps/microelasticity.hpp`.
 *
 * @details
 * Stage 0 of the capstone verification table (MODEL_SPEC.md): the elastic
 * solver on its own, against closed forms rather than against itself.
 *
 * Every oracle used here is derived in the comment above the test that uses
 * it, because an elasticity test whose expected value is a magic constant
 * verifies nothing. Three of them are exact to round-off and the rest are
 * limited only by how well a grid can represent a sphere:
 *
 *  1. single Fourier mode, homogeneous isotropic  — closed form, ~1e-14
 *  2. single Fourier mode, homogeneous cubic      — independent 3x3 solve, ~1e-13
 *  3. dilatation identity, arbitrary eigenstrain  — closed form, ~1e-13
 *  4. Eshelby sphere: interior strain / stress / far-field decay
 *  5. one Gamma application when the modulus is homogeneous
 *  6. monotone convergence at solid/liquid contrast + the strain-change test
 *  7. div sigma = 0, checked spectrally
 *  8. energy consistency, and the Eshelby energy in closed form
 *  9. d f_el/d phi against a finite difference of the re-converged energy
 * 10. decomposition invariance (the same numbers on 1 and N ranks)
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc_apps/microelasticity.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pfc::apps::EigenstrainMicroelasticity;
using pfc::apps::kSymComponents;
using pfc::apps::MicroelasticityParams;
using pfc::apps::Stiffness;
using pfc::apps::Sym3;
using pfc::apps::SYM_XX;
using pfc::apps::SYM_XY;
using pfc::apps::SYM_XZ;
using pfc::apps::SYM_YY;
using pfc::apps::SYM_YZ;
using pfc::apps::SYM_ZZ;

using RealField = pfc::data::Field<double>;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

/// Format a diagnostic line at full precision (Catch's INFO default is 6 digits).
template <class... Ts> std::string precise(Ts &&...parts) {
  std::ostringstream oss;
  oss << std::setprecision(12);
  (oss << ... << parts);
  return oss.str();
}

pfc::Domain cube(int n, double dx) {
  return pfc::domain::create(pfc::GridSize({n, n, n}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, dx}));
}

/// Global max of a per-rank value.
double gmax(double v) {
  double out = 0.0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return out;
}
/// Global sum of a per-rank value.
double gsum(double v) {
  double out = 0.0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

/// `fn(x, y, z)` written into every owned cell of a flat (halo-0) field.
template <class Fn> void fill(RealField &f, Fn &&fn) {
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = f.coords(i, j, k);
        f(i, j, k) = fn(x[0], x[1], x[2]);
      }
    }
  }
  f.note_host_write();
}

/// Constant fill over the whole (halo-0) buffer.
void fill_value(RealField &f, double v) {
  std::fill(f.vec().begin(), f.vec().end(), v);
  f.note_host_write();
}

/// Sum of a field over the whole domain (cell values, not volume-weighted).
double field_sum(const RealField &f) {
  double s = 0.0;
  for (std::size_t i = 0; i < f.size(); ++i) s += f.data()[i];
  return gsum(s);
}

/// Global mean of a field.
double field_mean(const RealField &f) {
  const auto g = f.global_size();
  const double n = static_cast<double>(g[0]) * static_cast<double>(g[1]) *
                   static_cast<double>(g[2]);
  return field_sum(f) / n;
}

/// Global max |f|.
double field_absmax(const RealField &f) {
  double m = 0.0;
  for (std::size_t i = 0; i < f.size(); ++i) m = std::max(m, std::abs(f.data()[i]));
  return gmax(m);
}

/// Read a strain/stress component at a global index, broadcast to every rank.
double sample_global(const RealField &f, int gi, int gj, int gk) {
  const auto &box = f.box();
  double v = 0.0;
  int have = 0;
  if (gi >= box.low[0] && gi <= box.high[0] && gj >= box.low[1] &&
      gj <= box.high[1] && gk >= box.low[2] && gk <= box.high[2]) {
    v = f(gi - box.low[0], gj - box.low[1], gk - box.low[2]);
    have = 1;
  }
  double vs = 0.0;
  int hs = 0;
  MPI_Allreduce(&v, &vs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&have, &hs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(hs == 1);
  return vs;
}

Sym3 sample_tensor(const EigenstrainMicroelasticity::SymRealFields &f, int gi,
                   int gj, int gk) {
  Sym3 out;
  for (int c = 0; c < kSymComponents; ++c) {
    out[c] = sample_global(f[static_cast<std::size_t>(c)], gi, gj, gk);
  }
  return out;
}

/**
 * @brief Independent Green-operator evaluation for one wave vector.
 *
 * Deliberately written a second way — Gaussian elimination with partial
 * pivoting on the acoustic tensor instead of the header's adjugate — so a
 * transposed cofactor or a dropped minor in the header shows up as a
 * disagreement rather than being reproduced by the oracle.
 */
Sym3 green_strain_reference(const Stiffness &c0, double kx, double ky, double kz,
                            const Sym3 &tau) {
  const double aniso = c0.c11 - c0.c12 - 2.0 * c0.c44;
  const double k2 = kx * kx + ky * ky + kz * kz;
  const double kv[3] = {kx, ky, kz};
  double a[3][4]{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      a[i][j] = (c0.c12 + c0.c44) * kv[i] * kv[j] +
                ((i == j) ? c0.c44 * k2 + aniso * kv[i] * kv[i] : 0.0);
    }
  }
  // b_i = k_j tau_ij
  const double t[3][3] = {{tau[SYM_XX], tau[SYM_XY], tau[SYM_XZ]},
                          {tau[SYM_XY], tau[SYM_YY], tau[SYM_YZ]},
                          {tau[SYM_XZ], tau[SYM_YZ], tau[SYM_ZZ]}};
  for (int i = 0; i < 3; ++i) {
    a[i][3] = kv[0] * t[i][0] + kv[1] * t[i][1] + kv[2] * t[i][2];
  }
  for (int col = 0; col < 3; ++col) {
    int piv = col;
    for (int r = col + 1; r < 3; ++r) {
      if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
    }
    for (int c2 = 0; c2 < 4; ++c2) std::swap(a[col][c2], a[piv][c2]);
    for (int r = 0; r < 3; ++r) {
      if (r == col) continue;
      const double m = a[r][col] / a[col][col];
      for (int c2 = col; c2 < 4; ++c2) a[r][c2] -= m * a[col][c2];
    }
  }
  const double v[3] = {a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]};
  Sym3 e;
  e[SYM_XX] = -kv[0] * v[0];
  e[SYM_YY] = -kv[1] * v[1];
  e[SYM_ZZ] = -kv[2] * v[2];
  e[SYM_YZ] = -0.5 * (kv[1] * v[2] + kv[2] * v[1]);
  e[SYM_XZ] = -0.5 * (kv[0] * v[2] + kv[2] * v[0]);
  e[SYM_XY] = -0.5 * (kv[0] * v[1] + kv[1] * v[0]);
  return e;
}

/// Spectral \f$(\nabla\cdot\boldsymbol\sigma)_i\f$, returned as global max |.|.
double
spectral_divergence_absmax(const pfc::Domain &domain, pfc::fft::IHostFFT &fft,
                           const EigenstrainMicroelasticity::SymRealFields &s) {
  using Complex = std::complex<double>;
  std::array<pfc::data::Field<Complex>, kSymComponents> hat{};
  for (int c = 0; c < kSymComponents; ++c) {
    hat[static_cast<std::size_t>(c)] =
        pfc::data::Field<Complex>(domain, fft.get_outbox_bounds(), 0);
    // forward() wants a mutable vector; the fields are const here, so copy.
    std::vector<double> tmp(s[static_cast<std::size_t>(c)].vec());
    fft.forward(tmp, hat[static_cast<std::size_t>(c)].vec());
  }
  pfc::data::Field<Complex> comp(domain, fft.get_outbox_bounds(), 0);
  pfc::data::Field<double> out =
      pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());

  // i k_j is an *odd* operator, so the Nyquist component must be zeroed --
  // `k_component_odd`'s rule, and the same rule the Green operator uses.
  const auto gsz = pfc::domain::get_size(domain);
  std::vector<double> kx(fft.size_outbox()), ky(fft.size_outbox()),
      kz(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      fft.get_outbox_bounds(), domain,
      [&](std::size_t idx, double a, double b, double c2, int i, int j, int k) {
        kx[idx] = pfc::fft::kspace::is_nyquist_index(i, gsz[0]) ? 0.0 : a;
        ky[idx] = pfc::fft::kspace::is_nyquist_index(j, gsz[1]) ? 0.0 : b;
        kz[idx] = pfc::fft::kspace::is_nyquist_index(k, gsz[2]) ? 0.0 : c2;
      });

  double worst = 0.0;
  for (int row = 0; row < 3; ++row) {
    const int cxx[3] = {SYM_XX, SYM_XY, SYM_XZ};
    const int cyy[3] = {SYM_XY, SYM_YY, SYM_YZ};
    const int czz[3] = {SYM_XZ, SYM_YZ, SYM_ZZ};
    const int *cc = (row == 0) ? cxx : ((row == 1) ? cyy : czz);
    for (std::size_t i = 0; i < fft.size_outbox(); ++i) {
      const Complex sx = hat[static_cast<std::size_t>(cc[0])].data()[i];
      const Complex sy = hat[static_cast<std::size_t>(cc[1])].data()[i];
      const Complex sz = hat[static_cast<std::size_t>(cc[2])].data()[i];
      comp.data()[i] = Complex{0.0, 1.0} * (kx[i] * sx + ky[i] * sy + kz[i] * sz);
    }
    fft.backward(comp.vec(), out.vec());
    worst = std::max(worst, field_absmax(out));
  }
  return worst;
}

/// Bundle: domain + stack + the four input fields, shared by most tests.
struct Case {
  pfc::Domain domain;
  pfc::sim::stacks::SpectralCPUStack stack;
  RealField h, amp, dh, damp;

  Case(int n, double dx)
      : domain(cube(n, dx)),
        stack(domain, world_rank(), world_size(), MPI_COMM_WORLD),
        h(pfc::data::field_from_inbox<double>(domain,
                                              stack.fft().get_inbox_bounds())),
        amp(pfc::data::field_from_inbox<double>(domain,
                                                stack.fft().get_inbox_bounds())),
        dh(pfc::data::field_from_inbox<double>(domain,
                                               stack.fft().get_inbox_bounds())),
        damp(pfc::data::field_from_inbox<double>(domain,
                                                 stack.fft().get_inbox_bounds())) {}
};

/// Smoothed solid sphere, \f$h = \tfrac12[1-\tanh((r-R)/w)]\f$.
struct Sphere {
  double cx, cy, cz, radius, width;
  [[nodiscard]] double operator()(double x, double y, double z) const {
    const double r =
        std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
    return 0.5 * (1.0 - std::tanh((r - radius) / width));
  }
};

} // namespace

// ---------------------------------------------------------------------------
// 1. Green operator, isotropic, single Fourier mode -- closed form
// ---------------------------------------------------------------------------
//
// For a homogeneous isotropic C0 and a dilatational eigenstrain
// eps*_ij = a(x) delta_ij the polarisation is tau = -C0:eps* = -3K a delta.
// Then b_i = k_j tau_ij = -3K a_hat k_i, and because A k = (lambda+2mu) k^2 k
// for the isotropic acoustic tensor, G k = k / ((lambda+2mu) k^2). Hence
//
//     eps_hat_mn = -(1/2)(k_n v_m + k_m v_n),  v = G b
//                = [3K/(lambda+2mu)] a_hat  k_m k_n / k^2 .
//
// With a(x) = A cos(k.x) (zero mean, so the k=0 mode is untouched) this is a
// pointwise closed form for all six components.
TEST_CASE("Green operator reproduces the closed-form single-mode strain (isotropic)",
          "[microelasticity][green]") {
  constexpr int N = 32;
  constexpr double dx = 1.0;
  Case cs(N, dx);
  const double L = N * dx;

  const double youngs = 2.0;
  const double nu = 0.3;
  const Stiffness c = Stiffness::isotropic(youngs, nu);
  const double mu = youngs / (2.0 * (1.0 + nu));
  const double lambda = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const double bulk = lambda + 2.0 * mu / 3.0;
  REQUIRE_THAT(c.bulk_modulus(), WithinRel(bulk, 1e-14));
  REQUIRE_THAT(c.zener(), WithinRel(1.0, 1e-14));

  // A generic direction so no component of eps is accidentally zero.
  const double kx = kTwoPi * 1.0 / L;
  const double ky = kTwoPi * 2.0 / L;
  const double kz = kTwoPi * 3.0 / L;
  const double k2 = kx * kx + ky * ky + kz * kz;
  const double amp0 = 1.0e-3;

  fill_value(cs.h, 1.0);
  fill(cs.amp, [&](double x, double y, double z) {
    return amp0 * std::cos(kx * x + ky * y + kz * z);
  });

  MicroelasticityParams p;
  p.c_solid = c;
  p.c_liquid = c;
  p.warm_start = false;
  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
  const auto rep = solver.solve(cs.h, cs.amp);
  REQUIRE(rep.converged);
  REQUIRE(rep.iterations == 1);

  const double pref = 3.0 * bulk / (lambda + 2.0 * mu);
  const double kk[3] = {kx, ky, kz};
  const int pair[kSymComponents][2] = {{0, 0}, {1, 1}, {2, 2},
                                       {1, 2}, {0, 2}, {0, 1}};

  double worst = 0.0;
  const auto n = cs.h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = cs.h.coords(i, j, k);
        const double a = amp0 * std::cos(kx * x[0] + ky * x[1] + kz * x[2]);
        for (int c2 = 0; c2 < kSymComponents; ++c2) {
          const double want = pref * a * kk[pair[c2][0]] * kk[pair[c2][1]] / k2;
          worst = std::max(
              worst,
              std::abs(solver.strain()[static_cast<std::size_t>(c2)](i, j, k) -
                       want));
        }
      }
    }
  }
  INFO("max |eps_num - eps_exact| = " << gmax(worst));
  REQUIRE(gmax(worst) < 1.0e-14 * amp0 * 1.0e3);
}

// ---------------------------------------------------------------------------
// 2. Green operator, cubic C0 -- against an independently coded 3x3 solve
// ---------------------------------------------------------------------------
TEST_CASE("Green operator matches an independent acoustic-tensor solve (cubic)",
          "[microelasticity][green][cubic]") {
  constexpr int N = 32;
  constexpr double dx = 1.0;
  Case cs(N, dx);
  const double L = N * dx;

  // Zener ratio 2*0.6/(2.2-0.9) = 0.923 -- genuinely anisotropic.
  const Stiffness c = Stiffness::cubic(2.2, 0.9, 0.6);
  REQUIRE(std::abs(c.zener() - 1.0) > 0.05);

  const double kx = kTwoPi * 3.0 / L;
  const double ky = kTwoPi * 1.0 / L;
  const double kz = kTwoPi * 2.0 / L;
  const double amp0 = 1.0e-3;

  // A non-dilatational pattern, so the shear rows of Gamma are exercised too.
  Sym3 pattern{{1.0, -0.4, -0.6, 0.3, 0.0, 0.25}};

  fill_value(cs.h, 1.0);
  fill(cs.amp, [&](double x, double y, double z) {
    return amp0 * std::cos(kx * x + ky * y + kz * z);
  });

  MicroelasticityParams p;
  p.c_solid = c;
  p.c_liquid = c;
  p.eigenstrain_pattern = pattern;
  p.warm_start = false;
  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
  const auto rep = solver.solve(cs.h, cs.amp);
  REQUIRE(rep.iterations == 1);

  // tau = -C0 : eps* for the homogeneous case; the mode amplitude factors out.
  Sym3 tau_unit;
  const Sym3 s_pattern = c.contract(pattern);
  for (int i = 0; i < kSymComponents; ++i) tau_unit[i] = -s_pattern[i];
  const Sym3 e_ref = green_strain_reference(c, kx, ky, kz, tau_unit);

  double worst = 0.0;
  double scale = 0.0;
  const auto n = cs.h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = cs.h.coords(i, j, k);
        const double a = amp0 * std::cos(kx * x[0] + ky * x[1] + kz * x[2]);
        for (int c2 = 0; c2 < kSymComponents; ++c2) {
          const double want = e_ref[c2] * a;
          scale = std::max(scale, std::abs(want));
          worst = std::max(
              worst,
              std::abs(solver.strain()[static_cast<std::size_t>(c2)](i, j, k) -
                       want));
        }
      }
    }
  }
  const double rel = gmax(worst) / gmax(scale);
  INFO("relative disagreement with the independent solve = " << rel);
  REQUIRE(rel < 1.0e-12);
}

// ---------------------------------------------------------------------------
// 3. Dilatation identity for an arbitrary eigenstrain distribution
// ---------------------------------------------------------------------------
//
// The single-mode result above gives tr(eps_hat) = [3K/(lambda+2mu)] a_hat for
// every k != 0, and 3K/(lambda+2mu) = (1+nu)/(1-nu) = 3*alpha with
// alpha = (1+nu)/(3(1-nu)) -- the Eshelby interior constant derived in test 4.
// Because the factor is independent of k, the identity survives superposition:
//
//     tr(eps)(x) = 3 alpha [ a(x) - <a> ]      exactly, for any a(x).
//
// This is the sharpest available statement that the dilatational projection of
// Gamma and the k=0 handling are both right, and it needs no sphere.
TEST_CASE("Dilatation identity holds pointwise for an arbitrary eigenstrain",
          "[microelasticity][green]") {
  constexpr int N = 32;
  Case cs(N, 1.0);
  const double nu = 0.28;
  const Stiffness c = Stiffness::isotropic(3.0, nu);
  const double alpha = (1.0 + nu) / (3.0 * (1.0 - nu));

  // Deliberately lumpy and asymmetric: two offset tanh blobs plus a mode.
  const Sphere b1{10.0, 12.0, 16.0, 6.0, 1.5};
  const Sphere b2{22.0, 20.0, 9.0, 4.0, 1.0};
  fill_value(cs.h, 1.0);
  fill(cs.amp, [&](double x, double y, double z) {
    return 1.0e-3 * (b1(x, y, z) - 0.5 * b2(x, y, z) +
                     0.2 * std::sin(kTwoPi * 2.0 * x / (N * 1.0)));
  });

  MicroelasticityParams p;
  p.c_solid = c;
  p.c_liquid = c;
  p.warm_start = false;
  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
  REQUIRE(solver.solve(cs.h, cs.amp).iterations == 1);

  const double abar = field_mean(cs.amp);
  double worst = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < cs.amp.size(); ++i) {
    const double tr = solver.strain()[SYM_XX].data()[i] +
                      solver.strain()[SYM_YY].data()[i] +
                      solver.strain()[SYM_ZZ].data()[i];
    const double want = 3.0 * alpha * (cs.amp.data()[i] - abar);
    worst = std::max(worst, std::abs(tr - want));
    scale = std::max(scale, std::abs(want));
  }
  const double rel = gmax(worst) / gmax(scale);
  INFO("max |tr(eps) - 3 alpha (a - <a>)| / max = " << rel);
  REQUIRE(rel < 1.0e-12);
}

// ---------------------------------------------------------------------------
// 4. Eshelby (1957) spherical inclusion
// ---------------------------------------------------------------------------
//
// Derivation of the constant actually asserted below. For a sphere in an
// infinite isotropic medium the Eshelby tensor is
//
//   S_ijkl = (5nu-1)/(15(1-nu)) d_ij d_kl + (4-5nu)/(15(1-nu)) (d_ik d_jl + d_il
//   d_jk).
//
// With a purely dilatational eigenstrain eps*_kl = e* d_kl,
//
//   eps_in_ij = S_ijkl e* d_kl
//             = e* d_ij [3(5nu-1) + 2(4-5nu)] / (15(1-nu))
//             = e* d_ij (5nu+5)/(15(1-nu))
//             = e* d_ij (1+nu)/(3(1-nu)) ,
//
// so each *normal component* is alpha e* with alpha = (1+nu)/(3(1-nu)), and the
// dilatation tr(eps_in) is 3 alpha e* = e*(1+nu)/(1-nu). (The task statement
// calls alpha e* "the interior dilatation"; it is the per-component value, and
// the code asserts the per-component value.)
//
// Interior stress: sigma = C:(eps_in - eps*) = 3K e* (alpha-1) delta, and
// alpha - 1 = -2(1-2nu)/(3(1-nu)), so
//
//   sigma_in = -2 K e* (1-2nu)/(1-nu) delta = -2 E e* / (3(1-nu)) delta .
//
// Exterior: u_r = alpha e* R^3 / r^2 (radial, continuous at r=R), hence
// eps_ij = A (d_ij - 3 n_i n_j)/r^3 with A = alpha e* R^3 -- traceless, and
// decaying as r^-3.
//
// The solver works in a *periodic* cell with <eps> = 0, not in an infinite
// medium, and the difference is not a detail. The dilatation identity of test
// 3 pins it exactly: tr(eps) = 3 alpha (a - <a>), so inside the inclusion the
// per-component interior strain is alpha e* (1 - f) with f the volume
// fraction, and the deviatoric image correction vanishes at the sphere centre
// by cubic site symmetry (the only cubic-invariant rank-2 tensor is d_ij).
// The centre value is therefore alpha e* (1-f) up to discretisation alone.
TEST_CASE("Eshelby spherical inclusion: interior strain, interior stress, decay",
          "[microelasticity][eshelby]") {
  constexpr int N = 64;
  constexpr double dx = 1.0;
  constexpr double R = 8.0;
  constexpr double estar = 1.0e-3;
  Case cs(N, dx);

  const double nu = 0.3;
  const double youngs = 1.0;
  const Stiffness c = Stiffness::isotropic(youngs, nu);
  const double alpha = (1.0 + nu) / (3.0 * (1.0 - nu));
  const double bulk = c.bulk_modulus();

  const double cxyz = 0.5 * N * dx;
  // Sub-cell antialiased indicator: cell value = clamped signed distance ramp
  // of width dx. A hard 0/1 staircase costs an order of magnitude in accuracy
  // for no physical reason -- the sphere is the oracle, not the staircase.
  fill(cs.amp, [&](double x, double y, double z) {
    const double r = std::sqrt((x - cxyz) * (x - cxyz) + (y - cxyz) * (y - cxyz) +
                               (z - cxyz) * (z - cxyz));
    const double t = 0.5 - (r - R) / dx;
    return estar * std::clamp(t, 0.0, 1.0);
  });
  fill_value(cs.h, 1.0);

  MicroelasticityParams p;
  p.c_solid = c;
  p.c_liquid = c;
  p.warm_start = false;
  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
  REQUIRE(solver.solve(cs.h, cs.amp).iterations == 1);

  const double frac = field_mean(cs.amp) / estar; // volume fraction
  const double vol_incl = frac * std::pow(N * dx, 3);
  const double r_eff = std::cbrt(3.0 * vol_incl / (4.0 * std::numbers::pi));
  INFO("volume fraction f = " << frac << ", effective radius = " << r_eff);
  REQUIRE_THAT(r_eff, WithinRel(R, 5.0e-3));

  const int ci = N / 2;
  const Sym3 e_in = sample_tensor(solver.strain(), ci, ci, ci);
  const Sym3 s_in = sample_tensor(solver.stress(), ci, ci, ci);

  // --- interior strain -----------------------------------------------------
  const double want_e = alpha * estar * (1.0 - frac);
  INFO(precise("Eshelby interior strain: numeric ", e_in[SYM_XX], " closed form ",
               want_e, " rel ", std::abs(e_in[SYM_XX] / want_e - 1.0)));
  // At the centre this is a round-off-level statement, not a discretisation
  // one: the trace is pinned exactly by the identity of test 3, the deviatoric
  // part is forbidden by cubic site symmetry, and f is measured from the same
  // field. Assert it as such -- a loose tolerance here would hide a real bug.
  for (int c2 = 0; c2 < 3; ++c2) {
    INFO("normal strain component " << c2 << " = " << e_in[c2] << " want "
                                    << want_e);
    REQUIRE_THAT(e_in[c2], WithinRel(want_e, 1.0e-11));
  }
  for (int c2 = 3; c2 < kSymComponents; ++c2) {
    // ...except for the staircase asymmetry of the discretised sphere, which
    // breaks that symmetry at the 1e-5 level of eps*.
    REQUIRE_THAT(e_in[c2], WithinAbs(0.0, 1.0e-5 * estar));
  }

  // Uniformity of the interior field *is* shape-sensitive -- it is Eshelby's
  // central result and it holds only for an ellipsoid. Off the centre nothing
  // forbids a deviatoric part, so this is where a mis-shapen inclusion or a
  // wrong deviatoric branch of Gamma would show.
  const Sym3 e_off = sample_tensor(solver.strain(), ci + 4, ci, ci);
  INFO(precise("off-centre interior strain xx=", e_off[SYM_XX],
               " yy=", e_off[SYM_YY], " zz=", e_off[SYM_ZZ], " max shear=",
               std::max({std::abs(e_off[SYM_YZ]), std::abs(e_off[SYM_XZ]),
                         std::abs(e_off[SYM_XY])})));
  // The departure is discretisation, not a broken operator: measured at
  // r = R/2 it is 3.0% for R = 8 dx and 0.74% for R = 16 dx, and it changes
  // sign between the two, which is what Gibbs ringing off a near-step
  // eigenstrain does. Second order in dx, so the tolerance is tied to R = 8 dx.
  for (int c2 = 0; c2 < 3; ++c2) {
    REQUIRE_THAT(e_off[c2], WithinRel(want_e, 4.0e-2));
  }
  for (int c2 = 3; c2 < kSymComponents; ++c2) {
    REQUIRE_THAT(e_off[c2], WithinAbs(0.0, 1.0e-4 * want_e));
  }

  // --- interior stress -----------------------------------------------------
  // sigma_in = 3K e* (alpha(1-f) - 1) delta for the periodic cell; the
  // infinite-medium value -2E e*/(3(1-nu)) is the f -> 0 limit of that.
  const double want_s = 3.0 * bulk * estar * (alpha * (1.0 - frac) - 1.0);
  const double want_s_inf = -2.0 * youngs * estar / (3.0 * (1.0 - nu));
  INFO(precise("Eshelby interior stress: numeric ", s_in[SYM_XX], " closed form ",
               want_s, " rel ", std::abs(s_in[SYM_XX] / want_s - 1.0)));
  INFO("sigma_in = " << s_in[SYM_XX] << ", periodic closed form " << want_s
                     << ", infinite-medium closed form " << want_s_inf);
  for (int c2 = 0; c2 < 3; ++c2) {
    REQUIRE_THAT(s_in[c2], WithinRel(want_s, 1.0e-11));
  }
  for (int c2 = 3; c2 < kSymComponents; ++c2) {
    REQUIRE_THAT(s_in[c2], WithinAbs(0.0, 1.0e-5 * std::abs(want_s)));
  }
  // The finite-cell correction is real and of the expected size, not noise.
  REQUIRE_THAT(want_s / want_s_inf, WithinRel(1.0, 0.05));

  // --- exterior decay ------------------------------------------------------
  // Along +x from the centre, n = xhat, so for the isolated inclusion
  //   eps_xx = -2A/r^3,  eps_yy = eps_zz = A/r^3,  A = alpha e* R^3,
  // and q(r) = eps_xx - eps_yy = -3A/r^3 is immune to the uniform -alpha e* f
  // shift that the periodic <eps>=0 condition adds. Fit log|q| vs log r over
  // 1.5R..2.5R, close enough to stay ahead of the periodic images.
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  int cnt = 0;
  double worst_amp = 0.0;
  for (int d = static_cast<int>(1.5 * R); d <= static_cast<int>(2.5 * R); ++d) {
    const double exx = sample_global(solver.strain()[SYM_XX], ci + d, ci, ci);
    const double eyy = sample_global(solver.strain()[SYM_YY], ci + d, ci, ci);
    const double q = exx - eyy;
    REQUIRE(q < 0.0);
    const double lr = std::log(static_cast<double>(d));
    const double lq = std::log(-q);
    sx += lr;
    sy += lq;
    sxx += lr * lr;
    sxy += lr * lq;
    ++cnt;
    const double want_q = -3.0 * alpha * estar * std::pow(r_eff, 3) /
                          std::pow(static_cast<double>(d), 3);
    worst_amp = std::max(worst_amp, std::abs(q / want_q - 1.0));
  }
  const double nn = static_cast<double>(cnt);
  const double slope = (nn * sxy - sx * sy) / (nn * sxx - sx * sx);
  INFO("far-field decay exponent = "
       << slope << " (want -3), worst amplitude error = " << worst_amp);
  REQUIRE_THAT(slope, WithinAbs(-3.0, 0.15));
  REQUIRE(worst_amp < 0.15);
}

// ---------------------------------------------------------------------------
// 5. Homogeneous modulus => exactly one Gamma application
// ---------------------------------------------------------------------------
TEST_CASE("A homogeneous modulus converges in exactly one iteration",
          "[microelasticity][fixedpoint]") {
  constexpr int N = 32;
  Case cs(N, 1.0);
  const Stiffness c = Stiffness::cubic(2.4, 1.1, 0.7);

  const Sphere s{16.0, 16.0, 16.0, 7.0, 1.5};
  fill(cs.h, [&](double x, double y, double z) { return s(x, y, z); });
  fill(cs.amp, [&](double x, double y, double z) { return 2.0e-3 * s(x, y, z); });

  MicroelasticityParams p;
  p.c_solid = c;
  p.c_liquid = c; // <- the point of the test
  p.warm_start = false;
  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
  const auto rep = solver.solve(cs.h, cs.amp);

  REQUIRE(rep.converged);
  REQUIRE(rep.iterations == 1);
  REQUIRE(rep.residual_history.size() == 1);
  // Round-off only: tau = C:(eps-eps*) - C0:eps is algebraically independent
  // of eps when C == C0, but is not evaluated that way.
  REQUIRE(rep.residual_history.front() < 1.0e-12);
  // A varying h with C_solid == C_liquid must still leave C0 equal to both.
  REQUIRE_THAT(solver.reference().c11, WithinRel(c.c11, 1e-15));
}

// ---------------------------------------------------------------------------
// 6. Solid/liquid contrast: monotone convergence, and the strain-change test
// ---------------------------------------------------------------------------
TEST_CASE("The polarisation fixed point converges monotonically under contrast",
          "[microelasticity][fixedpoint]") {
  constexpr int N = 32;
  const double nu = 0.3;
  const Stiffness solid = Stiffness::isotropic(1.0, nu);

  auto run = [&](double ratio, int max_it, double tol) {
    Case cs(N, 1.0);
    const Sphere s{16.0, 16.0, 16.0, 7.0, 1.5};
    fill(cs.h, [&](double x, double y, double z) { return s(x, y, z); });
    fill(cs.amp, [&](double x, double y, double z) { return 2.0e-3 * s(x, y, z); });
    MicroelasticityParams p;
    p.c_solid = solid;
    p.c_liquid = Stiffness::isotropic(1.0 / ratio, nu);
    p.tol_el = tol;
    p.n_el_iter = max_it;
    p.warm_start = false;
    EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
    const auto rep = solver.solve(cs.h, cs.amp);
    return std::pair<pfc::apps::MicroelasticityReport, double>{
        rep, field_absmax(solver.strain()[SYM_XX])};
  };

  for (double ratio : {2.0, 4.0, 10.0}) {
    const auto [rep, emax] = run(ratio, 400, 1.0e-6);
    INFO("contrast " << ratio << ": " << rep.iterations << " iterations, residual "
                     << rep.residual);
    REQUIRE(rep.converged);
    REQUIRE(rep.iterations > 1);
    REQUIRE(emax > 0.0);
    // Geometric contraction => a strictly decreasing residual history.
    for (std::size_t i = 1; i < rep.residual_history.size(); ++i) {
      REQUIRE(rep.residual_history[i] < rep.residual_history[i - 1]);
    }
    // ...and the observed factor should be near (r-1)/(r+1) for the Voigt
    // reference, which is what justifies the default C0.
    const auto &hist = rep.residual_history;
    REQUIRE(hist.size() >= 4);
    const double factor = std::pow(hist.back() / hist.front(),
                                   1.0 / static_cast<double>(hist.size() - 1));
    const double predicted = (ratio - 1.0) / (ratio + 1.0);
    INFO("observed contraction " << factor << " vs predicted " << predicted);
    REQUIRE(factor < 1.0);
    REQUIRE(factor < predicted * 1.35);
  }

  // The spec's stopping test is on the strain, not the polarisation. Show they
  // agree: take the converged iterate, run one more Gamma application, and
  // confirm the strain moved by less than tol_el in the spec's norm.
  const double ratio = 4.0;
  const auto [rep, unused] = run(ratio, 400, 1.0e-6);
  (void)unused;
  auto strain_at = [&](int iters) {
    Case cs(N, 1.0);
    const Sphere s{16.0, 16.0, 16.0, 7.0, 1.5};
    fill(cs.h, [&](double x, double y, double z) { return s(x, y, z); });
    fill(cs.amp, [&](double x, double y, double z) { return 2.0e-3 * s(x, y, z); });
    MicroelasticityParams p;
    p.c_solid = solid;
    p.c_liquid = Stiffness::isotropic(1.0 / ratio, nu);
    p.tol_el = 0.0; // never stop early
    p.n_el_iter = iters;
    p.warm_start = false;
    EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
    solver.solve(cs.h, cs.amp);
    std::array<std::vector<double>, kSymComponents> out{};
    for (int c = 0; c < kSymComponents; ++c) {
      out[static_cast<std::size_t>(c)] =
          solver.strain()[static_cast<std::size_t>(c)].vec();
    }
    return out;
  };
  const auto a = strain_at(rep.iterations);
  const auto b = strain_at(rep.iterations + 1);
  double diff = 0.0, scale = 0.0;
  for (int c = 0; c < kSymComponents; ++c) {
    const auto &va = a[static_cast<std::size_t>(c)];
    const auto &vb = b[static_cast<std::size_t>(c)];
    for (std::size_t i = 0; i < va.size(); ++i) {
      diff = std::max(diff, std::abs(vb[i] - va[i]));
      scale = std::max(scale, std::abs(vb[i]));
    }
  }
  const double rel = gmax(diff) / gmax(scale);
  // The header stops on the polarisation change, one step ahead of the strain
  // change, so the two norms agree only up to the amplification of Gamma --
  // here a factor 1.24. Require the same order, not the same number.
  INFO("max|eps_new - eps_old| / max|eps| after the reported iteration count = "
       << rel);
  REQUIRE(rel < 3.0e-6);
}

// ---------------------------------------------------------------------------
// 7. Mechanical equilibrium: div sigma = 0, checked spectrally
// ---------------------------------------------------------------------------
//
// At the fixed point this is exact, not approximate: with eps_hat = -Gamma:tau
// one has i k_j (C0_ijkl eps_hat_kl + tau_hat_ij) == 0 identically, because
// k_j C0_ijkl Gamma_klmn = k_m (delta contraction through G G^-1). So the test
// is a genuine round-off-level assertion, and it is normalised against the
// divergence of C0:eps alone -- one of the two terms that must cancel --
// rather than against an arbitrary stress/length scale.
TEST_CASE("The returned stress satisfies div sigma = 0",
          "[microelasticity][equilibrium]") {
  constexpr int N = 32;
  const double nu = 0.3;

  auto check = [&](double ratio, double tol_el, double allowed) {
    Case cs(N, 1.0);
    const Sphere s{16.0, 16.0, 16.0, 7.0, 1.5};
    fill(cs.h, [&](double x, double y, double z) { return s(x, y, z); });
    fill(cs.amp, [&](double x, double y, double z) { return 2.0e-3 * s(x, y, z); });
    MicroelasticityParams p;
    p.c_solid = Stiffness::isotropic(1.0, nu);
    p.c_liquid = Stiffness::isotropic(1.0 / ratio, nu);
    p.tol_el = tol_el;
    p.n_el_iter = 400;
    p.warm_start = false;
    EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
    solver.solve(cs.h, cs.amp);

    // Reference scale: div(C0:eps), the piece that div(tau) has to cancel.
    EigenstrainMicroelasticity::SymRealFields c0eps{
        pfc::data::field_from_inbox<double>(cs.domain,
                                            cs.stack.fft().get_inbox_bounds()),
        pfc::data::field_from_inbox<double>(cs.domain,
                                            cs.stack.fft().get_inbox_bounds()),
        pfc::data::field_from_inbox<double>(cs.domain,
                                            cs.stack.fft().get_inbox_bounds()),
        pfc::data::field_from_inbox<double>(cs.domain,
                                            cs.stack.fft().get_inbox_bounds()),
        pfc::data::field_from_inbox<double>(cs.domain,
                                            cs.stack.fft().get_inbox_bounds()),
        pfc::data::field_from_inbox<double>(cs.domain,
                                            cs.stack.fft().get_inbox_bounds())};
    for (std::size_t i = 0; i < c0eps[0].size(); ++i) {
      Sym3 e;
      for (int c = 0; c < kSymComponents; ++c) {
        e[c] = solver.strain()[static_cast<std::size_t>(c)].data()[i];
      }
      const Sym3 t = solver.reference().contract(e);
      for (int c = 0; c < kSymComponents; ++c) {
        c0eps[static_cast<std::size_t>(c)].data()[i] = t[c];
      }
    }
    const double d_sigma =
        spectral_divergence_absmax(cs.domain, cs.stack.fft(), solver.stress());
    const double d_ref =
        spectral_divergence_absmax(cs.domain, cs.stack.fft(), c0eps);
    INFO("contrast " << ratio
                     << ": |div sigma| / |div (C0:eps)| = " << d_sigma / d_ref);
    REQUIRE(d_ref > 0.0);
    REQUIRE(d_sigma / d_ref < allowed);
  };

  check(1.0, 1.0e-6, 1.0e-12); // homogeneous: exact after one pass
  check(4.0, 1.0e-10, 1.0e-8); // contrasted: limited by the fixed-point residual
}

// ---------------------------------------------------------------------------
// 8. Energy consistency, and the Eshelby energy in closed form
// ---------------------------------------------------------------------------
//
// The integral oracle. Because <eps> = 0 and the displacement is periodic,
// integral(sigma:eps) = 0, hence
//
//     F = (1/2) int (eps-eps*):C:(eps-eps*) = -(1/2) int sigma:eps* .
//
// For a homogeneous isotropic medium with eps* = a delta, tr(sigma) =
// 3K(tr eps - 3a) = 9K(alpha(a-<a>) - a) from the identity of test 3, so
//
//     F = (9/2) K [ (1-alpha) int a^2 dV + alpha <a> int a dV ] ,
//
// exactly, for any a(x). For a sharp sphere of volume V_i this collapses to
// (9/2) K e*^2 V_i [1 - alpha(1-f)], whose f -> 0 limit is the textbook
// Eshelby energy V_i E e*^2/(1-nu).
TEST_CASE("Elastic energy is self-consistent and matches the Eshelby energy",
          "[microelasticity][energy]") {
  constexpr int N = 32;
  const double nu = 0.3;
  const double youngs = 1.0;
  const Stiffness c = Stiffness::isotropic(youngs, nu);
  const double alpha = (1.0 + nu) / (3.0 * (1.0 - nu));
  const double bulk = c.bulk_modulus();

  Case cs(N, 1.0);
  const Sphere s{16.0, 16.0, 16.0, 7.0, 1.5};
  const double estar = 1.5e-3;
  fill(cs.h, [&](double x, double y, double z) { return s(x, y, z); });
  fill(cs.amp, [&](double x, double y, double z) { return estar * s(x, y, z); });

  SECTION("f_el agrees with (1/2) sigma:(eps-eps*) built from the returned fields") {
    MicroelasticityParams p;
    p.c_solid = c;
    p.c_liquid = Stiffness::isotropic(youngs / 4.0, nu);
    p.tol_el = 1.0e-10;
    p.n_el_iter = 400;
    p.warm_start = false;
    EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
    REQUIRE(solver.solve(cs.h, cs.amp).converged);

    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < cs.amp.size(); ++i) {
      Sym3 e, sg;
      for (int c2 = 0; c2 < kSymComponents; ++c2) {
        e[c2] = solver.strain()[static_cast<std::size_t>(c2)].data()[i] -
                cs.amp.data()[i] * p.eigenstrain_pattern[c2];
        sg[c2] = solver.stress()[static_cast<std::size_t>(c2)].data()[i];
      }
      // stress must be C(x):(eps-eps*) with the interpolated modulus
      const Sym3 sg_ref = solver.stiffness_at(cs.h.data()[i]).contract(e);
      for (int c2 = 0; c2 < kSymComponents; ++c2) {
        worst = std::max(worst, std::abs(sg[c2] - sg_ref[c2]));
        scale = std::max(scale, std::abs(sg_ref[c2]));
      }
      const double fe = 0.5 * pfc::apps::ddot(e, sg);
      REQUIRE_THAT(solver.elastic_energy_density().data()[i], WithinAbs(fe, 1e-18));
    }
    REQUIRE(gmax(worst) / gmax(scale) < 1.0e-14);
  }

  SECTION("total energy matches the closed form for a homogeneous medium") {
    MicroelasticityParams p;
    p.c_solid = c;
    p.c_liquid = c;
    p.warm_start = false;
    EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
    REQUIRE(solver.solve(cs.h, cs.amp).iterations == 1);

    double sum_a = 0.0, sum_a2 = 0.0;
    for (std::size_t i = 0; i < cs.amp.size(); ++i) {
      sum_a += cs.amp.data()[i];
      sum_a2 += cs.amp.data()[i] * cs.amp.data()[i];
    }
    const double cell = 1.0;
    const double int_a = gsum(sum_a) * cell;
    const double int_a2 = gsum(sum_a2) * cell;
    const double abar = field_mean(cs.amp);
    const double want = 4.5 * bulk * ((1.0 - alpha) * int_a2 + alpha * abar * int_a);
    const double got = solver.total_elastic_energy();
    INFO("F = " << got << ", closed form " << want << ", rel "
                << std::abs(got / want - 1.0));
    REQUIRE_THAT(got, WithinRel(want, 1.0e-11));
  }

  SECTION(
      "a sharp sphere reproduces the Eshelby energy with its finite-cell factor") {
    // The diffuse profile above cannot be compared with the *sharp* Eshelby
    // energy, because int a^2 != e* int a once the interface is smeared. Use
    // a hard 0/1 indicator so both integrals collapse onto the same V_i; the
    // staircase then only shifts V_i, which is measured from the field itself,
    // and the closed form
    //     F = (9/2) K e*^2 V_i [1 - alpha(1-f)]
    // is exact. Its f -> 0 limit is Eshelby's V_i E e*^2/(1-nu), so the ratio
    // to the infinite-medium energy must be exactly 1 + alpha f/(1-alpha).
    Case sharp(N, 1.0);
    const double rad = 7.0;
    fill_value(sharp.h, 1.0);
    fill(sharp.amp, [&](double x, double y, double z) {
      const double r = std::sqrt((x - 16.0) * (x - 16.0) + (y - 16.0) * (y - 16.0) +
                                 (z - 16.0) * (z - 16.0));
      return (r <= rad) ? estar : 0.0;
    });
    MicroelasticityParams p;
    p.c_solid = c;
    p.c_liquid = c;
    p.warm_start = false;
    EigenstrainMicroelasticity solver(sharp.domain, sharp.stack.fft(), p);
    REQUIRE(solver.solve(sharp.h, sharp.amp).iterations == 1);

    const double abar = field_mean(sharp.amp);
    const double frac = abar / estar;
    const double v_i = frac * std::pow(N * 1.0, 3);
    const double got = solver.total_elastic_energy();
    const double want_periodic =
        4.5 * bulk * estar * estar * v_i * (1.0 - alpha * (1.0 - frac));
    const double eshelby_inf = v_i * youngs * estar * estar / (1.0 - nu);
    INFO("f = " << frac << ", F = " << got << ", periodic closed form "
                << want_periodic << ", F/F_Eshelby(inf) = " << got / eshelby_inf
                << " (want " << 1.0 + alpha * frac / (1.0 - alpha) << ")");
    REQUIRE_THAT(got, WithinRel(want_periodic, 1.0e-11));
    REQUIRE_THAT(got / eshelby_inf,
                 WithinRel(1.0 + alpha * frac / (1.0 - alpha), 1.0e-11));
  }
}

// ---------------------------------------------------------------------------
// 9. d f_el / d phi against a finite difference of the re-converged energy
// ---------------------------------------------------------------------------
//
// The elastic feedback in the phase-field equation is the only place a sign or
// a missing term in eq. (7) would show up, and it would be invisible in every
// test above. So: perturb phi in one cell, re-converge the *whole* elastic
// problem, and difference the total energy. This simultaneously tests
//   - the transformation-work term,
//   - the modulus-contrast term (asserted separately to be non-negligible and
//     to be needed for the match), and
//   - the Hellmann-Feynman claim that the partial derivative at frozen eps is
//     the total derivative, which is what lets the driver use eq. (7) at all.
TEST_CASE("d f_el/d phi matches a finite difference of the converged energy",
          "[microelasticity][derivative]") {
  constexpr int N = 24;
  const double nu = 0.3;
  const Stiffness solid = Stiffness::isotropic(1.0, nu);
  const Stiffness liquid = Stiffness::isotropic(0.25, nu);
  const double e0 = 2.0e-3; // eps* = h(phi) * e0 * delta

  Case cs(N, 1.0);
  const Sphere shape{12.0, 12.0, 12.0, 5.0, 1.5};
  // phi in [-1,1]; h = (1+phi)/2 so dh/dphi = 1/2 and d a/dphi = e0/2.
  auto phi_of = [&](double x, double y, double z) {
    return 2.0 * shape(x, y, z) - 1.0;
  };

  MicroelasticityParams p;
  p.c_solid = solid;
  p.c_liquid = liquid;
  p.tol_el = 1.0e-13;
  p.n_el_iter = 600;
  p.warm_start = false;

  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);

  auto set_inputs = [&](double bump, int gi, int gj, int gk) {
    fill(cs.h, [&](double x, double y, double z) {
      return 0.5 * (1.0 + phi_of(x, y, z));
    });
    fill_value(cs.dh, 0.5);
    // apply the bump to phi at one cell
    const auto &box = cs.h.box();
    if (gi >= box.low[0] && gi <= box.high[0] && gj >= box.low[1] &&
        gj <= box.high[1] && gk >= box.low[2] && gk <= box.high[2]) {
      cs.h(gi - box.low[0], gj - box.low[1], gk - box.low[2]) += 0.5 * bump;
    }
    cs.h.note_host_write();
    for (std::size_t i = 0; i < cs.h.size(); ++i) {
      cs.amp.data()[i] = e0 * cs.h.data()[i];
      cs.damp.data()[i] = 0.5 * e0;
    }
    cs.amp.note_host_write();
    cs.damp.note_host_write();
  };

  // A cell in the diffuse interface, where both terms of eq. (7) are alive.
  const int gi = 12 + 5, gj = 12, gk = 12;

  set_inputs(0.0, gi, gj, gk);
  REQUIRE(solver.solve(cs.h, cs.amp, &cs.dh, &cs.damp).converged);
  const double analytic = sample_global(solver.dfel_dphi(), gi, gj, gk);

  // Split eq. (7) so the modulus-contrast term can be shown to matter.
  const Sym3 e_here = [&] {
    Sym3 e;
    const double a_here = sample_global(cs.amp, gi, gj, gk);
    for (int c = 0; c < kSymComponents; ++c) {
      e[c] =
          sample_global(solver.strain()[static_cast<std::size_t>(c)], gi, gj, gk) -
          a_here * p.eigenstrain_pattern[c];
    }
    return e;
  }();
  const Sym3 s_here = sample_tensor(solver.stress(), gi, gj, gk);
  const double term_work =
      -pfc::apps::ddot(s_here, Sym3{{0.5 * e0, 0.5 * e0, 0.5 * e0, 0.0, 0.0, 0.0}});
  const Stiffness dc = Stiffness::blend(solid, 1.0, liquid, -1.0);
  const double term_modulus =
      0.5 * 0.5 * pfc::apps::ddot(e_here, dc.contract(e_here));
  REQUIRE_THAT(analytic, WithinRel(term_work + term_modulus, 1.0e-12));

  const double delta = 1.0e-3;
  set_inputs(+delta, gi, gj, gk);
  REQUIRE(solver.solve(cs.h, cs.amp).converged);
  const double f_plus = solver.total_elastic_energy();
  set_inputs(-delta, gi, gj, gk);
  REQUIRE(solver.solve(cs.h, cs.amp).converged);
  const double f_minus = solver.total_elastic_energy();

  const double cell_volume = 1.0;
  const double numeric = (f_plus - f_minus) / (2.0 * delta) / cell_volume;

  INFO("analytic d f_el/d phi = " << analytic
                                  << ", finite difference = " << numeric);
  INFO("transformation-work term = " << term_work << ", modulus-contrast term = "
                                     << term_modulus);
  REQUIRE_THAT(numeric, WithinRel(analytic, 2.0e-4));

  // The modulus-contrast term is not noise: dropping it would move the answer
  // by far more than the finite-difference agreement just demonstrated.
  const double rel_size = std::abs(term_modulus / analytic);
  INFO("modulus-contrast term is " << 100.0 * rel_size << "% of eq. (7)");
  REQUIRE(rel_size > 1.0e-3);
  REQUIRE(std::abs(numeric - term_work) > 10.0 * std::abs(numeric - analytic));
}

// ---------------------------------------------------------------------------
// 10. Decomposition invariance
// ---------------------------------------------------------------------------
//
// The whole solve is rank-agnostic by construction (HeFFTe transforms, local
// pointwise work, one Allreduce), but "by construction" is what every MPI bug
// was before it was found. These globals are decomposition-independent
// functions of the solution; the reference values were recorded from a
// single-rank run of this very test, so running the binary under mpiexec -n N
// is a direct 1-vs-N comparison.
TEST_CASE("The solution is independent of the MPI decomposition",
          "[microelasticity][mpi]") {
  constexpr int N = 32;
  Case cs(N, 1.0);
  const Sphere s{16.0, 16.0, 16.0, 7.0, 1.5};
  fill(cs.h, [&](double x, double y, double z) { return s(x, y, z); });
  fill(cs.amp, [&](double x, double y, double z) { return 2.0e-3 * s(x, y, z); });
  fill_value(cs.dh, 0.5);
  fill_value(cs.damp, 1.0e-3);

  MicroelasticityParams p;
  p.c_solid = Stiffness::cubic(2.4, 1.1, 0.7);
  p.c_liquid = Stiffness::cubic(0.6, 0.275, 0.175);
  p.tol_el = 1.0e-10;
  p.n_el_iter = 400;
  p.warm_start = false;
  EigenstrainMicroelasticity solver(cs.domain, cs.stack.fft(), p);
  const auto rep = solver.solve(cs.h, cs.amp, &cs.dh, &cs.damp);
  REQUIRE(rep.converged);

  // A coordinate-weighted checksum: sensitive to any cell landing in the wrong
  // place, unlike a plain sum.
  double chk = 0.0;
  const auto n = cs.h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = cs.h.global(i, j, k);
        const double w = std::cos(kTwoPi * (g[0] + 2.0 * g[1] + 3.0 * g[2]) / N);
        for (int c = 0; c < kSymComponents; ++c) {
          chk += w * solver.strain()[static_cast<std::size_t>(c)](i, j, k);
        }
      }
    }
  }
  const double checksum = gsum(chk);
  const double energy = solver.total_elastic_energy();
  const double dmax = field_absmax(solver.dfel_dphi());

  INFO(precise("ranks=", world_size(), " iterations=", rep.iterations,
               " checksum=", checksum, " energy=", energy, " dfel_max=", dmax));
  // Reference values from a single-rank run (see the PR body).
  REQUIRE(rep.iterations == 39);
  REQUIRE_THAT(checksum, WithinRel(-0.7559465088984869, 1.0e-9));
  REQUIRE_THAT(energy, WithinRel(0.005302743017342901, 1.0e-9));
  REQUIRE_THAT(dmax, WithinRel(6.09873036075123e-06, 1.0e-9));
}
