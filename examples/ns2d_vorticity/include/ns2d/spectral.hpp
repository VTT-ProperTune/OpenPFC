// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral.hpp
 * @brief Shared 2-D periodic spectral plane for the NS (#21/#22) and MHD
 *        (#23) research prototypes.
 *
 * Owns the Orszag 2/3 mask, odd-\f$k\f$ multipliers, Poisson
 * \f$\hat\phi=\hat\omega/|k|^2\f$, and IFRK4 stage algebra. It is not a
 * generic multiphysics framework: only the pieces both solvers actually
 * share.
 *
 * Sign convention (issue #23): \f$\omega=-\nabla^2\phi\f$,
 * \f$j=-\nabla^2 a\f$, \f$\mathbf u=(\partial_y\phi,-\partial_x\phi)\f$,
 * \f$\mathbf B=(\partial_y a,-\partial_x a)\f$.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace ns2d {

class SpectralPlane {
public:
  using Complex = std::complex<double>;

  SpectralPlane(pfc::fft::IHostFFT &fft, const pfc::data::Field<double> &layout)
      : m_fft(&fft), m_inbox(fft.get_inbox_bounds()),
        m_outbox(fft.get_outbox_bounds()), m_gsize(layout.global_size()),
        m_spacing(layout.spacing()) {
    if (m_gsize[2] != 1) {
      throw std::invalid_argument("ns2d: expected a one-cell z slab (Nx x Ny x 1)");
    }
    const std::size_t out_n = fft.size_outbox();
    m_tmp_hat.assign(out_n, Complex{});
    m_acc_hat.assign(out_n, Complex{});
    m_mask.assign(out_n, 0.0);
    m_kx_odd.assign(out_n, 0.0);
    m_ky_odd.assign(out_n, 0.0);
    m_inv_k2.assign(out_n, 0.0);
    m_k2.assign(out_n, 0.0);
    pfc::fft::kspace::fill_two_thirds_mask(m_outbox, m_gsize, m_spacing,
                                           m_mask.data(), m_mask.size());
    pfc::fft::kspace::for_each_kpoint(
        m_outbox, m_gsize, m_spacing,
        [&](std::size_t idx, double kx, double ky, double /*kz*/, int i, int j,
            int k) {
          m_kx_odd[idx] =
              pfc::fft::kspace::is_nyquist_index(i, m_gsize[0]) ? 0.0 : kx;
          m_ky_odd[idx] =
              pfc::fft::kspace::is_nyquist_index(j, m_gsize[1]) ? 0.0 : ky;
          const double k2 = kx * kx + ky * ky;
          m_k2[idx] = k2;
          const bool dc = (i == 0 && j == 0 && k == 0);
          m_inv_k2[idx] = (dc || k2 == 0.0) ? 0.0 : 1.0 / k2;
          if (dc) m_has_dc = true;
        });
  }

  [[nodiscard]] pfc::fft::IHostFFT &fft() noexcept { return *m_fft; }
  [[nodiscard]] const pfc::fft::Box3i &inbox() const noexcept { return m_inbox; }
  [[nodiscard]] const pfc::fft::Box3i &outbox() const noexcept { return m_outbox; }
  [[nodiscard]] pfc::Int3 gsize() const noexcept { return m_gsize; }
  [[nodiscard]] pfc::Real3 spacing() const noexcept { return m_spacing; }
  [[nodiscard]] bool rank_owns_zero_mode() const noexcept { return m_has_dc; }
  [[nodiscard]] const std::vector<double> &mask() const noexcept { return m_mask; }
  [[nodiscard]] const std::vector<double> &inv_k2() const noexcept {
    return m_inv_k2;
  }
  [[nodiscard]] const std::vector<double> &k2() const noexcept { return m_k2; }
  [[nodiscard]] const std::vector<double> &kx_odd() const noexcept {
    return m_kx_odd;
  }
  [[nodiscard]] const std::vector<double> &ky_odd() const noexcept {
    return m_ky_odd;
  }
  [[nodiscard]] std::size_t out_n() const noexcept { return m_mask.size(); }
  [[nodiscard]] std::size_t in_n() const noexcept { return m_fft->size_inbox(); }

  void project_hat(std::vector<Complex> &hat) const {
    for (std::size_t i = 0; i < hat.size(); ++i) hat[i] *= m_mask[i];
  }

  void zero_dc(std::vector<Complex> &hat) const {
    if (!m_has_dc) return;
    pfc::fft::kspace::for_each_kpoint(
        m_outbox, m_gsize, m_spacing,
        [&](std::size_t idx, double, double, double, int i, int j, int k) {
          if (i == 0 && j == 0 && k == 0) hat[idx] = Complex{0.0, 0.0};
        });
  }

  /// \f$\hat\phi=\hat\omega/|k|^2\f$ with \f$\hat\phi(0)=0\f$
  /// (\f$\omega=-\nabla^2\phi\f$).
  void poisson(const std::vector<Complex> &omega_hat,
               std::vector<Complex> &phi_hat) const {
    for (std::size_t i = 0; i < omega_hat.size(); ++i) {
      phi_hat[i] = omega_hat[i] * m_inv_k2[i];
    }
  }

  /// \f$\hat j = |k|^2 \hat a\f$ (\f$j=-\nabla^2 a\f$).
  void current_from_flux(const std::vector<Complex> &a_hat,
                         std::vector<Complex> &j_hat) const {
    for (std::size_t i = 0; i < a_hat.size(); ++i) {
      j_hat[i] = a_hat[i] * m_k2[i];
    }
  }

  void invert_ik(const std::vector<Complex> &in, const std::vector<double> &kodd,
                 double sign, std::vector<double> &out) {
    for (std::size_t i = 0; i < in.size(); ++i) {
      m_tmp_hat[i] = in[i] * Complex(0.0, sign * kodd[i]);
    }
    m_fft->backward(m_tmp_hat, out);
  }

  /// Velocity or magnetic field from a stream/flux function: \f$( \partial_y,
  /// -\partial_x )\f$.
  void curl_from_hat(const std::vector<Complex> &psi_hat, std::vector<double> &cx,
                     std::vector<double> &cy) {
    invert_ik(psi_hat, m_ky_odd, +1.0, cx);
    invert_ik(psi_hat, m_kx_odd, -1.0, cy);
  }

  void fill_exp(double coeff, double dt, std::vector<double> &exp_dt,
                std::vector<double> &exp_half) const {
    exp_dt.resize(m_k2.size());
    exp_half.resize(m_k2.size());
    for (std::size_t i = 0; i < m_k2.size(); ++i) {
      const double L = -coeff * m_k2[i];
      exp_dt[i] = std::exp(L * dt);
      exp_half[i] = std::exp(L * 0.5 * dt);
    }
  }

  [[nodiscard]] std::size_t real_idx(int i, int j, int k) const noexcept {
    const auto nx = static_cast<std::size_t>(m_inbox.size[0]);
    const auto ny = static_cast<std::size_t>(m_inbox.size[1]);
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * nx +
           static_cast<std::size_t>(k) * nx * ny;
  }

  void spectral_div(const std::vector<double> &cx, const std::vector<double> &cy,
                    std::vector<double> &div) {
    m_fft->forward(cx, m_tmp_hat);
    for (std::size_t i = 0; i < m_tmp_hat.size(); ++i) {
      m_acc_hat[i] = m_tmp_hat[i] * Complex(0.0, m_kx_odd[i]);
    }
    m_fft->forward(cy, m_tmp_hat);
    for (std::size_t i = 0; i < m_tmp_hat.size(); ++i) {
      m_acc_hat[i] += m_tmp_hat[i] * Complex(0.0, m_ky_odd[i]);
    }
    m_fft->backward(m_acc_hat, div);
  }

private:
  pfc::fft::IHostFFT *m_fft{nullptr};
  pfc::fft::Box3i m_inbox{};
  pfc::fft::Box3i m_outbox{};
  pfc::Int3 m_gsize{};
  pfc::Real3 m_spacing{};
  bool m_has_dc{false};
  std::vector<Complex> m_tmp_hat, m_acc_hat;
  std::vector<double> m_mask, m_kx_odd, m_ky_odd, m_inv_k2, m_k2;
};

/// One IFRK4 step for a single prognostic spectrum (NS vorticity).
template <class Nonlinear>
void ifrk4_one(SpectralPlane &sp, std::vector<SpectralPlane::Complex> &u_hat,
               const std::vector<double> &exp_dt,
               const std::vector<double> &exp_half, double dt, Nonlinear &&N,
               std::vector<SpectralPlane::Complex> &n1,
               std::vector<SpectralPlane::Complex> &n2,
               std::vector<SpectralPlane::Complex> &n3,
               std::vector<SpectralPlane::Complex> &n4,
               std::vector<SpectralPlane::Complex> &stage) {
  sp.project_hat(u_hat);
  N(u_hat, n1);
  for (std::size_t i = 0; i < u_hat.size(); ++i) {
    stage[i] = exp_half[i] * (u_hat[i] + (0.5 * dt) * n1[i]);
  }
  sp.project_hat(stage);
  N(stage, n2);
  for (std::size_t i = 0; i < u_hat.size(); ++i) {
    stage[i] = exp_half[i] * u_hat[i] + (0.5 * dt) * n2[i];
  }
  sp.project_hat(stage);
  N(stage, n3);
  for (std::size_t i = 0; i < u_hat.size(); ++i) {
    stage[i] = exp_dt[i] * u_hat[i] + dt * exp_half[i] * n3[i];
  }
  sp.project_hat(stage);
  N(stage, n4);
  const double dt6 = dt / 6.0;
  for (std::size_t i = 0; i < u_hat.size(); ++i) {
    u_hat[i] =
        exp_dt[i] * u_hat[i] +
        dt6 * (exp_dt[i] * n1[i] + 2.0 * exp_half[i] * (n2[i] + n3[i]) + n4[i]);
  }
  sp.project_hat(u_hat);
}

/// IFRK4 for a coupled pair with (possibly different) linear symbols.
template <class Nonlinear>
void ifrk4_pair(SpectralPlane &sp, std::vector<SpectralPlane::Complex> &w_hat,
                std::vector<SpectralPlane::Complex> &a_hat,
                const std::vector<double> &ew_dt, const std::vector<double> &ew_half,
                const std::vector<double> &ea_dt, const std::vector<double> &ea_half,
                double dt, Nonlinear &&N, std::vector<SpectralPlane::Complex> &nw1,
                std::vector<SpectralPlane::Complex> &na1,
                std::vector<SpectralPlane::Complex> &nw2,
                std::vector<SpectralPlane::Complex> &na2,
                std::vector<SpectralPlane::Complex> &nw3,
                std::vector<SpectralPlane::Complex> &na3,
                std::vector<SpectralPlane::Complex> &nw4,
                std::vector<SpectralPlane::Complex> &na4,
                std::vector<SpectralPlane::Complex> &sw,
                std::vector<SpectralPlane::Complex> &sa) {
  sp.project_hat(w_hat);
  sp.project_hat(a_hat);
  N(w_hat, a_hat, nw1, na1);
  for (std::size_t i = 0; i < w_hat.size(); ++i) {
    sw[i] = ew_half[i] * (w_hat[i] + (0.5 * dt) * nw1[i]);
    sa[i] = ea_half[i] * (a_hat[i] + (0.5 * dt) * na1[i]);
  }
  sp.project_hat(sw);
  sp.project_hat(sa);
  N(sw, sa, nw2, na2);
  for (std::size_t i = 0; i < w_hat.size(); ++i) {
    sw[i] = ew_half[i] * w_hat[i] + (0.5 * dt) * nw2[i];
    sa[i] = ea_half[i] * a_hat[i] + (0.5 * dt) * na2[i];
  }
  sp.project_hat(sw);
  sp.project_hat(sa);
  N(sw, sa, nw3, na3);
  for (std::size_t i = 0; i < w_hat.size(); ++i) {
    sw[i] = ew_dt[i] * w_hat[i] + dt * ew_half[i] * nw3[i];
    sa[i] = ea_dt[i] * a_hat[i] + dt * ea_half[i] * na3[i];
  }
  sp.project_hat(sw);
  sp.project_hat(sa);
  N(sw, sa, nw4, na4);
  const double dt6 = dt / 6.0;
  for (std::size_t i = 0; i < w_hat.size(); ++i) {
    w_hat[i] =
        ew_dt[i] * w_hat[i] +
        dt6 * (ew_dt[i] * nw1[i] + 2.0 * ew_half[i] * (nw2[i] + nw3[i]) + nw4[i]);
    a_hat[i] =
        ea_dt[i] * a_hat[i] +
        dt6 * (ea_dt[i] * na1[i] + 2.0 * ea_half[i] * (na2[i] + na3[i]) + na4[i]);
  }
  sp.project_hat(w_hat);
  sp.project_hat(a_hat);
}

/**
 * Elsasser speeds z± = u ± B.
 *
 * `max_inf` is the max absolute Cartesian component of either z+ or z-.
 * `max_mag` is the max Euclidean |z±|.
 * `max_l1` is max(|z_x^±| + |z_y^±|).
 * `max_sum_inv` is max(|z_x^±|/dx + |z_y^±|/dy), the inverse-time
 * multidimensional characteristic bound. The fail-closed CFL is
 * `dt * max_sum_inv`.
 */
struct ElsasserSpeeds {
  double max_inf{0.0};
  double max_mag{0.0};
  double max_l1{0.0};
  double max_sum_inv{0.0};
};

[[nodiscard]] inline ElsasserSpeeds
elsasser_speeds(const std::vector<double> &u, const std::vector<double> &v,
                const std::vector<double> &bx, const std::vector<double> &by,
                double dx = 1.0, double dy = 1.0) {
  ElsasserSpeeds s;
  const std::size_t n = u.size();
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  auto accum = [&](double zx, double zy) {
    const double ax = std::abs(zx);
    const double ay = std::abs(zy);
    s.max_inf = std::max(s.max_inf, std::max(ax, ay));
    s.max_mag = std::max(s.max_mag, std::hypot(zx, zy));
    s.max_l1 = std::max(s.max_l1, ax + ay);
    s.max_sum_inv = std::max(s.max_sum_inv, ax * inv_dx + ay * inv_dy);
  };
  for (std::size_t i = 0; i < n; ++i) {
    accum(u[i] + bx[i], v[i] + by[i]);
    accum(u[i] - bx[i], v[i] - by[i]);
  }
  return s;
}

/// Signed FFT wave-number index (Nyquist folding), matching `k_component`.
using pfc::fft::kspace::signed_wave_index;

/**
 * Copy matching integer wave numbers from a finer r2c hat into a coarser hat.
 *
 * HeFFTe/FFTW-style unnormalized forward transforms scale as the number of
 * real cells, so retained coefficients are multiplied by
 * \f$N_{\mathrm{coarse}}^2/N_{\mathrm{fine}}^2\f$. Backward on the coarse
 * grid then recovers the band-limited field. Same-size grids keep scale 1.
 *
 * Modes present on the coarse grid but missing on the fine grid stay zero.
 */
inline void restrict_hat_by_k(const SpectralPlane &fine,
                              const std::vector<SpectralPlane::Complex> &fhat,
                              const SpectralPlane &coarse,
                              std::vector<SpectralPlane::Complex> &chat) {
  chat.assign(coarse.out_n(), SpectralPlane::Complex{});
  const auto fn = fine.gsize();
  const auto cn = coarse.gsize();
  const double scale = (static_cast<double>(cn[0]) * cn[1] * cn[2]) /
                       (static_cast<double>(fn[0]) * fn[1] * fn[2]);
  auto key = [](int ki, int kj) -> long long {
    return (static_cast<long long>(static_cast<std::uint32_t>(ki)) << 32) |
           static_cast<std::uint32_t>(kj);
  };
  std::unordered_map<long long, std::size_t> fmap;
  fmap.reserve(fine.out_n());
  pfc::fft::kspace::for_each_kpoint(
      fine.outbox(), fine.gsize(), fine.spacing(),
      [&](std::size_t fi, double, double, double, int i, int j, int) {
        fmap[key(signed_wave_index(i, fn[0]), signed_wave_index(j, fn[1]))] = fi;
      });
  pfc::fft::kspace::for_each_kpoint(
      coarse.outbox(), coarse.gsize(), coarse.spacing(),
      [&](std::size_t ci, double, double, double, int i, int j, int) {
        const auto it =
            fmap.find(key(signed_wave_index(i, cn[0]), signed_wave_index(j, cn[1])));
        if (it != fmap.end()) chat[ci] = fhat[it->second] * scale;
      });
}

[[nodiscard]] inline pfc::Domain make_slab(int n, double length) {
  if (n < 4) throw std::invalid_argument("ns2d: N >= 4 is required");
  if (length <= 0.0) {
    throw std::invalid_argument("ns2d: box length must be positive");
  }
  const double dx = length / static_cast<double>(n);
  return pfc::domain::create(pfc::GridSize({n, n, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, length}));
}

[[nodiscard]] inline pfc::Domain make_twopi_slab(int n) {
  return make_slab(n, 2.0 * pfc::pi);
}

[[nodiscard]] inline pfc::Domain make_unit_slab(int n) { return make_slab(n, 1.0); }

[[nodiscard]] inline pfc::Domain make_periodic_square(int n) {
  return make_twopi_slab(n);
}

} // namespace ns2d
