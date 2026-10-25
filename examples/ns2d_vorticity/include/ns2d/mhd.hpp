// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file mhd.hpp
 * @brief 2-D incompressible visco-resistive MHD (OpenPFC issue #23).
 *
 * Prognostic fields \f$(\omega,a)\f$ on the shared \ref SpectralPlane:
 * \f[
 *   \partial_t\omega+\mathbf u\cdot\nabla\omega
 *   =\mathbf B\cdot\nabla j+\nu\nabla^2\omega,
 *   \qquad
 *   \partial_t a+\mathbf u\cdot\nabla a=\eta\nabla^2 a,
 * \f]
 * with \f$\mathbf u=(\partial_y\phi,-\partial_x\phi)\f$,
 * \f$\mathbf B=(\partial_y a,-\partial_x a)\f$,
 * \f$\omega=-\nabla^2\phi\f$, \f$j=-\nabla^2 a\f$.
 *
 * The Lorentz term is \f$+\mathbf B\cdot\nabla j\f$. A test that flips
 * this sign must fail on an Alfvénic two-mode state. This is 2-D
 * incompressible MHD, not Strauss (1976) reduced MHD.
 *
 * State-level Orszag 2/3 projection is applied to both spectra and to
 * every nonlinear product, same policy as the #22 NS solver.
 *
 * Gauge: \f$\hat a(0)=0\f$ is enforced after every stage. The reported
 * invariant is \f$A_2=\frac12\langle(a-\langle a\rangle)^2\rangle\f$.
 *
 * Timestep: fixed \f$\Delta t\f$. `--cfl` only selects
 * \f$\Delta t=\mathrm{CFL}_{\mathrm{nom}}\Delta x\f$. The fail-closed
 * characteristic CFL is
 * \f$\Delta t\,\max(|z_x^\pm|/\Delta x+|z_y^\pm|/\Delta y)\f$ with
 * Elsasser fields \f$z^\pm=\mathbf u\pm\mathbf B\f$.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#include <ns2d/spectral.hpp>

namespace ns2d {

struct MHDParams {
  double nu{0.02};
  double eta{0.02};
  double dt{0.01};
  /// +1 is the issue-#23 Lorentz sign; tests may pass -1 to prove sensitivity.
  double lorentz_sign{+1.0};
};

struct MHDDiagnostics {
  double time{0.0};
  double ke{0.0};
  double me{0.0};
  double energy{0.0};
  double cross_helicity{0.0};
  double a2{0.0};
  double enstrophy{0.0};
  double mean_sq_j{0.0};
  double max_abs_omega{0.0};
  double max_abs_j{0.0};
  double max_speed{0.0};
  double max_b{0.0};
  double div_u_linf{0.0};
  double div_b_linf{0.0};
  double dissipation{0.0};
  double energy_budget_residual{0.0};
  /// Nominal selector: dt/dx (what `--cfl` sets when used).
  double cfl_nominal{0.0};
  /// Deprecated bound: dt * max(|u_i|,|B_i|) / dx.
  double cfl_ub{0.0};
  /// Component-max Elsasser CFL: dt * max_i |z±_i| / dx.
  double cfl_elsasser{0.0};
  /// Euclidean Elsasser CFL: dt * max |z±| / dx.
  double cfl_elsasser_mag{0.0};
  /// Fail-closed 2-D bound: dt * max(|z_x^±|/dx + |z_y^±|/dy).
  double cfl_elsasser_sum{0.0};
  double max_z_inf{0.0};
  double max_z_mag{0.0};
  double max_z_l1{0.0};
  double mean_omega{0.0};
  double mean_a{0.0};
};

class MHDSolver {
public:
  using Complex = std::complex<double>;

  MHDSolver(pfc::sim::stacks::SpectralCPUStack &stack, MHDParams params)
      : m_stack(&stack), m_plane(stack.fft(), stack.u()), m_params(params),
        m_a(pfc::data::field_from_inbox<double>(stack.u().domain(),
                                                stack.fft().get_inbox_bounds())),
        m_j(pfc::data::field_from_inbox<double>(stack.u().domain(),
                                                stack.fft().get_inbox_bounds())) {
    if (params.nu < 0.0 || params.eta < 0.0 || params.dt <= 0.0) {
      throw std::invalid_argument("mhd2d: nu, eta >= 0 and dt > 0");
    }
    const std::size_t in_n = m_plane.in_n();
    const std::size_t out_n = m_plane.out_n();
    m_u.assign(in_n, 0.0);
    m_v.assign(in_n, 0.0);
    m_bx.assign(in_n, 0.0);
    m_by.assign(in_n, 0.0);
    m_wx.assign(in_n, 0.0);
    m_wy.assign(in_n, 0.0);
    m_jx.assign(in_n, 0.0);
    m_jy.assign(in_n, 0.0);
    m_ax.assign(in_n, 0.0);
    m_ay.assign(in_n, 0.0);
    m_nw.assign(in_n, 0.0);
    m_na.assign(in_n, 0.0);
    m_div_u.assign(in_n, 0.0);
    m_div_b.assign(in_n, 0.0);
    m_w_hat.assign(out_n, Complex{});
    m_a_hat.assign(out_n, Complex{});
    m_phi_hat.assign(out_n, Complex{});
    m_j_hat.assign(out_n, Complex{});
    m_sw.assign(out_n, Complex{});
    m_sa.assign(out_n, Complex{});
    m_nw1.assign(out_n, Complex{});
    m_na1.assign(out_n, Complex{});
    m_nw2.assign(out_n, Complex{});
    m_na2.assign(out_n, Complex{});
    m_nw3.assign(out_n, Complex{});
    m_na3.assign(out_n, Complex{});
    m_nw4.assign(out_n, Complex{});
    m_na4.assign(out_n, Complex{});
    m_plane.fill_exp(m_params.nu, m_params.dt, m_ew_dt, m_ew_half);
    m_plane.fill_exp(m_params.eta, m_params.dt, m_ea_dt, m_ea_half);
  }

  [[nodiscard]] MHDParams params() const noexcept { return m_params; }
  [[nodiscard]] double time() const noexcept { return m_time; }
  [[nodiscard]] pfc::data::Field<double> &omega() noexcept { return m_stack->u(); }
  [[nodiscard]] const pfc::data::Field<double> &omega() const noexcept {
    return m_stack->u();
  }
  [[nodiscard]] pfc::data::Field<double> &a() noexcept { return m_a; }
  [[nodiscard]] const pfc::data::Field<double> &a() const noexcept { return m_a; }
  [[nodiscard]] pfc::data::Field<double> &j() noexcept { return m_j; }
  [[nodiscard]] const std::vector<double> &u() const noexcept { return m_u; }
  [[nodiscard]] const std::vector<double> &v() const noexcept { return m_v; }
  [[nodiscard]] const std::vector<double> &bx() const noexcept { return m_bx; }
  [[nodiscard]] const std::vector<double> &by() const noexcept { return m_by; }
  [[nodiscard]] SpectralPlane &plane() noexcept { return m_plane; }
  [[nodiscard]] const std::vector<Complex> &last_n_omega_hat() const noexcept {
    return m_nw1;
  }

  template <class Fw, class Fa> void initialize(Fw &&fw, Fa &&fa) {
    m_stack->u().apply(std::forward<Fw>(fw));
    m_a.apply(std::forward<Fa>(fa));
    m_time = 0.0;
    m_have_prev_energy = false;
    project_current_state();
    recover_fields();
  }

  void project_current_state() {
    m_plane.fft().forward(m_stack->u().vec(), m_w_hat);
    m_plane.project_hat(m_w_hat);
    m_plane.fft().backward(m_w_hat, m_stack->u().vec());
    m_plane.fft().forward(m_a.vec(), m_a_hat);
    m_plane.project_hat(m_a_hat);
    m_plane.zero_dc(m_a_hat);
    m_plane.fft().backward(m_a_hat, m_a.vec());
  }

  void recover_fields() {
    m_plane.fft().forward(m_stack->u().vec(), m_w_hat);
    m_plane.project_hat(m_w_hat);
    m_plane.fft().forward(m_a.vec(), m_a_hat);
    m_plane.project_hat(m_a_hat);
    m_plane.zero_dc(m_a_hat);
    m_plane.poisson(m_w_hat, m_phi_hat);
    m_plane.current_from_flux(m_a_hat, m_j_hat);
    m_plane.curl_from_hat(m_phi_hat, m_u, m_v);
    m_plane.curl_from_hat(m_a_hat, m_bx, m_by);
    m_plane.fft().backward(m_j_hat, m_j.vec());
  }

  /// Evaluate \f$\hat N_\omega\f$ at the current state (for sign tests).
  void nonlinear_omega_from_current() {
    m_plane.fft().forward(m_stack->u().vec(), m_w_hat);
    m_plane.project_hat(m_w_hat);
    m_plane.fft().forward(m_a.vec(), m_a_hat);
    m_plane.project_hat(m_a_hat);
    nonlinear_from_hat(m_w_hat, m_a_hat, m_nw1, m_na1);
  }

  [[nodiscard]] double n_omega_linf(MPI_Comm comm) const {
    double local = 0.0;
    for (const auto &z : m_nw1) local = std::max(local, std::abs(z));
    double g = 0.0;
    MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g;
  }

  void step() {
    m_plane.fft().forward(m_stack->u().vec(), m_w_hat);
    m_plane.fft().forward(m_a.vec(), m_a_hat);
    ifrk4_pair(
        m_plane, m_w_hat, m_a_hat, m_ew_dt, m_ew_half, m_ea_dt, m_ea_half,
        m_params.dt,
        [&](const std::vector<Complex> &wh, const std::vector<Complex> &ah,
            std::vector<Complex> &nw,
            std::vector<Complex> &na) { nonlinear_from_hat(wh, ah, nw, na); },
        m_nw1, m_na1, m_nw2, m_na2, m_nw3, m_na3, m_nw4, m_na4, m_sw, m_sa);
    m_plane.zero_dc(m_a_hat);
    m_plane.fft().backward(m_w_hat, m_stack->u().vec());
    m_plane.fft().backward(m_a_hat, m_a.vec());
    m_time += m_params.dt;
  }

  [[nodiscard]] MHDDiagnostics diagnostics(MPI_Comm comm) {
    recover_fields();
    m_plane.spectral_div(m_u, m_v, m_div_u);
    m_plane.spectral_div(m_bx, m_by, m_div_b);

    double ke = 0.0, me = 0.0, hc = 0.0, a2 = 0.0, ens = 0.0, j2 = 0.0;
    double wsum = 0.0, asum = 0.0;
    double maxw = 0.0, maxj = 0.0, maxu = 0.0, maxb = 0.0;
    double du = 0.0, db = 0.0;
    const auto &wfield = m_stack->u();
    wfield.for_each_owned([&](int i, int j, int k) {
      const std::size_t c = m_plane.real_idx(i, j, k);
      const double w = wfield(i, j, k);
      const double aa = m_a(i, j, k);
      const double jj = m_j(i, j, k);
      const double uu = m_u[c], vv = m_v[c], bx = m_bx[c], by = m_by[c];
      ke += 0.5 * (uu * uu + vv * vv);
      me += 0.5 * (bx * bx + by * by);
      hc += uu * bx + vv * by;
      a2 += aa * aa;
      ens += 0.5 * w * w;
      j2 += jj * jj;
      wsum += w;
      asum += aa;
      maxw = std::max(maxw, std::abs(w));
      maxj = std::max(maxj, std::abs(jj));
      maxu = std::max(maxu, std::max(std::abs(uu), std::abs(vv)));
      maxb = std::max(maxb, std::max(std::abs(bx), std::abs(by)));
      du = std::max(du, std::abs(m_div_u[c]));
      db = std::max(db, std::abs(m_div_b[c]));
    });
    const auto spc_local = m_plane.spacing();
    const ElsasserSpeeds zs =
        elsasser_speeds(m_u, m_v, m_bx, m_by, spc_local[0], spc_local[1]);

    double g_ke = 0, g_me = 0, g_hc = 0, g_a2 = 0, g_ens = 0, g_j2 = 0;
    double g_w = 0, g_a = 0, g_maxw = 0, g_maxj = 0, g_maxu = 0, g_maxb = 0;
    double g_du = 0, g_db = 0;
    MPI_Allreduce(&ke, &g_ke, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&me, &g_me, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&hc, &g_hc, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&a2, &g_a2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&ens, &g_ens, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&j2, &g_j2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&wsum, &g_w, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&asum, &g_a, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&maxw, &g_maxw, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&maxj, &g_maxj, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&maxu, &g_maxu, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&maxb, &g_maxb, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&du, &g_du, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&db, &g_db, 1, MPI_DOUBLE, MPI_MAX, comm);
    double g_zinf = 0.0, g_zmag = 0.0, g_zl1 = 0.0, g_zsum = 0.0;
    MPI_Allreduce(&zs.max_inf, &g_zinf, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&zs.max_mag, &g_zmag, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&zs.max_l1, &g_zl1, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&zs.max_sum_inv, &g_zsum, 1, MPI_DOUBLE, MPI_MAX, comm);

    const auto gs = m_plane.gsize();
    const double ncells = static_cast<double>(gs[0]) * gs[1] * gs[2];
    const auto spc = m_plane.spacing();
    const double dx = std::min(spc[0], spc[1]);

    MHDDiagnostics d;
    d.time = m_time;
    d.ke = g_ke / ncells;
    d.me = g_me / ncells;
    d.energy = d.ke + d.me;
    d.cross_helicity = g_hc / ncells;
    d.mean_omega = g_w / ncells;
    d.mean_a = g_a / ncells;
    // Gauge-safe: 1/2 <(a-<a>)^2> = 1/2 (<a^2> - <a>^2). DC is also zeroed.
    d.a2 = 0.5 * (g_a2 / ncells - d.mean_a * d.mean_a);
    d.enstrophy = g_ens / ncells;
    d.mean_sq_j = g_j2 / ncells;
    d.max_abs_omega = g_maxw;
    d.max_abs_j = g_maxj;
    d.max_speed = g_maxu;
    d.max_b = g_maxb;
    d.div_u_linf = g_du;
    d.div_b_linf = g_db;
    d.dissipation = m_params.nu * (2.0 * d.enstrophy) + m_params.eta * d.mean_sq_j;
    d.cfl_nominal = m_params.dt / dx;
    d.cfl_ub = m_params.dt * std::max(g_maxu, g_maxb) / dx;
    d.max_z_inf = g_zinf;
    d.max_z_mag = g_zmag;
    d.max_z_l1 = g_zl1;
    d.cfl_elsasser = m_params.dt * g_zinf / dx;
    d.cfl_elsasser_mag = m_params.dt * g_zmag / dx;
    d.cfl_elsasser_sum = m_params.dt * g_zsum;
    if (m_have_prev_energy && d.time > m_prev_time) {
      const double dedt = (d.energy - m_prev_energy) / (d.time - m_prev_time);
      d.energy_budget_residual = dedt + 0.5 * (d.dissipation + m_prev_dissipation);
    } else {
      d.energy_budget_residual = 0.0;
    }
    m_prev_energy = d.energy;
    m_prev_dissipation = d.dissipation;
    m_prev_time = d.time;
    m_have_prev_energy = true;
    return d;
  }

  [[nodiscard]] double linf_omega_error(MPI_Comm comm, const auto &exact) const {
    double local = 0.0;
    m_stack->u().for_each_owned([&](double x, double y, double /*z*/, double w) {
      local = std::max(local, std::abs(w - exact(x, y)));
    });
    double g = 0.0;
    MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g;
  }

  [[nodiscard]] double linf_a_error(MPI_Comm comm, const auto &exact) const {
    double local = 0.0;
    m_a.for_each_owned([&](double x, double y, double /*z*/, double val) {
      local = std::max(local, std::abs(val - exact(x, y)));
    });
    double g = 0.0;
    MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g;
  }

private:
  void nonlinear_from_hat(const std::vector<Complex> &w_hat,
                          const std::vector<Complex> &a_hat,
                          std::vector<Complex> &nw, std::vector<Complex> &na) {
    m_plane.poisson(w_hat, m_phi_hat);
    m_plane.current_from_flux(a_hat, m_j_hat);
    m_plane.curl_from_hat(m_phi_hat, m_u, m_v);
    m_plane.curl_from_hat(a_hat, m_bx, m_by);
    m_plane.invert_ik(w_hat, m_plane.kx_odd(), +1.0, m_wx);
    m_plane.invert_ik(w_hat, m_plane.ky_odd(), +1.0, m_wy);
    m_plane.invert_ik(m_j_hat, m_plane.kx_odd(), +1.0, m_jx);
    m_plane.invert_ik(m_j_hat, m_plane.ky_odd(), +1.0, m_jy);
    m_plane.invert_ik(a_hat, m_plane.kx_odd(), +1.0, m_ax);
    m_plane.invert_ik(a_hat, m_plane.ky_odd(), +1.0, m_ay);
    const double s = m_params.lorentz_sign;
    for (std::size_t i = 0; i < m_nw.size(); ++i) {
      const double adv_w = m_u[i] * m_wx[i] + m_v[i] * m_wy[i];
      const double lorentz = m_bx[i] * m_jx[i] + m_by[i] * m_jy[i];
      m_nw[i] = -adv_w + s * lorentz;
      m_na[i] = -(m_u[i] * m_ax[i] + m_v[i] * m_ay[i]);
    }
    m_plane.fft().forward(m_nw, nw);
    m_plane.fft().forward(m_na, na);
    m_plane.project_hat(nw);
    m_plane.project_hat(na);
    m_plane.zero_dc(nw);
    m_plane.zero_dc(na);
  }

  pfc::sim::stacks::SpectralCPUStack *m_stack{nullptr};
  SpectralPlane m_plane;
  MHDParams m_params{};
  double m_time{0.0};
  double m_prev_energy{0.0};
  double m_prev_dissipation{0.0};
  double m_prev_time{0.0};
  bool m_have_prev_energy{false};
  pfc::data::Field<double> m_a;
  pfc::data::Field<double> m_j;
  std::vector<double> m_u, m_v, m_bx, m_by, m_wx, m_wy, m_jx, m_jy, m_ax, m_ay;
  std::vector<double> m_nw, m_na, m_div_u, m_div_b;
  std::vector<Complex> m_w_hat, m_a_hat, m_phi_hat, m_j_hat, m_sw, m_sa;
  std::vector<Complex> m_nw1, m_na1, m_nw2, m_na2, m_nw3, m_na3, m_nw4, m_na4;
  std::vector<double> m_ew_dt, m_ew_half, m_ea_dt, m_ea_half;
};

} // namespace ns2d
