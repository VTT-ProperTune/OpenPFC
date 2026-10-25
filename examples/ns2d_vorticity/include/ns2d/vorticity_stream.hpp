// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file vorticity_stream.hpp
 * @brief Periodic 2-D vorticity–streamfunction Navier–Stokes on the CPU
 *        spectral stack (OpenPFC issues #21 / #22).
 *
 * Uses the shared \ref SpectralPlane for Poisson, odd-\f$k\f$ derivatives,
 * 2/3 projection and IFRK4. See spectral.hpp for the sign convention
 * \f$\omega=-\nabla^2\phi\f$, \f$\mathbf u=(\partial_y\phi,-\partial_x\phi)\f$.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#include <ns2d/spectral.hpp>

namespace ns2d {

struct Params {
  double nu{0.1};
  double dt{0.01};
};

struct Diagnostics {
  double time{0.0};
  double ke{0.0};
  double enstrophy{0.0};
  double max_abs_omega{0.0};
  double max_speed{0.0};
  double div_linf{0.0};
  double div_l2{0.0};
  double cfl{0.0};
  double mean_omega{0.0};
};

class VorticityStreamCPU {
public:
  using Complex = std::complex<double>;

  VorticityStreamCPU(pfc::sim::stacks::SpectralCPUStack &stack, Params params)
      : m_stack(&stack), m_plane(stack.fft(), stack.u()), m_params(params) {
    if (params.nu < 0.0 || params.dt <= 0.0) {
      throw std::invalid_argument("ns2d: nu >= 0 and dt > 0 are required");
    }
    const std::size_t in_n = m_plane.in_n();
    const std::size_t out_n = m_plane.out_n();
    m_u.assign(in_n, 0.0);
    m_v.assign(in_n, 0.0);
    m_wx.assign(in_n, 0.0);
    m_wy.assign(in_n, 0.0);
    m_n.assign(in_n, 0.0);
    m_div.assign(in_n, 0.0);
    m_omega_hat.assign(out_n, Complex{});
    m_psi_hat.assign(out_n, Complex{});
    m_stage_hat.assign(out_n, Complex{});
    m_n1_hat.assign(out_n, Complex{});
    m_n2_hat.assign(out_n, Complex{});
    m_n3_hat.assign(out_n, Complex{});
    m_n4_hat.assign(out_n, Complex{});
    m_plane.fill_exp(m_params.nu, m_params.dt, m_exp_Ldt, m_exp_half);
  }

  [[nodiscard]] Params params() const noexcept { return m_params; }
  [[nodiscard]] double time() const noexcept { return m_time; }
  [[nodiscard]] pfc::data::Field<double> &omega() noexcept { return m_stack->u(); }
  [[nodiscard]] const pfc::data::Field<double> &omega() const noexcept {
    return m_stack->u();
  }
  [[nodiscard]] const std::vector<double> &u() const noexcept { return m_u; }
  [[nodiscard]] const std::vector<double> &v() const noexcept { return m_v; }
  [[nodiscard]] const std::vector<double> &psi_inv_k2() const noexcept {
    return m_plane.inv_k2();
  }
  [[nodiscard]] const std::vector<double> &dealias_mask() const noexcept {
    return m_plane.mask();
  }
  [[nodiscard]] const std::vector<Complex> &last_n_hat() const noexcept {
    return m_n1_hat;
  }
  [[nodiscard]] bool rank_owns_zero_mode() const noexcept {
    return m_plane.rank_owns_zero_mode();
  }
  [[nodiscard]] SpectralPlane &plane() noexcept { return m_plane; }

  void project_hat(std::vector<Complex> &hat) const { m_plane.project_hat(hat); }

  template <class Fn> void initialize_omega(Fn &&fn) {
    m_stack->u().apply(std::forward<Fn>(fn));
    m_time = 0.0;
    project_current_state();
    recover_velocity_from_omega();
  }

  void project_current_state() {
    auto &w = m_stack->u();
    m_plane.fft().forward(w.vec(), m_omega_hat);
    m_plane.project_hat(m_omega_hat);
    m_plane.fft().backward(m_omega_hat, w.vec());
  }

  /// Current vorticity coefficients after the 2/3 projection.
  void copy_omega_hat(std::vector<Complex> &hat) {
    hat.resize(m_plane.out_n());
    m_plane.fft().forward(m_stack->u().vec(), hat);
    m_plane.project_hat(hat);
  }

  void jacobian_from_current(bool project_inputs) {
    m_plane.fft().forward(m_stack->u().vec(), m_omega_hat);
    if (project_inputs) m_plane.project_hat(m_omega_hat);
    nonlinear_from_hat(m_omega_hat, m_n1_hat);
  }

  void step() {
    auto &w = m_stack->u();
    m_plane.fft().forward(w.vec(), m_omega_hat);
    ifrk4_one(
        m_plane, m_omega_hat, m_exp_Ldt, m_exp_half, m_params.dt,
        [&](const std::vector<Complex> &uh, std::vector<Complex> &nh) {
          nonlinear_from_hat(uh, nh);
        },
        m_n1_hat, m_n2_hat, m_n3_hat, m_n4_hat, m_stage_hat);
    m_plane.fft().backward(m_omega_hat, w.vec());
    m_time += m_params.dt;
  }

  void recover_velocity_from_omega() {
    m_plane.fft().forward(m_stack->u().vec(), m_omega_hat);
    m_plane.project_hat(m_omega_hat);
    m_plane.poisson(m_omega_hat, m_psi_hat);
    m_plane.curl_from_hat(m_psi_hat, m_u, m_v);
  }

  [[nodiscard]] Diagnostics diagnostics(MPI_Comm comm) {
    recover_velocity_from_omega();
    m_plane.spectral_div(m_u, m_v, m_div);

    double ke_sum = 0.0, ens_sum = 0.0, w_sum = 0.0, div_sq = 0.0;
    double max_w = 0.0, max_speed = 0.0, div_linf = 0.0;
    const auto &wfield = m_stack->u();
    wfield.for_each_owned([&](int i, int j, int k) {
      const std::size_t c = m_plane.real_idx(i, j, k);
      const double w = wfield(i, j, k);
      const double uu = m_u[c];
      const double vv = m_v[c];
      ke_sum += 0.5 * (uu * uu + vv * vv);
      ens_sum += 0.5 * w * w;
      w_sum += w;
      max_w = std::max(max_w, std::abs(w));
      max_speed = std::max(max_speed, std::max(std::abs(uu), std::abs(vv)));
      div_linf = std::max(div_linf, std::abs(m_div[c]));
      div_sq += m_div[c] * m_div[c];
    });

    double g_ke = 0.0, g_ens = 0.0, g_w = 0.0, g_divsq = 0.0;
    double g_maxw = 0.0, g_maxspd = 0.0, g_divinf = 0.0;
    MPI_Allreduce(&ke_sum, &g_ke, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&ens_sum, &g_ens, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&w_sum, &g_w, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&div_sq, &g_divsq, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&max_w, &g_maxw, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&max_speed, &g_maxspd, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&div_linf, &g_divinf, 1, MPI_DOUBLE, MPI_MAX, comm);

    const auto gs = m_plane.gsize();
    const double ncells = static_cast<double>(gs[0]) * gs[1] * gs[2];
    const auto sp = m_plane.spacing();
    const double dx = std::min(sp[0], sp[1]);
    Diagnostics d;
    d.time = m_time;
    d.ke = g_ke / ncells;
    d.enstrophy = g_ens / ncells;
    d.max_abs_omega = g_maxw;
    d.max_speed = g_maxspd;
    d.div_linf = g_divinf;
    d.div_l2 = std::sqrt(g_divsq / ncells);
    d.cfl = m_params.dt * g_maxspd / dx;
    d.mean_omega = g_w / ncells;
    return d;
  }

  [[nodiscard]] double linf_omega_error(MPI_Comm comm, const auto &exact) const {
    double local = 0.0;
    m_stack->u().for_each_owned([&](double x, double y, double /*z*/, double w) {
      local = std::max(local, std::abs(w - exact(x, y)));
    });
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global;
  }

private:
  void nonlinear_from_hat(const std::vector<Complex> &omega_hat,
                          std::vector<Complex> &n_hat) {
    m_plane.poisson(omega_hat, m_psi_hat);
    m_plane.curl_from_hat(m_psi_hat, m_u, m_v);
    m_plane.invert_ik(omega_hat, m_plane.kx_odd(), +1.0, m_wx);
    m_plane.invert_ik(omega_hat, m_plane.ky_odd(), +1.0, m_wy);
    for (std::size_t i = 0; i < m_n.size(); ++i) {
      m_n[i] = -(m_u[i] * m_wx[i] + m_v[i] * m_wy[i]);
    }
    m_plane.fft().forward(m_n, n_hat);
    m_plane.project_hat(n_hat);
    m_plane.zero_dc(n_hat);
  }

  pfc::sim::stacks::SpectralCPUStack *m_stack{nullptr};
  SpectralPlane m_plane;
  Params m_params{};
  double m_time{0.0};
  std::vector<double> m_u, m_v, m_wx, m_wy, m_n, m_div;
  std::vector<Complex> m_omega_hat, m_psi_hat, m_stage_hat;
  std::vector<Complex> m_n1_hat, m_n2_hat, m_n3_hat, m_n4_hat;
  std::vector<double> m_exp_Ldt, m_exp_half;
};

} // namespace ns2d
