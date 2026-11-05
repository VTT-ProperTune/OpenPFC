// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file elastic_coupled_host.hpp
 * @brief Host reference for one coherent equal-stiffness Cahn–Hilliard run.
 *
 * Same ETD1 update as `run_cahn_hilliard_elastic`, without that driver's
 * range projection. Inside (0, 1) the projection does not move the field.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cahn_hilliard/cahn_hilliard_pointwise.hpp>
#include <cahn_hilliard/elastic_coupled_step.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

namespace cahn_hilliard {

inline ElasticCoupledReport advance_coherent_elastic_host(
    const pfc::Domain &domain, const CahnHilliardParams &chemistry,
    const pfc::solvers::Stiffness &stiffness, double eps0, double c_ref,
    const std::vector<double> &initial, double dt, int steps, int rank, int nproc,
    MPI_Comm comm) {
  using Ops = pfc::sim::SpectralETDOps<pfc::HostSpace>;
  if (!(dt > 0.0) || steps < 0 || !std::isfinite(eps0) || eps0 < 0.0 ||
      !std::isfinite(c_ref)) {
    throw std::invalid_argument(
        "advance_coherent_elastic_host: require dt>0, steps>=0, eps0>=0");
  }

  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, comm);
  auto &c = stack.u();
  if (initial.size() != c.size()) {
    throw std::invalid_argument(
        "advance_coherent_elastic_host: initial size does not match the inbox");
  }
  std::copy(initial.begin(), initial.end(), c.data());
  c.note_host_write();

  const std::size_t n_in = stack.fft().size_inbox();
  const std::size_t n_out = stack.fft().size_outbox();
  if (n_in != c.size()) {
    throw std::invalid_argument(
        "advance_coherent_elastic_host: inbox halo is not zero");
  }
  const auto sp = c.spacing();
  const double cell = sp[0] * sp[1] * sp[2];

  pfc::data::Field<double> h(domain, stack.fft().get_inbox_bounds(), 0);
  pfc::data::Field<double> amp(domain, stack.fft().get_inbox_bounds(), 0);
  pfc::data::Field<double> dh(domain, stack.fft().get_inbox_bounds(), 0);
  pfc::data::Field<double> damp(domain, stack.fft().get_inbox_bounds(), 0);
  pfc::data::Field<double> n_real(domain, stack.fft().get_inbox_bounds(), 0);

  pfc::solvers::MicroelasticityParams mp;
  mp.stiffness_at_one = stiffness;
  mp.stiffness_at_zero = stiffness;
  mp.eigenstrain_pattern = pfc::solvers::Sym3{{eps0, eps0, eps0, 0.0, 0.0, 0.0}};
  mp.comm = comm;
  mp.warm_start = true;
  pfc::solvers::EigenstrainMicroelasticity solver(domain, stack.fft(), mp);

  Ops::ComplexField c_hat(domain, stack.fft().get_outbox_bounds(), 0);
  Ops::ComplexField n_hat(domain, stack.fft().get_outbox_bounds(), 0);
  auto candidate = Ops::make_complex(n_out);
  auto exp_Ldt = Ops::make_real(n_out);
  auto n_weight = Ops::make_real(n_out);
  auto mask = Ops::make_real(n_out);
  {
    std::vector<double> L(n_out), expv(n_out), w(n_out), msk(n_out, 1.0);
    pfc::fft::kspace::for_each_kpoint(
        stack.fft().get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          const double k_lap = pfc::fft::kspace::k_laplacian_value(kx, ky, kz);
          L[i] = chemistry.M *
                 (chemistry.fpp0 * k_lap - chemistry.kappa * k_lap * k_lap);
          const double Mnl = chemistry.M * k_lap;
          const auto coeff = pfc::integrator::spectral_exp_coeffs(L[i], dt);
          expv[i] = coeff.exp_Ldt;
          w[i] = Mnl * coeff.phi1_L;
        });
    pfc::fft::kspace::fill_two_thirds_mask(
        stack.fft().get_outbox_bounds(), pfc::domain::get_size(domain),
        pfc::domain::get_spacing(domain), msk.data(), msk.size());
    Ops::upload(exp_Ldt, expv);
    Ops::upload(n_weight, w);
    Ops::upload(mask, msk);
  }

  const CahnHilliardPointwise ch_pw{.omega_nd = chemistry.omega_nd,
                                    .l1_nd = chemistry.l1_nd,
                                    .c0 = chemistry.c0,
                                    .fprime0 = chemistry.fprime0,
                                    .fpp0 = chemistry.fpp0};

  auto elastic = [&] {
    for (std::size_t i = 0; i < n_in; ++i) {
      h.data()[i] = 1.0;
      dh.data()[i] = 0.0;
      damp.data()[i] = 1.0;
      amp.data()[i] = c.data()[i] - c_ref;
    }
    return solver.solve(h, amp, &dh, &damp);
  };

  ElasticCoupledReport report;
  report.mass0 = composition_mass(c.data(), n_in, cell, comm);
  for (int step = 0; step < steps; ++step) {
    elastic();
    const auto &mu = solver.elastic_energy_derivative();
    for (std::size_t i = 0; i < n_in; ++i) {
      n_real.data()[i] = ch_pw.n_nl(c.data()[i]) + mu.data()[i];
    }
    Ops::forward(stack.fft(), n_real, n_hat);
    Ops::forward(stack.fft(), c, c_hat);
    Ops::multiply(n_hat, mask, n_hat);
    Ops::combine(c_hat, n_hat, exp_Ldt, n_weight, candidate);
    Ops::swap(c_hat, candidate);
    Ops::backward(stack.fft(), c_hat, c);
  }

  const auto rep = elastic();
  const auto &mu = solver.elastic_energy_derivative();
  double max_mu = 0.0;
  for (std::size_t i = 0; i < n_in; ++i) {
    max_mu = std::max(max_mu, std::abs(mu.data()[i]));
  }
  double max_mu_global = 0.0;
  MPI_Allreduce(&max_mu, &max_mu_global, 1, MPI_DOUBLE, MPI_MAX, comm);
  report.elastic_energy = solver.total_elastic_energy();
  report.max_abs_mu_el = max_mu_global;
  report.iterations = rep.iterations;
  report.residual = rep.residual;
  report.converged = rep.converged;
  report.mass = composition_mass(c.data(), n_in, cell, comm);
  report.composition.assign(c.data(), c.data() + static_cast<std::ptrdiff_t>(n_in));
  return report;
}

} // namespace cahn_hilliard
