// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file elastic_driver.hpp
 * @brief Cahn–Hilliard with coherent Vegard eigenstrain on the existing Green
 *        operator.
 *
 * @details
 * Not a separate application: the same Fe–Cr Cahn–Hilliard model, plus
 * the Khachaturyan Green solve already used by the dendrite and by inverse
 * homogenization. Both phases are treated as solids of equal stiffness
 * (a coherent α/α′ pair, not a solid–liquid contrast), so the modulus is
 * homogeneous and the solve is one Γ application per step.
 *
 * \f[
 *   \varepsilon^{*}_{ij} = \varepsilon_0\,(c-c_{\mathrm{ref}})\,\delta_{ij},
 *   \qquad
 *   \mu_{\mathrm{el}} = \frac{\partial f_{\mathrm{el}}}{\partial c},
 *   \qquad
 *   \partial_t c = M\nabla^2\bigl(f'(c)-\kappa\nabla^2 c+\mu_{\mathrm{el}}\bigr).
 * \f]
 *
 * Stiffnesses are converted from Pa into the CH energy unit \f$RT/V_m\f$
 * so \f$\mu_{\mathrm{el}}\f$ adds to \f$f'(c)\f$ with no extra scale.
 * Host-only: the Green operator has no device path (#157).
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <cahn_hilliard/cahn_hilliard_physics.hpp>
#include <cahn_hilliard/concentration_seed.hpp>
#include <cahn_hilliard/diagnostics.hpp>
#include <cahn_hilliard/fe_cr_thermo.hpp>
#include <openpfc/frontend/io/diagnostics_series.hpp>
#include <openpfc/frontend/ui/field_modifier_registry.hpp>
#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/frontend/ui/json_snapshot_fields.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/solvers/microelasticity/microelasticity.hpp>

namespace cahn_hilliard {

struct ElasticCHParams {
  double eps0{0.04};  ///< Vegard slope; illustrative, not a fitted da/dc
  double c_ref{-1.0}; ///< eigenstrain zero; <0 means use the alloy c0
  double E{2.0e11};   ///< Young's modulus (Pa); used if c11==0
  double nu{0.3};
  double c11{0.0}; ///< cubic C11 (Pa); nonzero selects cubic over E,ν
  double c12{0.0};
  double c44{0.0};
};

inline ElasticCHParams parse_elasticity(const nlohmann::json &cfg) {
  ElasticCHParams p;
  if (!cfg.contains("elasticity")) return p;
  const auto &e = cfg.at("elasticity");
  p.eps0 = e.value("eps0", p.eps0);
  p.c_ref = e.value("c_ref", p.c_ref);
  p.E = e.value("E", p.E);
  p.nu = e.value("nu", p.nu);
  p.c11 = e.value("c11", p.c11);
  p.c12 = e.value("c12", p.c12);
  p.c44 = e.value("c44", p.c44);
  if (!(p.eps0 >= 0.0) || !std::isfinite(p.eps0))
    throw std::invalid_argument("elasticity.eps0 must be >= 0");
  return p;
}

/// Convert a physical cubic/isotropic stiffness (Pa) into CH energy units.
inline pfc::solvers::Stiffness stiffness_in_rt_vm(const ElasticCHParams &el,
                                                  const CahnHilliardParams &ch) {
  const double f0 = ch.scales().f0(ch.T);
  if (!(f0 > 0.0)) throw std::invalid_argument("RT/Vm energy scale must be > 0");
  if (el.c11 > 0.0) {
    return pfc::solvers::Stiffness::cubic(el.c11 / f0, el.c12 / f0, el.c44 / f0);
  }
  return pfc::solvers::Stiffness::isotropic(el.E / f0, el.nu);
}

inline void apply_initial_condition(const nlohmann::json &cfg, pfc::Domain domain,
                                    pfc::data::Field<double> &c, MPI_Comm comm) {
  if (!cfg.contains("initial_conditions") || cfg.at("initial_conditions").empty())
    throw std::invalid_argument(
        "cahn_hilliard_elastic: initial_conditions required");
  const pfc::SimulationContext ctx(comm);
  for (const auto &raw : cfg.at("initial_conditions")) {
    nlohmann::json ic = raw;
    pfc::ui::copy_json_alias(ic, "c0", "offset");
    require_concentration_noise(ic);
    const std::string type = ic.at("type").get<std::string>();
    auto modifier = pfc::ui::create_field_modifier(type, ic);
    modifier->apply(ctx, c.output(), domain, c.box(), 0.0);
  }
}

/**
 * @brief JSON → coherent elastic Cahn–Hilliard on the host spectral stack.
 *
 * Prints `CAHN_HILLIARD_ELASTIC ...` on rank 0. Optional `fields[]` and
 * `diagnostics.csv` match the other science drivers.
 */
inline int run_cahn_hilliard_elastic(int rank, int nproc, MPI_Comm comm,
                                     const std::string &json_path) {
  using Ops = pfc::sim::SpectralETDOps<pfc::HostSpace>;
  using json = nlohmann::json;

  int status = 0;
  try {
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("cannot open " + json_path);
    json cfg = json::parse(in);

    const auto domain = pfc::ui::from_json<pfc::Domain>(cfg);

    CahnHilliardParams ch;
    if (cfg.contains("model") && cfg.at("model").contains("params"))
      apply_cahn_hilliard_json(cfg.at("model").at("params"), ch);
    else
      ch.recompute_derived();
    const auto el = parse_elasticity(cfg);
    const double c_ref = (el.c_ref >= 0.0) ? el.c_ref : ch.c0;
    const auto C = stiffness_in_rt_vm(el, ch);

    const auto &ts = cfg.at("timestepping");
    const double t1 = ts.at("t1").get<double>();
    const double dt = ts.at("dt").get<double>();
    const double saveat = ts.value("saveat", -1.0);
    if (!(dt > 0.0) || !(t1 >= 0.0))
      throw std::invalid_argument("timestepping: require dt>0 and t1>=0");

    pfc::sim::stacks::SpectralCPUStack stack(domain, rank, nproc, comm);
    auto &c = stack.u();
    apply_initial_condition(cfg, domain, c, comm);

    const std::size_t n_in = stack.fft().size_inbox();
    const std::size_t n_out = stack.fft().size_outbox();
    pfc::data::Field<double> h(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<double> amp(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<double> dh(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<double> damp(domain, stack.fft().get_inbox_bounds(), 0);
    pfc::data::Field<double> n_real(domain, stack.fft().get_inbox_bounds(), 0);
    std::fill(h.vec().begin(), h.vec().end(), 1.0);
    std::fill(dh.vec().begin(), dh.vec().end(), 0.0);
    std::fill(damp.vec().begin(), damp.vec().end(), 1.0);

    pfc::solvers::MicroelasticityParams mp;
    mp.stiffness_at_one = C;
    mp.stiffness_at_zero = C;
    mp.eigenstrain_pattern =
        pfc::solvers::Sym3{{el.eps0, el.eps0, el.eps0, 0.0, 0.0, 0.0}};
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
      std::vector<double> L(n_out), Mnl(n_out), expv(n_out), w(n_out),
          msk(n_out, 1.0);
      pfc::fft::kspace::for_each_kpoint(
          stack.fft().get_outbox_bounds(), domain,
          [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
            const double k_lap = pfc::fft::kspace::k_laplacian_value(kx, ky, kz);
            L[i] = ch.M * (ch.fpp0 * k_lap - ch.kappa * k_lap * k_lap);
            Mnl[i] = ch.M * k_lap;
            const auto coeff = pfc::integrator::spectral_exp_coeffs(L[i], dt);
            expv[i] = coeff.exp_Ldt;
            w[i] = Mnl[i] * coeff.phi1_L;
          });
      pfc::fft::kspace::fill_two_thirds_mask(
          stack.fft().get_outbox_bounds(), pfc::domain::get_size(domain),
          pfc::domain::get_spacing(domain), msk.data(), msk.size());
      Ops::upload(exp_Ldt, expv);
      Ops::upload(n_weight, w);
      Ops::upload(mask, msk);
    }

    const CahnHilliardPointwise ch_pw{.omega_nd = ch.omega_nd,
                                      .l1_nd = ch.l1_nd,
                                      .c0 = ch.c0,
                                      .fprime0 = ch.fprime0,
                                      .fpp0 = ch.fpp0};

    Diagnostics<pfc::HostSpace> diagnostics(domain, stack.fft(), comm);
    pfc::io::SnapshotSeries snapshots(c.domain(), c.box(),
                                      pfc::io::SnapshotSeriesOptions{.comm = comm});
    pfc::ui::bind_snapshot_field(snapshots, cfg, "c", c);
    pfc::ui::finish_snapshot_fields(snapshots, cfg);
    std::optional<pfc::io::DiagnosticsSeries> diag;
    if (cfg.contains("diagnostics")) {
      const std::filesystem::path path =
          cfg.at("diagnostics").at("csv").get<std::string>();
      diag.emplace(
          std::vector<std::string>{"mean", "mass", "min", "max", "bulk_energy",
                                   "gradient_energy", "total_energy", "el_energy",
                                   "el_iterations", "max_abs_mu_el", "k1",
                                   "domain_length", "k_peak", "dominant_wavelength",
                                   "axis_power", "diag_power"},
          pfc::io::DiagnosticsSeriesOptions{
              .path = path, .comm = comm, .overwrite = true});
    }

    auto assemble_amp = [&] {
      for (std::size_t i = 0; i < n_in; ++i) amp.data()[i] = c.data()[i] - c_ref;
    };

    auto project_c = [&] {
      // ETD1 treats N explicitly; once interfaces sharpen, a step can leave
      // (0,1) and the log potential explodes. Project onto [lo,hi] and
      // restore the mean so the scheme stays conservative.
      constexpr double lo = 1.0e-8, hi = 1.0 - 1.0e-8;
      const double mean0 = [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < n_in; ++i) s += c.data()[i];
        double g = 0.0;
        MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
        const auto n = pfc::domain::get_size(domain);
        return g / (double(n[0]) * n[1] * n[2]);
      }();
      for (int it = 0; it < 8; ++it) {
        for (std::size_t i = 0; i < n_in; ++i) {
          double v = c.data()[i];
          if (v < lo) v = lo;
          if (v > hi) v = hi;
          c.data()[i] = v;
        }
        double s = 0.0;
        for (std::size_t i = 0; i < n_in; ++i) s += c.data()[i];
        double g = 0.0;
        MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
        const auto n = pfc::domain::get_size(domain);
        const double mean1 = g / (double(n[0]) * n[1] * n[2]);
        const double shift = mean0 - mean1;
        if (std::abs(shift) < 1.0e-16) break;
        for (std::size_t i = 0; i < n_in; ++i) c.data()[i] += shift;
      }
      for (std::size_t i = 0; i < n_in; ++i) {
        double v = c.data()[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        c.data()[i] = v;
      }
    };

    auto axis_vs_diag = [&]() -> std::pair<double, double> {
      Ops::forward(stack.fft(), c, c_hat);
      double loc[2]{};
      pfc::fft::kspace::for_each_kpoint(
          stack.fft().get_outbox_bounds(), domain,
          [&](std::size_t i, double kx, double ky, double, int ix, int, int) {
            const double p =
                pfc::fft::r2c_multiplicity(ix, pfc::domain::get_size(domain)[0]) *
                std::norm(c_hat.data()[i]);
            const double akx = std::abs(kx), aky = std::abs(ky);
            if (akx + aky < 1.0e-15) return;
            if (akx > 2.0 * aky || aky > 2.0 * akx)
              loc[0] += p;
            else
              loc[1] += p;
          });
      double glob[2]{};
      MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
      const double tot = glob[0] + glob[1];
      if (tot <= 0.0) return {0.0, 0.0};
      return {glob[0] / tot, glob[1] / tot};
    };

    auto report = [&](int step, double t, int el_iters, double max_mu) {
      snapshots.write(step, t);
      if (!diag) return;
      auto s = diagnostics.sample(c, ch);
      const auto [p_axis, p_diag] = axis_vs_diag();
      const double el_e = solver.total_elastic_energy();
      diag->write(step, t,
                  {s.mean, s.mass, s.minimum, s.maximum, s.bulk_energy,
                   s.gradient_energy, s.total_energy() + el_e, el_e,
                   static_cast<double>(el_iters), max_mu, s.k1, s.domain_length,
                   s.k_peak, s.dominant_wavelength, p_axis, p_diag});
    };

    const int n_steps = static_cast<int>(std::llround(t1 / dt));
    const int every = (saveat > 0.0)
                          ? std::max(1, static_cast<int>(std::llround(saveat / dt)))
                          : n_steps;

    auto elastic_step = [&]() -> std::pair<int, double> {
      assemble_amp();
      const auto rep = solver.solve(h, amp, &dh, &damp);
      const auto &mu = solver.elastic_energy_derivative();
      double max_mu = 0.0;
      for (std::size_t i = 0; i < n_in; ++i) {
        const double m = mu.data()[i];
        max_mu = std::max(max_mu, std::abs(m));
        n_real.data()[i] = ch_pw.n_nl(c.data()[i]) + m;
      }
      return {rep.iterations, max_mu};
    };

    auto [it0, mu0] = elastic_step();
    report(0, 0.0, it0, mu0);

    double t = 0.0;
    for (int step = 1; step <= n_steps; ++step) {
      auto [iters, max_mu] = elastic_step();
      Ops::forward(stack.fft(), n_real, n_hat);
      Ops::forward(stack.fft(), c, c_hat);
      Ops::multiply(n_hat, mask, n_hat);
      Ops::combine(c_hat, n_hat, exp_Ldt, n_weight, candidate);
      Ops::swap(c_hat, candidate);
      Ops::backward(stack.fft(), c_hat, c);
      project_c();
      t = dt * static_cast<double>(step);
      if (step % every == 0 || step == n_steps) report(step, t, iters, max_mu);
    }

    if (rank == 0) {
      auto s = diagnostics.sample(c, ch);
      const auto [p_axis, p_diag] = axis_vs_diag();
      std::printf("CAHN_HILLIARD_ELASTIC mean=%.17g min=%.17g max=%.17g "
                  "el_energy=%.17g domain_length=%.17g eps0=%.17g "
                  "axis_power=%.17g diag_power=%.17g\n",
                  s.mean, s.minimum, s.maximum, solver.total_elastic_energy(),
                  s.domain_length, el.eps0, p_axis, p_diag);
    } else {
      (void)diagnostics.sample(c, ch);
      (void)solver.total_elastic_energy();
    }
    if (diag) diag->close();
    snapshots.close();
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "cahn_hilliard_elastic: " << e.what() << "\n";
    status = 2;
  }
  return status;
}

} // namespace cahn_hilliard
