// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_hip_growth.cpp
 * @brief Coupled GPU science driver: DeviceStepper plus the device Green
 *        operator. Same application as `alloy_dendrite_growth`.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "alloy_dendrite_hip_growth requires HIP spectral"
#endif

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>

#include <alloy_dendrite/cli.hpp>
#include <alloy_dendrite/device_elasticity_hip.hpp>
#include <alloy_dendrite/device_stepper_hip.hpp>
#include <alloy_dendrite/diagnostics.hpp>
#include <alloy_dendrite/elasticity.hpp>
#include <alloy_dendrite/parameters.hpp>

namespace {

using DevField = pfc::data::Field<double, pfc::HIPSpace>;
using RealField = pfc::data::Field<double>;
using pfc::apps::Stiffness;

struct Cfg {
  alloy_dendrite::ModelParams model{};
  int nx = 128;
  int ny = 128;
  int nz = 1;
  double dx = 0.8;
  int fd_order = 4;
  int steps = 2000;
  int sample_every = 100;
  double dt = 0.0;
  double dt_safety = 0.2;
  double omega = 0.55;
  double seed_radius = 10.0;
  bool elastic = false;
  int n_el_substep = 1;
  bool warm_start = true;
  double tol_el = 1.0e-6;
  int n_el_iter = 50;
  double youngs = 1.0;
  double poisson = 0.3;
  double liquid_shear = pfc::apps::kDefaultLiquidShearFraction;
  double eps_c = 0.01;
  double eps_T = 0.0;
  double u_ref = 0.0;
  double lambda_el = 0.0;
  std::string csv;
  std::string run_id = "hip-growth";
};

void push_seed(DevField &dst, const RealField &src) {
  const auto n = dst.local_size();
  const int hw = dst.storage_halo();
  const std::size_t npx = static_cast<std::size_t>(n[0] + 2 * hw);
  const std::size_t npy = static_cast<std::size_t>(n[1] + 2 * hw);
  dst.with_host_view([&](double *data, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          data[(static_cast<std::size_t>(i) + hw) +
               (static_cast<std::size_t>(j) + hw) * npx +
               (static_cast<std::size_t>(k) + hw) * npx * npy] = src(i, j, k);
  });
  dst.sync_to_device();
}

int run(const Cfg &cfg, int rank, int nproc, MPI_Comm comm) {
  const auto &p = cfg.model;
  const int dim = (cfg.nz == 1) ? 2 : 3;
  const double dt_lim =
      alloy_dendrite::explicit_dt_limit(p, cfg.dx, dim, cfg.fd_order);
  const double dt = (cfg.dt > 0.0) ? cfg.dt : cfg.dt_safety * dt_lim;

  const auto domain = pfc::domain::create(
      pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
  const auto fft_grid =
      pfc::decomposition::spectral_fft_proc_grid(domain.size, nproc);
  auto decomp = pfc::decomposition::create(domain, fft_grid);

  std::unique_ptr<alloy_dendrite::DeviceStepper<2>> d2;
  std::unique_ptr<alloy_dendrite::DeviceStepper<3>> d3;
  if (cfg.nz == 1) {
    d2 = std::make_unique<alloy_dendrite::DeviceStepper<2>>(domain, decomp, rank,
                                                           comm, p, cfg.fd_order);
  } else {
    d3 = std::make_unique<alloy_dendrite::DeviceStepper<3>>(domain, decomp, rank,
                                                           comm, p, cfg.fd_order);
  }
  auto &phi = d2 ? d2->phi() : d3->phi();
  auto &U = d2 ? d2->solute() : d3->solute();
  auto &th = d2 ? d2->temperature() : d3->temperature();
  const auto &geom = d2 ? d2->geom() : d3->geom();

  const double xc = 0.5 * cfg.nx * cfg.dx;
  const double yc = 0.5 * cfg.ny * cfg.dx;
  const double zc = 0.5 * cfg.nz * cfg.dx;
  const double inv_w = 1.0 / (std::sqrt(2.0) * p.W0);
  RealField seed = pfc::data::field_from_inbox<double>(
      domain, pfc::decomposition::local_box(decomp, rank));
  seed.for_each_owned([&](int i, int j, int k) {
    const auto c = seed.coords(i, j, k);
    const double dz = (cfg.nz == 1) ? 0.0 : (c[2] - zc);
    const double r = std::sqrt((c[0] - xc) * (c[0] - xc) + (c[1] - yc) * (c[1] - yc) +
                               dz * dz);
    seed(i, j, k) = std::tanh((cfg.seed_radius * p.W0 - r) * inv_w);
  });
  push_seed(phi, seed);
  seed.for_each_owned([&](int i, int j, int k) { seed(i, j, k) = -cfg.omega; });
  push_seed(U, seed);
  seed.for_each_owned([&](int i, int j, int k) { seed(i, j, k) = 0.0; });
  push_seed(th, seed);
  if (d2) d2->seed_conserved_solute();
  else d3->seed_conserved_solute();

  alloy_dendrite::ElasticParams ep;
  ep.c_solid = Stiffness::isotropic(cfg.youngs, cfg.poisson);
  ep.mu_liquid_fraction = cfg.liquid_shear;
  ep.eps_c = cfg.eps_c;
  ep.eps_T = cfg.eps_T;
  ep.U_ref = cfg.u_ref;
  ep.tol_el = cfg.tol_el;
  ep.n_el_iter = cfg.n_el_iter;
  ep.n_el_substep = cfg.n_el_substep;
  ep.warm_start = cfg.warm_start;

  std::unique_ptr<alloy_dendrite::DeviceElasticCoupling> elastic;
  alloy_dendrite::ElasticReport el{};
  if (cfg.elastic) {
    elastic = std::make_unique<alloy_dendrite::DeviceElasticCoupling>(
        domain, decomp, rank, comm, ep, geom);
    if (d2) d2->set_elastic_driving_force(&elastic->driving_force());
    else d3->set_elastic_driving_force(&elastic->driving_force());
    el = elastic->solve(phi, U, th);
    if (hipDeviceSynchronize() != hipSuccess) {
      throw std::runtime_error("hipDeviceSynchronize failed after first solve");
    }
  }

  alloy_dendrite::CsvAppender csv;
  if (!cfg.csv.empty()) {
    csv = alloy_dendrite::CsvAppender(
        cfg.csv,
        "run_id,step,t,t_pf_ms,t_el_ms,el_iters,el_energy,el_residual,"
        "el_max_dfel,mean_p,sig_vm_max,x_tip,v_tip,rho_tip",
        rank);
  }

  double t_pf_sum = 0.0, t_el_sum = 0.0;
  int n_el = 0;
  double x_prev = 0.0;
  bool have_tip = false;

  for (int step = 1; step <= cfg.steps; ++step) {
    MPI_Barrier(comm);
    double a = MPI_Wtime();
    if (d2) d2->step(dt);
    else d3->step(dt);
    if (hipDeviceSynchronize() != hipSuccess) {
      throw std::runtime_error("hipDeviceSynchronize failed after step");
    }
    double b = MPI_Wtime();
    const double tpf = b - a;
    t_pf_sum += tpf;

    double tel = 0.0;
    if (elastic && elastic->due(step)) {
      a = MPI_Wtime();
      el = elastic->solve(phi, U, th);
      if (hipDeviceSynchronize() != hipSuccess) {
        throw std::runtime_error("hipDeviceSynchronize failed after elastic");
      }
      b = MPI_Wtime();
      tel = b - a;
      t_el_sum += tel;
      ++n_el;
    }

    if (step % cfg.sample_every == 0 || step == cfg.steps) {
      double x_tip = 0.0, v_tip = 0.0, rho = 0.0;
      if (nproc == 1 && cfg.nz == 1) {
        std::vector<double> plane(static_cast<std::size_t>(cfg.nx) * cfg.ny);
        phi.with_host_view([&](const double *data, std::size_t) {
          const int hw = phi.storage_halo();
          const auto n = phi.local_size();
          const std::size_t npx = static_cast<std::size_t>(n[0] + 2 * hw);
          const std::size_t npy = static_cast<std::size_t>(n[1] + 2 * hw);
          const std::size_t k0 =
              static_cast<std::size_t>(hw) * npx * npy;
          for (int j = 0; j < n[1]; ++j)
            for (int i = 0; i < n[0]; ++i)
              plane[static_cast<std::size_t>(i) +
                    static_cast<std::size_t>(j) * cfg.nx] =
                  data[(static_cast<std::size_t>(i) + hw) +
                       (static_cast<std::size_t>(j) + hw) * npx + k0];
        });
        phi.note_device_write(); // a read, not a hand-over
        const auto tip = alloy_dendrite::measure_tip(
            plane, cfg.nx, cfg.ny, cfg.dx, cfg.dx, cfg.nx / 2, cfg.ny / 2, 8);
        x_tip = tip.x_tip;
        rho = tip.rho;
        if (have_tip) v_tip = (x_tip - x_prev) / (cfg.sample_every * dt);
        x_prev = x_tip;
        have_tip = true;
      }
      if (csv.active()) {
        csv.row(alloy_dendrite::format(
            "%s,%d,%.6f,%.6f,%.6f,%d,%.6e,%.3e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f",
            cfg.run_id.c_str(), step, step * dt, 1e3 * tpf, 1e3 * tel,
            el.iterations, el.total_energy, el.residual, el.max_dfel_dphi,
            el.mean_stress_trace, elastic ? elastic->max_von_mises() : 0.0,
            x_tip, v_tip, rho));
      }
    }
  }

  if (rank == 0) {
    std::cout << std::setprecision(6);
    std::cout << "ALLOY_HIP_GROWTH run_id=" << cfg.run_id << " nx=" << cfg.nx
              << " ny=" << cfg.ny << " nz=" << cfg.nz << " steps=" << cfg.steps
              << " elastic=" << (cfg.elastic ? 1 : 0) << " eps_c=" << cfg.eps_c
              << " eps_T=" << cfg.eps_T << " ranks=" << nproc << "\n";
    std::cout << "ALLOY_HIP_GROWTH_MS t_pf_mean=" << 1e3 * t_pf_sum / cfg.steps
              << " t_el_mean=" << (n_el > 0 ? 1e3 * t_el_sum / n_el : 0.0)
              << " el_solves=" << n_el << " last_iters=" << el.iterations
              << " last_energy=" << el.total_energy
              << " last_mean_p=" << el.mean_stress_trace
              << " last_max_dfel=" << el.max_dfel_dphi << "\n";
  }
  return EXIT_SUCCESS;
}

int run_cli(int argc, char **argv, int rank, int nproc) {
  alloy_dendrite::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) {
      std::cout << "Usage: " << argv[0]
                << " [--nx= --ny= --nz= --elastic=0|1 --eps-c= --eps-T= ...]\n";
    }
    return EXIT_SUCCESS;
  }
  Cfg cfg;
  cfg.model.eps4 = 0.05;
  cfg.nx = opt.integer("nx", cfg.nx);
  cfg.ny = opt.integer("ny", cfg.ny);
  cfg.nz = opt.integer("nz", cfg.nz);
  cfg.dx = opt.real("dx", cfg.dx);
  cfg.fd_order = opt.integer("fd-order", cfg.fd_order);
  cfg.steps = opt.integer("steps", cfg.steps);
  cfg.sample_every = opt.integer("sample-every", cfg.sample_every);
  cfg.dt = opt.real("dt", cfg.dt);
  cfg.dt_safety = opt.real("dt-safety", cfg.dt_safety);
  cfg.omega = opt.real("omega", cfg.omega);
  cfg.seed_radius = opt.real("seed-radius", cfg.seed_radius);
  cfg.model.lambda = opt.real("lambda", cfg.model.lambda);
  cfg.model.k = opt.real("k", cfg.model.k);
  cfg.model.D_l = opt.real("Dl", cfg.model.D_l);
  cfg.model.D_th = opt.real("Dth", cfg.model.D_th);
  cfg.model.M_c = opt.real("Mc", cfg.model.M_c);
  cfg.model.eps4 = opt.real("eps4", cfg.model.eps4);
  cfg.model.evolve_theta = opt.flag("evolve-theta", cfg.model.evolve_theta);
  cfg.elastic = opt.flag("elastic", cfg.elastic);
  cfg.n_el_substep = opt.integer("n-el-substep", cfg.n_el_substep);
  cfg.warm_start = opt.flag("warm-start", cfg.warm_start);
  cfg.tol_el = opt.real("tol-el", cfg.tol_el);
  cfg.n_el_iter = opt.integer("n-el-iter", cfg.n_el_iter);
  cfg.youngs = opt.real("youngs", cfg.youngs);
  cfg.poisson = opt.real("poisson", cfg.poisson);
  cfg.liquid_shear = opt.real("liquid-shear", cfg.liquid_shear);
  cfg.eps_c = opt.real("eps-c", cfg.eps_c);
  cfg.eps_T = opt.real("eps-T", cfg.eps_T);
  cfg.u_ref = opt.real("u-ref", cfg.u_ref);
  cfg.lambda_el = opt.real("lambda-el", cfg.lambda_el);
  cfg.model.lambda_el = cfg.lambda_el;
  cfg.csv = opt.text("csv", cfg.csv);
  cfg.run_id = opt.text("run-id", cfg.run_id);
  opt.require_all_consumed();
  pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);
  return run(cfg, rank, nproc, MPI_COMM_WORLD);
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        try {
          return run_cli(app_argc, app_argv, rank, nproc);
        } catch (const std::exception &e) {
          if (rank == 0) std::cerr << "alloy_dendrite_hip_growth: " << e.what() << "\n";
          return EXIT_FAILURE;
        }
      });
}
