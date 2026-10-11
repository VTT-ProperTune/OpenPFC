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
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
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

#include <alloy_dendrite/device_elasticity_hip.hpp>
#include <alloy_dendrite/device_stepper_hip.hpp>
#include <alloy_dendrite/diagnostics.hpp>
#include <alloy_dendrite/elasticity.hpp>
#include <alloy_dendrite/material.hpp>
#include <alloy_dendrite/parameters.hpp>
#include <openpfc/frontend/io/snapshot_series.hpp>
#include <openpfc/frontend/utils/cli_options.hpp>

namespace {

using DevField = pfc::data::Field<double, pfc::HIPSpace>;
using RealField = pfc::data::Field<double>;
using pfc::solvers::Stiffness;

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
  Stiffness c_solid{};
  double liquid_shear = alloy_dendrite::material::kLiquidShearFraction;
  double liquid_bulk = 1.0;
  double eps_c = 0.0;
  double eps_T = 0.0;
  double u_ref = 0.0;
  double lambda_el = 0.0;
  std::string csv;
  std::string run_id = "hip-growth";
  double fta_gradient = 0.0;
  double fta_pulling = 0.0;
  double fta_x0 = std::numeric_limits<double>::quiet_NaN();
  struct FieldDump {
    std::string dir;
    int every{1};
  };
  FieldDump fields{};
};

[[nodiscard]] bool fta_active(const Cfg &cfg) noexcept {
  return cfg.fta_gradient != 0.0 || cfg.fta_pulling != 0.0;
}

void push_seed(DevField &dst, const RealField &src);

void impose_fta(DevField &theta, RealField &scratch, double gradient, double x0,
                double pulling, double t) {
  scratch.for_each_owned([&](int i, int j, int k) {
    const double x = scratch.coords(i, j, k)[0];
    scratch(i, j, k) = alloy_dendrite::fta_undercooling(gradient, x, x0, pulling, t);
  });
  push_seed(theta, scratch);
}

[[nodiscard]] double solid_front_x(const RealField &phi, MPI_Comm comm) {
  double local = -std::numeric_limits<double>::infinity();
  phi.for_each_owned([&](int i, int j, int k) {
    if (phi(i, j, k) > 0.0) local = std::max(local, phi.coords(i, j, k)[0]);
  });
  double global = local;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global;
}

void download_owned(DevField &src, RealField &dst) {
  const auto n = dst.local_size();
  const int hw = src.storage_halo();
  const std::size_t npx = static_cast<std::size_t>(n[0] + 2 * hw);
  const std::size_t npy = static_cast<std::size_t>(n[1] + 2 * hw);
  src.with_host_read([&](const double *data, std::size_t) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          dst(i, j, k) = data[(static_cast<std::size_t>(i) + hw) +
                              (static_cast<std::size_t>(j) + hw) * npx +
                              (static_cast<std::size_t>(k) + hw) * npx * npy];
  });
  // Dump is a read. with_host_read note_synced() both sides; the next
  // HaloExchange::sync_to_device() must not H2D the host halo. Same
  // pattern as alloy_dendrite_coupled_cost::pull_three.
  src.note_device_write();
  dst.note_host_write();
}

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

void report_hbm(int rank, int nproc, MPI_Comm comm, long long n_global,
                std::size_t known_owned_bytes, bool elastic) {
  std::size_t free_b = 0, total_b = 0;
  const hipError_t err = hipMemGetInfo(&free_b, &total_b);
  if (err != hipSuccess) {
    if (rank == 0)
      std::cerr << "hipMemGetInfo failed: " << hipGetErrorString(err) << '\n';
    return;
  }
  const unsigned long long used = static_cast<unsigned long long>(total_b - free_b);
  const unsigned long long free_ull = static_cast<unsigned long long>(free_b);
  const unsigned long long total_ull = static_cast<unsigned long long>(total_b);
  unsigned long long used_sum = 0, used_max = 0, free_min = 0, total_gcd = 0;
  MPI_Reduce(&used, &used_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&used, &used_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm);
  MPI_Reduce(&free_ull, &free_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, comm);
  MPI_Reduce(&total_ull, &total_gcd, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm);
  if (rank == 0) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << std::setprecision(6) << "HIP_MEM ranks=" << nproc
              << " used_sum_gib=" << used_sum / gib
              << " used_max_gib=" << used_max / gib
              << " free_min_gib=" << free_min / gib
              << " total_gcd_gib=" << total_gcd / gib
              << " bytes_per_cell=" << used_sum / static_cast<double>(n_global)
              << " known_owned_phi_U_th_bytes=" << known_owned_bytes
              << " elastic=" << (elastic ? 1 : 0) << '\n'
              << std::flush;
  }
}

int run(const Cfg &cfg, int rank, int nproc, MPI_Comm comm) {
  const auto &p = cfg.model;
  const int dim = (cfg.nz == 1) ? 2 : 3;
  const double dt_lim =
      alloy_dendrite::explicit_dt_limit(p, cfg.dx, dim, cfg.fd_order);
  const double dt = (cfg.dt > 0.0) ? cfg.dt : cfg.dt_safety * dt_lim;
  const bool fta = fta_active(cfg);
  if (fta && p.evolve_theta) {
    throw std::invalid_argument(
        "alloy_dendrite_hip_growth: FTA (--gradient/--pulling) freezes theta; "
        "do not also set evolve_theta");
  }
  if (fta && !(p.M_c > 0.0)) {
    throw std::invalid_argument(
        "alloy_dendrite_hip_growth: --gradient/--pulling is a no-op at Mc=0; "
        "FTA needs M_c theta in equation (2). Pass --Mc>0.");
  }
  if (cfg.fta_pulling > 0.0 && dt > 0.8 * cfg.dx / cfg.fta_pulling) {
    throw std::invalid_argument(
        "alloy_dendrite_hip_growth: dt violates the FTA interface CFL dx/V_p");
  }

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
  const double fta_x0 = std::isfinite(cfg.fta_x0) ? cfg.fta_x0 : xc;
  const double inv_w = 1.0 / (std::sqrt(2.0) * p.W0);
  RealField seed = pfc::data::field_from_inbox<double>(
      domain, pfc::decomposition::local_box(decomp, rank));
  seed.for_each_owned([&](int i, int j, int k) {
    const auto c = seed.coords(i, j, k);
    const double dz = (cfg.nz == 1) ? 0.0 : (c[2] - zc);
    const double r =
        std::sqrt((c[0] - xc) * (c[0] - xc) + (c[1] - yc) * (c[1] - yc) + dz * dz);
    seed(i, j, k) = std::tanh((cfg.seed_radius * p.W0 - r) * inv_w);
  });
  push_seed(phi, seed);
  seed.for_each_owned([&](int i, int j, int k) { seed(i, j, k) = -cfg.omega; });
  push_seed(U, seed);
  if (fta) {
    impose_fta(th, seed, cfg.fta_gradient, fta_x0, cfg.fta_pulling, 0.0);
  } else {
    seed.for_each_owned([&](int i, int j, int k) { seed(i, j, k) = 0.0; });
    push_seed(th, seed);
  }
  if (d2)
    d2->seed_conserved_solute();
  else
    d3->seed_conserved_solute();

  alloy_dendrite::ElasticParams ep;
  ep.c_solid = cfg.c_solid;
  ep.mu_liquid_fraction = cfg.liquid_shear;
  ep.bulk_liquid_fraction = cfg.liquid_bulk;
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
    if (d2)
      d2->set_elastic_driving_force(&elastic->driving_force());
    else
      d3->set_elastic_driving_force(&elastic->driving_force());
    el = elastic->solve(phi, U, th);
    if (hipDeviceSynchronize() != hipSuccess) {
      throw std::runtime_error("hipDeviceSynchronize failed after first solve");
    }
  }

  {
    const auto ln = phi.local_size();
    const std::size_t n_owned = static_cast<std::size_t>(ln[0]) *
                                static_cast<std::size_t>(ln[1]) *
                                static_cast<std::size_t>(ln[2]);
    const long long n_global = static_cast<long long>(cfg.nx) * cfg.ny * cfg.nz;
    report_hbm(rank, nproc, comm, n_global, 3 * n_owned * sizeof(double),
               static_cast<bool>(elastic));
  }

  alloy_dendrite::CsvAppender csv;
  if (!cfg.csv.empty()) {
    csv = alloy_dendrite::CsvAppender(
        cfg.csv,
        "run_id,step,t,t_pf_ms,t_el_ms,el_iters,el_energy,el_wstar,"
        "el_balance_rel,el_residual,el_max_dfel,mean_p,sig_vm_max,x_tip,v_tip,"
        "rho_tip",
        rank);
  }

  RealField host_phi = pfc::data::field_from_inbox<double>(
      domain, pfc::decomposition::local_box(decomp, rank));
  RealField host_U = pfc::data::field_from_inbox<double>(
      domain, pfc::decomposition::local_box(decomp, rank));
  pfc::io::SnapshotSeries snapshots(
      host_phi.domain(), host_phi.box(),
      pfc::io::SnapshotSeriesOptions{
          .directory = cfg.fields.dir, .prefix = cfg.run_id, .comm = comm});
  snapshots.set_cadence(
      pfc::io::SnapshotCadence(cfg.fields.every < 1 ? 1 : cfg.fields.every));
  if (!cfg.fields.dir.empty()) {
    snapshots.add_field("phi", host_phi);
    snapshots.add_field("U", host_U);
    if (cfg.elastic && elastic) {
      snapshots.add_field("f_el", const_cast<pfc::data::Field<double> &>(
                                      elastic->elastic_energy_density()));
      snapshots.add_field(
          "dfel_dphi", const_cast<pfc::data::Field<double> &>(elastic->dfel_host()));
    }
  }
  int n_seen = 0;

  double t_pf_sum = 0.0, t_el_sum = 0.0;
  int n_el = 0;
  double x_prev = 0.0;
  bool have_tip = false;

  for (int step = 1; step <= cfg.steps; ++step) {
    MPI_Barrier(comm);
    double a = MPI_Wtime();
    if (d2)
      d2->step(dt);
    else
      d3->step(dt);
    if (hipDeviceSynchronize() != hipSuccess) {
      throw std::runtime_error("hipDeviceSynchronize failed after step");
    }
    if (fta) {
      impose_fta(th, seed, cfg.fta_gradient, fta_x0, cfg.fta_pulling, step * dt);
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
      download_owned(phi, host_phi);
      download_owned(U, host_U);
      if (hipDeviceSynchronize() != hipSuccess) {
        throw std::runtime_error("hipDeviceSynchronize failed after download");
      }
      double x_tip = 0.0, v_tip = 0.0, rho = 0.0;
      if (cfg.nz == 1) {
        const auto plane = alloy_dendrite::global_xy_plane(host_phi, 0, comm);
        const auto tip = alloy_dendrite::measure_tip(
            plane, cfg.nx, cfg.ny, cfg.dx, cfg.dx, cfg.nx / 2, cfg.ny / 2, 8);
        x_tip = tip.x_tip;
        rho = tip.rho;
      } else {
        x_tip = solid_front_x(host_phi, comm);
      }
      if (have_tip) v_tip = (x_tip - x_prev) / (cfg.sample_every * dt);
      x_prev = x_tip;
      have_tip = true;
      if (csv.active()) {
        csv.row(alloy_dendrite::format(
            "%s,%d,%.6f,%.6f,%.6f,%d,%.6e,%.6e,%.3e,%.3e,%.6e,%.6e,%.6e,%.6f,"
            "%.6e,%.6f",
            cfg.run_id.c_str(), step, step * dt, 1e3 * tpf, 1e3 * tel, el.iterations,
            el.total_energy, el.virial_energy, el.energy_balance_rel, el.residual,
            el.max_dfel_dphi, el.mean_stress_trace,
            elastic ? elastic->max_von_mises() : 0.0, x_tip, v_tip, rho));
      }
      snapshots.write_if_due(n_seen, step, step * dt);
      ++n_seen;
    }
  }
  snapshots.close();

  if (rank == 0) {
    std::cout << std::setprecision(6);
    std::cout << "ALLOY_HIP_GROWTH run_id=" << cfg.run_id << " nx=" << cfg.nx
              << " ny=" << cfg.ny << " nz=" << cfg.nz << " steps=" << cfg.steps
              << " elastic=" << (cfg.elastic ? 1 : 0)
              << " lambda=" << cfg.model.lambda
              << " lambda_el=" << cfg.model.lambda_el << " eps_c=" << cfg.eps_c
              << " eps_T=" << cfg.eps_T << " u_ref=" << cfg.u_ref
              << " c11=" << cfg.c_solid.c11 << " liquid_shear=" << cfg.liquid_shear
              << " ranks=" << nproc << "\n";
    if (fta) {
      std::cout << "ALLOY_HIP_GROWTH_FTA gradient=" << cfg.fta_gradient
                << " pulling=" << cfg.fta_pulling << " x0=" << fta_x0
                << " Mc=" << p.M_c << " evolve_theta=0 dt=" << dt << "\n";
    }
    std::cout << "ALLOY_HIP_GROWTH_MS t_pf_mean=" << 1e3 * t_pf_sum / cfg.steps
              << " t_el_mean=" << (n_el > 0 ? 1e3 * t_el_sum / n_el : 0.0)
              << " el_solves=" << n_el << " last_iters=" << el.iterations
              << " last_energy=" << el.total_energy
              << " last_wstar=" << el.virial_energy
              << " last_el_balance_rel=" << el.energy_balance_rel
              << " last_mean_p=" << el.mean_stress_trace
              << " last_max_dfel=" << el.max_dfel_dphi << "\n";
  }
  return EXIT_SUCCESS;
}

int run_cli(int argc, char **argv, int rank, int nproc) {
  pfc::utils::CliOptions opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) {
      std::cout << "Usage: " << argv[0]
                << " [--nx= --ny= --nz= --elastic=0|1 --fields-dir=DIR ...]\n"
                << "Defaults match CPU growth: lambda=D_l/a2, Al-4.5wt%Cu\n"
                << "stiffness/eps_c from material.hpp. With --elastic=1,\n"
                << "lambda_el defaults to lambda (calibrated in f_ref units).\n"
                << "Elastic-off comparison: --elastic=1 --lambda-el=0.\n"
                << "FTA directional: theta=(G/dT_h)(x-x0-Vp t). evolve_theta\n"
                << "stays off; Mc theta stays live. Mc=0 with --gradient or\n"
                << "--pulling is rejected. Do not pass --elastic.\n"
                << "  --gradient=X --pulling=X --fta-x0=X --Mc=X --evolve-theta=0\n"
                << "  3-D example: --nx=48 --ny=48 --nz=48 --dx=0.8 --steps=20\n"
                << "    --Mc=0.5 --evolve-theta=0 --gradient=0.02 --pulling=0.05\n"
                << "    --csv=results/fta3d.csv --fields-dir=results/fta3d\n";
    }
    return EXIT_SUCCESS;
  }
  namespace mat = alloy_dendrite::material;
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
  cfg.model.k = opt.real("k", cfg.model.k);
  cfg.model.D_l = opt.real("Dl", cfg.model.D_l);
  cfg.model.lambda = opt.real("lambda", cfg.model.D_l / alloy_dendrite::kA2);
  cfg.model.D_th = opt.real("Dth", cfg.model.D_th);
  cfg.model.M_c = opt.real("Mc", cfg.model.M_c);
  cfg.model.eps4 = opt.real("eps4", cfg.model.eps4);
  cfg.model.evolve_theta = opt.flag("evolve-theta", cfg.model.evolve_theta);
  cfg.fta_gradient = opt.real("gradient", cfg.fta_gradient);
  cfg.fta_pulling = opt.real("pulling", cfg.fta_pulling);
  cfg.fta_x0 = opt.real("fta-x0", cfg.fta_x0);
  cfg.elastic = opt.flag("elastic", cfg.elastic);
  cfg.n_el_substep = opt.integer("n-el-substep", cfg.n_el_substep);
  cfg.warm_start = opt.flag("warm-start", cfg.warm_start);
  cfg.tol_el = opt.real("tol-el", cfg.tol_el);
  cfg.n_el_iter = opt.integer("n-el-iter", cfg.n_el_iter);
  cfg.liquid_shear = opt.real("liquid-shear", cfg.liquid_shear);
  cfg.liquid_bulk = opt.real("liquid-bulk", cfg.liquid_bulk);
  cfg.eps_c = opt.real("eps-c", mat::kEpsC);
  cfg.eps_T = opt.real("eps-T", cfg.eps_T);
  cfg.u_ref = opt.real("u-ref", -cfg.omega);
  if (opt.has("youngs") || opt.has("poisson")) {
    cfg.c_solid =
        Stiffness::isotropic(opt.real("youngs", 1.0), opt.real("poisson", 0.3));
  } else {
    const double soften = opt.real("el-soften", mat::kSofteningAtTm);
    cfg.c_solid = mat::al_cu_solid_stiffness(soften);
    cfg.c_solid.c11 = opt.real("el-c11", cfg.c_solid.c11);
    cfg.c_solid.c12 = opt.real("el-c12", cfg.c_solid.c12);
    cfg.c_solid.c44 = opt.real("el-c44", cfg.c_solid.c44);
  }
  if (cfg.elastic) {
    cfg.lambda_el = opt.real("lambda-el", cfg.model.lambda);
  } else if (opt.has("lambda-el") || opt.has("eps-c") || opt.has("eps-T") ||
             opt.has("u-ref") || opt.has("youngs") || opt.has("poisson") ||
             opt.has("el-c11") || opt.has("el-c12") || opt.has("el-c44") ||
             opt.has("el-soften") || opt.has("liquid-shear") ||
             opt.has("liquid-bulk")) {
    throw std::invalid_argument(
        "elastic constants are only read with --elastic=1. For an "
        "elastic-off reference on the same code path, use "
        "'--elastic=1 --lambda-el=0'");
  }
  cfg.model.lambda_el = cfg.lambda_el;
  cfg.csv = opt.text("csv", cfg.csv);
  cfg.run_id = opt.text("run-id", cfg.run_id);
  cfg.fields.dir = opt.text("fields-dir", "");
  cfg.fields.every = opt.integer("fields-every", cfg.fields.every);
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
          if (rank == 0)
            std::cerr << "alloy_dendrite_hip_growth: " << e.what() << "\n";
          return EXIT_FAILURE;
        }
      });
}
