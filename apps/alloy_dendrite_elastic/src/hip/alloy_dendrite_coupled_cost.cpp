// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_coupled_cost.cpp
 * @brief What one coupled step actually costs: GPU phase field, host
 *        round-trip, host elastic solve -- measured, not estimated.
 *
 * @details
 * ## The question this binary exists to answer
 *
 * `openpfc_apps/microelasticity.hpp` is host-only by deliberate design: its
 * author found that `SpectralETDOps` is shaped around one scalar field plus
 * two aux slots, and a six-component symmetric tensor with a spatially
 * varying stiffness does not fit. So a GPU phase field coupled to that solver
 * either needs a device port of the tensor solve, or pays a host round-trip
 * on every elastic solve. Choosing between those on the basis of "12
 * transforms per Eyre-Milton iteration and about 16 iterations sounds
 * expensive" is guessing. This driver runs the actual coupled step and times
 * its four parts separately:
 *
 *     t_pf    the GPU phase-field step  (4 kernels, 3 device halo exchanges)
 *     t_d2h   device -> host, phi / U / theta
 *     t_prep  host, building h, a, dh/dphi, da/dphi from them
 *     t_el    the Eyre-Milton fixed point, with the iteration count
 *     t_h2d   host -> device, dF_el/dphi
 *
 * so the route can be chosen from the ratio rather than from the adjective.
 *
 * ## Why the decompositions are forced to agree
 *
 * `pfc::sim::stacks::FDGPUStack` cuts the domain with
 * `pfc::decomposition::create`, a minimum-surface brick; `SpectralCPUStack`
 * cuts it with `spectral_fft_proc_grid`, a slab, because that is what HeFFTe
 * wants. Left alone, the two disagree the moment `nproc > 1`, and every
 * elastic solve then pays an MPI redistribution *on top of* the host
 * round-trip -- a cost that has nothing to do with the physics and everything
 * to do with two stacks having different opinions. `DeviceStepper` therefore
 * takes its decomposition as an argument and this driver hands it the FFT's,
 * so the round-trip is a per-rank copy with no communication in it. The price
 * is a slab-shaped FD halo instead of a brick-shaped one, which is a worse
 * surface-to-volume ratio; `--fd-grid=brick` measures that price by building
 * the FD side on `decomposition::create` instead and refusing to couple
 * (there is no point timing an elastic solve whose input would have to be
 * redistributed first -- that route is measured by its absence).
 *
 * ## Warm start
 *
 * `MicroelasticityParams::warm_start` reuses the previous solution as the
 * initial iterate. In a time loop where the interface moves a fraction of a
 * cell per step this should collapse the iteration count, and it is the one
 * knob that could change the answer, so it is a switch (`--warm-start`) and
 * the per-step iteration count is reported rather than averaged away:
 * `--csv` writes one row per elastic solve.
 *
 * Usage: `alloy_dendrite_coupled_cost [--key=value ...]`, `--help` for the list.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "alloy_dendrite_coupled_cost requires HIP (-DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/bind_local_device.hpp>

#include <openpfc_apps/microelasticity.hpp>

#include <alloy_dendrite/cli.hpp>
#include <alloy_dendrite/device_stepper_hip.hpp>
#include <alloy_dendrite/diagnostics.hpp>
#include <alloy_dendrite/parameters.hpp>

namespace {

using DevField = pfc::data::Field<double, pfc::HIPSpace>;
using RealField = pfc::data::Field<double>;
using pfc::apps::EigenstrainMicroelasticity;
using pfc::apps::MicroelasticityParams;
using pfc::apps::MicroelasticityScheme;
using pfc::apps::Stiffness;

struct CostConfig {
  alloy_dendrite::ModelParams model{};
  int nx = 128;
  int ny = 128;
  int nz = 128;
  double dx = 0.8;
  int fd_order = 4;
  int steps = 40;
  int warmup = 5;
  double dt = 0.0;
  double dt_safety = 0.2;
  double omega = 0.55;
  double seed_radius = 12.0;

  /// Couple equations (5)-(7) at all. `0` times the bare GPU step.
  bool elastic = true;
  /// Elastic solve every `n_el_substep` phase-field steps (spec, "Numerics").
  int n_el_substep = 1;
  bool warm_start = true;
  double tol_el = 1.0e-6;
  int n_el_iter = 50;
  /// Young's modulus and Poisson ratio of the solid; the liquid is
  /// `soft_liquid(solid, liquid_shear)`.
  double youngs = 1.0;
  double poisson = 0.3;
  double liquid_shear = pfc::apps::kDefaultLiquidShearFraction;
  /// Eigenstrain amplitudes: `a = h(phi) [eps_c (U - U_ref) + eps_T theta]`.
  double eps_c = 0.01;
  double eps_T = 0.0;
  double u_ref = 0.0;
  /// Elastic feedback strength in equation (2). Zero still solves and still
  /// pays, which is what makes it a clean cost measurement.
  double lambda_el = 0.0;
  /**
   * @brief Which decomposition the FD fields are cut on.
   *
   * `slab` means "whatever `spectral_fft_proc_grid` gave the FFT", which is
   * the only setting that can couple. The name is the common case rather
   * than the universal one: that function only returns a true 1-D slab at
   * nine ranks or more (`kSpectralSlabMinRanks`), and below that it returns
   * the same minimum-surface brick `decomposition::create` would, so at
   * eight GCDs the two settings differ only in that `brick` is allowed to
   * disagree with the FFT and therefore refuses to couple.
   */
  std::string fd_grid = "slab";
  std::string csv;
  std::string run_id = "cost";
  bool quiet = false;
};

/// Mean of the trailing entries, skipping the warm-up.
[[nodiscard]] double mean_of(const std::vector<double> &v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

[[nodiscard]] double mean_of(const std::vector<int> &v) {
  if (v.empty()) return 0.0;
  return static_cast<double>(std::accumulate(v.begin(), v.end(), 0)) /
         static_cast<double>(v.size());
}

/// Max over ranks, so the reported cost is the one the step actually waits for.
[[nodiscard]] double reduce_max(double x, MPI_Comm comm) {
  double out = 0.0;
  MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
  return out;
}

/**
 * @brief Pull the owned cells of three padded device fields into three
 *        unpadded host fields.
 *
 * `Field::with_host_view` moves the *whole padded* buffer, halo included, so
 * this costs `(n + 2hw)^3` per field rather than `n^3`. That overhead is real
 * and is deliberately not optimised away here: a hand-rolled `hipMemcpy3D` of
 * the owned sub-block would shave it, but the number this driver exists to
 * produce is "what does the obvious coupling cost", and the obvious coupling
 * uses the `Field` API. The measured D2H figure therefore bounds the tuned
 * one from above, which is the safe direction for a route decision.
 */
void pull_three(DevField &phi_d, DevField &u_d, DevField &th_d, RealField &phi_h,
                RealField &u_h, RealField &th_h) {
  DevField *src[3] = {&phi_d, &u_d, &th_d};
  RealField *dst[3] = {&phi_h, &u_h, &th_h};
  for (int f = 0; f < 3; ++f) {
    const auto n = src[f]->local_size();
    const int hw = src[f]->storage_halo();
    const std::size_t npx = static_cast<std::size_t>(n[0] + 2 * hw);
    const std::size_t npy = static_cast<std::size_t>(n[1] + 2 * hw);
    RealField &out = *dst[f];
    src[f]->with_host_view([&](const double *data, std::size_t) {
      for (int k = 0; k < n[2]; ++k) {
        for (int j = 0; j < n[1]; ++j) {
          for (int i = 0; i < n[0]; ++i) {
            out(i, j, k) = data[(static_cast<std::size_t>(i) + hw) +
                                (static_cast<std::size_t>(j) + hw) * npx +
                                (static_cast<std::size_t>(k) + hw) * npx * npy];
          }
        }
      }
    });
    src[f]->note_device_write(); // a read, not a hand-over
  }
}

/// Scatter an unpadded host field into the owned cells of a padded device
/// field and push it. Only owned cells are written: `dF_el/dphi` is read
/// pointwise in stage B, so its halo never matters.
void push_owned(const RealField &src, DevField &dst) {
  const auto n = dst.local_size();
  const int hw = dst.storage_halo();
  const std::size_t npx = static_cast<std::size_t>(n[0] + 2 * hw);
  const std::size_t npy = static_cast<std::size_t>(n[1] + 2 * hw);
  dst.with_host_view([&](double *data, std::size_t) {
    for (int k = 0; k < n[2]; ++k) {
      for (int j = 0; j < n[1]; ++j) {
        for (int i = 0; i < n[0]; ++i) {
          data[(static_cast<std::size_t>(i) + hw) +
               (static_cast<std::size_t>(j) + hw) * npx +
               (static_cast<std::size_t>(k) + hw) * npx * npy] = src(i, j, k);
        }
      }
    }
  });
  dst.sync_to_device();
}

int run_cost(const CostConfig &cfg, int rank, int nproc, MPI_Comm comm) {
  using alloy_dendrite::DeviceStepper;

  const auto &p = cfg.model;
  const double dt_lim = alloy_dendrite::explicit_dt_limit(p, cfg.dx, 3);
  const double dt = (cfg.dt > 0.0) ? cfg.dt : cfg.dt_safety * dt_lim;

  const auto domain = pfc::domain::create(
      pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));

  // The FFT stack first: it owns the decomposition the FD side has to match.
  pfc::sim::stacks::SpectralCPUStack fftstack(domain, rank, nproc, comm);
  const auto fft_grid =
      pfc::decomposition::spectral_fft_proc_grid(domain.size, nproc);
  auto fd_decomp =
      (cfg.fd_grid == "brick")
          ? pfc::decomposition::create(domain, nproc)
          : pfc::decomposition::create(domain, fft_grid);

  const auto fd_box = pfc::decomposition::local_box(fd_decomp, rank);
  const auto fft_box = fftstack.u().box();
  const bool boxes_match =
      fd_box.low == fft_box.low && fd_box.size == fft_box.size;
  if (cfg.elastic && !boxes_match) {
    throw std::runtime_error(
        "alloy_dendrite_coupled_cost: the FD and FFT local boxes differ, so a "
        "coupled step would need an MPI redistribution this driver does not "
        "implement. Use --fd-grid=slab (the default) to couple, or "
        "--elastic=0 to time the bare phase-field step on the FD-optimal "
        "brick.");
  }

  DeviceStepper<3> dev(domain, fd_decomp, rank, comm, p, cfg.fd_order);

  // ---- initial condition: one tanh sphere in a supersaturated melt -----
  const double xc = 0.5 * cfg.nx * cfg.dx;
  const double yc = 0.5 * cfg.ny * cfg.dx;
  const double zc = 0.5 * cfg.nz * cfg.dx;
  const double inv_w = 1.0 / (std::sqrt(2.0) * p.W0);
  {
    RealField seed = pfc::data::field_from_inbox<double>(
        domain, fftstack.fft().get_inbox_bounds());
    // phi
    seed.for_each_owned([&](int i, int j, int k) {
      const auto c = seed.coords(i, j, k);
      const double r = std::sqrt((c[0] - xc) * (c[0] - xc) +
                                 (c[1] - yc) * (c[1] - yc) +
                                 (c[2] - zc) * (c[2] - zc));
      seed(i, j, k) = std::tanh((cfg.seed_radius * p.W0 - r) * inv_w);
    });
    push_owned(seed, dev.phi());
    seed.for_each_owned([&](int i, int j, int k) { seed(i, j, k) = -cfg.omega; });
    push_owned(seed, dev.solute());
    seed.for_each_owned([&](int i, int j, int k) { seed(i, j, k) = 0.0; });
    push_owned(seed, dev.temperature());
  }
  dev.seed_conserved_solute();

  // ---- elastic side ----------------------------------------------------
  const auto inbox = fftstack.fft().get_inbox_bounds();
  RealField h = pfc::data::field_from_inbox<double>(domain, inbox);
  RealField amp = pfc::data::field_from_inbox<double>(domain, inbox);
  RealField dh = pfc::data::field_from_inbox<double>(domain, inbox);
  RealField damp = pfc::data::field_from_inbox<double>(domain, inbox);
  RealField phi_h = pfc::data::field_from_inbox<double>(domain, inbox);
  RealField u_h = pfc::data::field_from_inbox<double>(domain, inbox);
  RealField th_h = pfc::data::field_from_inbox<double>(domain, inbox);
  DevField dfel_d(domain, fd_box, cfg.fd_order / 2);

  MicroelasticityParams mp;
  mp.c_solid = Stiffness::isotropic(cfg.youngs, cfg.poisson);
  mp.c_liquid = pfc::apps::soft_liquid(mp.c_solid, cfg.liquid_shear);
  mp.scheme = MicroelasticityScheme::EyreMilton;
  mp.tol_el = cfg.tol_el;
  mp.n_el_iter = cfg.n_el_iter;
  mp.warm_start = cfg.warm_start;
  mp.comm = comm;

  std::unique_ptr<EigenstrainMicroelasticity> solver;
  if (cfg.elastic) {
    solver = std::make_unique<EigenstrainMicroelasticity>(domain, fftstack.fft(), mp);
    dev.set_elastic_driving_force(&dfel_d);
  }

  // ---- the timed loop --------------------------------------------------
  std::vector<double> t_pf, t_d2h, t_prep, t_el, t_h2d;
  std::vector<int> its;
  std::vector<double> resid;
  alloy_dendrite::CsvAppender csv;
  if (!cfg.csv.empty()) {
    csv = alloy_dendrite::CsvAppender(
        cfg.csv,
        "run_id,step,t_pf_ms,t_d2h_ms,t_prep_ms,t_el_ms,t_h2d_ms,el_iters,"
        "el_residual,el_converged",
        rank);
  }

  for (int step = 1; step <= cfg.steps; ++step) {
    const bool record = step > cfg.warmup;

    MPI_Barrier(comm);
    double a = MPI_Wtime();
    dev.step(dt);
    if (hipDeviceSynchronize() != hipSuccess) {
      throw std::runtime_error("hipDeviceSynchronize failed after a step");
    }
    double b = MPI_Wtime();
    const double pf = b - a;

    double d2h = 0.0, prep = 0.0, el = 0.0, h2d = 0.0;
    int iters = 0;
    double residual = 0.0;
    bool converged = true;
    if (cfg.elastic && (step % cfg.n_el_substep == 0)) {
      a = MPI_Wtime();
      pull_three(dev.phi(), dev.solute(), dev.temperature(), phi_h, u_h, th_h);
      b = MPI_Wtime();
      d2h = b - a;

      a = b;
      // eps* = h(phi) [eps_c (U - U_ref) + eps_T theta] I; the solver's model
      // is eps* = a(x) P with P = I, so a = h * (that bracket) and the two
      // derivative fields are dh/dphi = 1/2 and da/dphi = (1/2) * bracket.
      h.for_each_owned([&](int i, int j, int k) {
        const double ph = phi_h(i, j, k);
        const double bracket =
            cfg.eps_c * (u_h(i, j, k) - cfg.u_ref) + cfg.eps_T * th_h(i, j, k);
        const double hv = 0.5 * (1.0 + ph);
        h(i, j, k) = hv;
        amp(i, j, k) = hv * bracket;
        dh(i, j, k) = 0.5;
        damp(i, j, k) = 0.5 * bracket;
      });
      b = MPI_Wtime();
      prep = b - a;

      a = b;
      const auto rep = solver->solve(h, amp, &dh, &damp);
      b = MPI_Wtime();
      el = b - a;
      iters = rep.iterations;
      residual = rep.residual;
      converged = rep.converged;

      a = b;
      push_owned(solver->dfel_dphi(), dfel_d);
      b = MPI_Wtime();
      h2d = b - a;
    }

    if (record) {
      t_pf.push_back(reduce_max(pf, comm));
      if (cfg.elastic && (step % cfg.n_el_substep == 0)) {
        t_d2h.push_back(reduce_max(d2h, comm));
        t_prep.push_back(reduce_max(prep, comm));
        t_el.push_back(reduce_max(el, comm));
        t_h2d.push_back(reduce_max(h2d, comm));
        its.push_back(iters);
        resid.push_back(residual);
      }
      if (csv.active()) {
        csv.row(alloy_dendrite::format(
            "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.3e,%d", cfg.run_id.c_str(), step,
            1e3 * pf, 1e3 * d2h, 1e3 * prep, 1e3 * el, 1e3 * h2d, iters, residual,
            converged ? 1 : 0));
      }
    }
  }

  if (rank == 0 && !cfg.quiet) {
    const double pf = 1e3 * mean_of(t_pf);
    const double d2h = 1e3 * mean_of(t_d2h);
    const double prep = 1e3 * mean_of(t_prep);
    const double el = 1e3 * mean_of(t_el);
    const double h2d = 1e3 * mean_of(t_h2d);
    const double rt = d2h + h2d;
    std::cout << std::setprecision(6);
    std::cout << "ALLOY_COST run_id=" << cfg.run_id << " nx=" << cfg.nx
              << " ny=" << cfg.ny << " nz=" << cfg.nz << " dx=" << cfg.dx
              << " fd_order=" << cfg.fd_order << " ranks=" << nproc
              << " fd_grid=" << cfg.fd_grid << " steps=" << cfg.steps
              << " warmup=" << cfg.warmup << " dt=" << dt
              << " elastic=" << (cfg.elastic ? 1 : 0)
              << " warm_start=" << (cfg.warm_start ? 1 : 0)
              << " n_el_substep=" << cfg.n_el_substep << " tol_el=" << cfg.tol_el
              << " liquid_shear=" << cfg.liquid_shear << "\n";
    std::cout << "ALLOY_COST_MS"
              << " t_pf=" << pf << " t_d2h=" << d2h << " t_prep=" << prep
              << " t_el=" << el << " t_h2d=" << h2d << " t_roundtrip=" << rt
              << " t_total=" << (pf + d2h + prep + el + h2d) << "\n";
    std::cout << "ALLOY_COST_RATIO"
              << " roundtrip_over_pf=" << (pf > 0.0 ? rt / pf : 0.0)
              << " elastic_over_pf=" << (pf > 0.0 ? el / pf : 0.0)
              << " elastic_over_roundtrip=" << (rt > 0.0 ? el / rt : 0.0)
              << " coupled_over_pf="
              << (pf > 0.0 ? (pf + rt + prep + el) / pf : 0.0) << "\n";
    if (!its.empty()) {
      const auto mm = std::minmax_element(its.begin(), its.end());
      std::cout << "ALLOY_COST_ITER"
                << " mean=" << mean_of(its) << " min=" << *mm.first
                << " max=" << *mm.second << " first=" << its.front()
                << " last=" << its.back() << " mean_residual=" << mean_of(resid)
                << "\n";
    }
  }
  return EXIT_SUCCESS;
}

void print_usage(std::ostream &os, const char *exe) {
  CostConfig d;
  os << "Usage: " << exe << " [--key=value ...]\n\n"
     << "Times one coupled step: GPU phase field, host round-trip, host "
        "elastic solve.\n\n"
     << "Grid and integration\n"
     << "  --nx --ny --nz         grid (" << d.nx << "," << d.ny << "," << d.nz
     << ")\n"
     << "  --dx=X                 spacing in W0            (" << d.dx << ")\n"
     << "  --fd-order=N           even FD order, 2..14     (" << d.fd_order << ")\n"
     << "  --steps=N              steps                    (" << d.steps << ")\n"
     << "  --warmup=N             steps excluded from the means (" << d.warmup
     << ")\n"
     << "  --dt=X --dt-safety=X   explicit step; 0 = auto  (" << d.dt << ")\n"
     << "  --fd-grid=slab|brick   FD decomposition; slab matches the FFT ("
     << d.fd_grid << ")\n\n"
     << "Phase field\n"
     << "  --omega --seed-radius --lambda --k --Dl --Dth --Mc --eps4 "
        "--at-scale\n"
     << "  --aniso-form=S      karma-rappel (spec, default) or "
        "unnormalised (PR #147)\n"
     << "\n"
     << "Elasticity (equations (5)-(7))\n"
     << "  --elastic=0|1          couple at all            (" << d.elastic << ")\n"
     << "  --n-el-substep=N       solve every N steps      (" << d.n_el_substep
     << ")\n"
     << "  --warm-start=0|1       reuse the last solution  (" << d.warm_start
     << ")\n"
     << "  --tol-el=X             fixed-point tolerance    (" << d.tol_el << ")\n"
     << "  --n-el-iter=N          iteration cap            (" << d.n_el_iter
     << ")\n"
     << "  --youngs --poisson     solid stiffness          (" << d.youngs << ", "
     << d.poisson << ")\n"
     << "  --liquid-shear=X       mu_l / mu_s              (" << d.liquid_shear
     << ")\n"
     << "  --eps-c --eps-T --u-ref  eigenstrain amplitudes\n"
     << "  --lambda-el=X          feedback strength in eq. (2) (" << d.lambda_el
     << ")\n\n"
     << "Output\n"
     << "  --csv=PATH             per-step timings\n"
     << "  --run-id=NAME --quiet=1\n";
}

int run(int argc, char **argv, int rank, int nproc) {
  alloy_dendrite::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) print_usage(std::cout, argv[0]);
    return EXIT_SUCCESS;
  }

  CostConfig cfg;
  cfg.model.eps4 = 0.2;
  cfg.model.D_th = 2.0;
  cfg.model.M_c = 0.5;

  cfg.nx = opt.integer("nx", cfg.nx);
  cfg.ny = opt.integer("ny", cfg.ny);
  cfg.nz = opt.integer("nz", cfg.nz);
  cfg.dx = opt.real("dx", cfg.dx);
  cfg.fd_order = opt.integer("fd-order", cfg.fd_order);
  cfg.steps = opt.integer("steps", cfg.steps);
  cfg.warmup = opt.integer("warmup", cfg.warmup);
  cfg.dt = opt.real("dt", cfg.dt);
  cfg.dt_safety = opt.real("dt-safety", cfg.dt_safety);
  cfg.fd_grid = opt.text("fd-grid", cfg.fd_grid);
  cfg.omega = opt.real("omega", cfg.omega);
  cfg.seed_radius = opt.real("seed-radius", cfg.seed_radius);
  cfg.model.lambda = opt.real("lambda", cfg.model.lambda);
  cfg.model.k = opt.real("k", cfg.model.k);
  cfg.model.D_l = opt.real("Dl", cfg.model.D_l);
  cfg.model.D_th = opt.real("Dth", cfg.model.D_th);
  cfg.model.M_c = opt.real("Mc", cfg.model.M_c);
  cfg.model.eps4 = opt.real("eps4", cfg.model.eps4);
  cfg.model.aniso_form = alloy_dendrite::parse_anisotropy_form(opt.text(
      "aniso-form", alloy_dendrite::anisotropy_form_name(cfg.model.aniso_form)));
  cfg.model.at_scale = opt.real("at-scale", cfg.model.at_scale);
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
  cfg.quiet = opt.flag("quiet", cfg.quiet);
  opt.require_all_consumed();

  if (cfg.n_el_substep < 1) {
    throw std::invalid_argument("--n-el-substep must be >= 1");
  }
  if (cfg.fd_grid != "slab" && cfg.fd_grid != "brick") {
    throw std::invalid_argument("--fd-grid must be 'slab' or 'brick'");
  }

  pfc::runtime::gpu::bind_local_device(MPI_COMM_WORLD);
  return run_cost(cfg, rank, nproc, MPI_COMM_WORLD);
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        try {
          return run(app_argc, app_argv, rank, nproc);
        } catch (const std::exception &e) {
          if (rank == 0) {
            std::cerr << "alloy_dendrite_coupled_cost: " << e.what() << "\n";
          }
          return EXIT_FAILURE;
        }
      });
}
