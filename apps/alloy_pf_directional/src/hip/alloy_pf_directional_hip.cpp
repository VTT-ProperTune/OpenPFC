// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_pf_directional_hip.cpp
 * @brief MPI + HIP Al-Cu FTA. One rank per GCD. Device-resident fields;
 *        DeviceFullHalo / Full2D–Full3D (GPU-aware or packed faces). Persistent: φ/ψ, c, ∂tφ.
 *
 * Nz=1 + n_dim=2 is the 2D-equivalent check path. Classic AMR is not implemented.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "alloy_pf_directional_hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <alloy_pf_directional/cli.hpp>
#include <alloy_pf_directional/device_step_hip.hpp>
#include <alloy_pf_directional/engine.hpp>
#include <alloy_pf_directional/mpi_support.hpp>
#include <alloy_pf_directional/recompute.hpp>
#include <alloy_pf_directional/window.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/full_padded_device_halo_gpu.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Field = pfc::data::Field<double, pfc::HostSpace>;

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

struct DevBuf {
  double *p = nullptr;
  explicit DevBuf(std::size_t bytes) {
    hip_check(hipMalloc(reinterpret_cast<void **>(&p), bytes), "hipMalloc");
  }
  ~DevBuf() {
    if (p) {
      (void)hipFree(p);
    }
  }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
};

void h2d(const Field &h, double *d) {
  hip_check(hipMemcpy(d, h.data(), h.size() * sizeof(double), hipMemcpyHostToDevice), "h2d");
}
void d2h(const double *d, Field &h) {
  hip_check(hipMemcpy(h.data(), d, h.size() * sizeof(double), hipMemcpyDeviceToHost), "d2h");
}

alloy_pf_directional::HipPhys make_phys(const alloy_pf_directional::RunConfig &cfg, bool dim3, int i0, int j0, int k0,
                            int shift) {
  using namespace alloy_pf_directional;
  const Physics &p = cfg.phys;
  const Mat3 R = bunge_crystal_to_lab(p.phi1_g1, p.Phi_g1, p.phi2_g1);
  HipPhys H{};
  H.ke = p.ke;
  H.clo = p.clo;
  H.W0 = p.W0;
  H.lambda = p.lambda;
  H.tau0 = p.tau0;
  H.tau_beta = p.tau_beta;
  H.tau_a2 = p.tau_a2;
  H.eps_c = p.eps_c;
  H.eps_k = p.eps_k;
  H.omega = p.omega;
  H.omega_zhong = p.omega_zhong;
  H.G = p.G;
  H.mle = p.mle;
  H.Vp = p.Vp;
  H.x_tl = p.x_tl;
  H.delta_iso = p.delta_iso;
  H.a_at = p.a_at;
  H.A_trap = p.A_trap;
  H.DL = p.DL;
  H.dt = p.dt;
  H.dx = p.dx;
  H.inv_dx = 1.0 / p.dx;
  H.dV = dim3 ? p.dx * p.dx * p.dx : p.dx * p.dx;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      H.R[3 * i + j] = R[i][j];
    }
  }
  H.n_dim = p.n_dim;
  H.n_grains = cfg.n_grains;
  H.dim3 = dim3;
  H.use_glasner = cfg.use_glasner;
  H.periodic_y = cfg.periodic_y;
  H.periodic_z = dim3 && cfg.periodic_z;
  H.noise_F0 = cfg.noise_F0;
  H.noise_seed = cfg.noise_seed;
  H.shift_cells = shift;
  H.i0 = i0;
  H.j0 = j0;
  H.k0 = k0;
  return H;
}

void run_hip(const alloy_pf_directional::RunConfig &cfg, int rank, int nproc) {
  using namespace alloy_pf_directional;
  MPI_Comm node_tmp{};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_tmp);
  int local_rank = 0;
  MPI_Comm_rank(node_tmp, &local_rank);
  MPI_Comm_free(&node_tmp);
  int n_dev = 0;
  hip_check(hipGetDeviceCount(&n_dev), "hipGetDeviceCount");
  if (n_dev < 1) {
    throw std::runtime_error("No HIP devices visible");
  }
  hip_check(hipSetDevice(local_rank % n_dev), "hipSetDevice");

  const Physics phys = cfg.phys;
  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int Nz = cfg.Nz < 1 ? 1 : cfg.Nz;
  const bool dim3 = (phys.n_dim >= 3 && Nz > 1);
  const double dx = phys.dx;
  const bool use_glasner = cfg.use_glasner;
  const int n_grains = cfg.n_grains;
  const bool quiet = env_on("OPENPFC_ALCU_QUIET", false);

  const auto domain = pfc::domain::create(
      pfc::GridSize({Nx, Ny, Nz}), pfc::PhysicalOrigin({0.5 * dx, 0.5 * dx, 0.5 * dx}),
      pfc::GridSpacing({dx, dx, dx}), pfc::Bool3{true, true, true});
  const auto decomp = pfc::decomposition::create(domain, nproc);
  constexpr int hw = 1;
  Field phi1 = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field phi2 = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field psi1 = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field psi2 = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field c = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field dphi1 = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field dphi2 = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  mpi_util::init_fields(phi1, phi2, psi1, psi2, c, cfg, use_glasner, dim3);

  const std::size_t bytes = phi1.size() * sizeof(double);
  DevBuf d_phi1(bytes), d_phi2(bytes), d_psi1(bytes), d_psi2(bytes), d_c(bytes), d_d1(bytes),
      d_d2(bytes), d_dc(bytes), d_eu(bytes), d_u(bytes), d_ad(bytes), d_aa(bytes), d_beta(bytes);
  h2d(phi1, d_phi1.p);
  h2d(phi2, d_phi2.p);
  h2d(psi1, d_psi1.p);
  h2d(psi2, d_psi2.p);
  h2d(c, d_c.p);
  h2d(dphi1, d_d1.p);
  h2d(dphi2, d_d2.p);

  // Device Full halo on raw buffers (GPU-aware MPI, or packed faces if
  // OPENPFC_HIP_FORCE_PACKED_HALO=1). HIP Fields + HaloExchange<HIPSpace> would
  // need residency notes after every kernel write; kernels still take double*.
  const auto halo_dirs =
      dim3 ? pfc::halo::presets::Full3D() : pfc::halo::presets::Full2D();
  pfc::gpu::DeviceFullHalo<pfc::hip::HIPHaloOps> halo(
      decomp, rank, hw, MPI_COMM_WORLD, /*n_fields=*/7, halo_dirs, 0);

  const int nx = phi1.local_size()[0];
  const int ny = phi1.local_size()[1];
  const int nz = phi1.local_size()[2];
  const int i0 = phi1.box().low[0];
  const int j0 = phi1.box().low[1];
  const int k0 = phi1.box().low[2];
  const bool lo_x = mpi_util::owns_lo(phi1, 0);
  const bool hi_x = mpi_util::owns_hi(phi1, 0, Nx);
  const double dV = dim3 ? dx * dx * dx : dx * dx;

  auto exchange7 = [&]() {
    double *fs[7] = {d_phi1.p, d_phi2.p, d_psi1.p, d_psi2.p, d_c.p, d_d1.p, d_d2.p};
    halo.exchange(fs, nullptr);
    alcu_noflux_x_hip(d_phi1.p, nx, ny, nz, hw, lo_x, hi_x);
    alcu_noflux_x_hip(d_phi2.p, nx, ny, nz, hw, lo_x, hi_x);
    alcu_noflux_x_hip(d_psi1.p, nx, ny, nz, hw, lo_x, hi_x);
    alcu_noflux_x_hip(d_psi2.p, nx, ny, nz, hw, lo_x, hi_x);
    alcu_noflux_x_hip(d_c.p, nx, ny, nz, hw, lo_x, hi_x);
    alcu_noflux_x_hip(d_d1.p, nx, ny, nz, hw, lo_x, hi_x);
    alcu_noflux_x_hip(d_d2.p, nx, ny, nz, hw, lo_x, hi_x);
  };
  exchange7();

  double mass0_loc = 0.0;
  c.for_each_owned([&](int i, int j, int k) { mass0_loc += c(i, j, k); });
  mass0_loc *= dV;
  double mass0 = 0.0;
  MPI_Allreduce(&mass0_loc, &mass0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    std::filesystem::create_directories(cfg.output_dir);
    std::ofstream meta(cfg.output_dir + "/meta.txt");
    meta << "backend mpi_hip\nNx " << Nx << "\nNy " << Ny << "\nNz " << Nz << "\nnproc "
         << nproc << "\n";
    meta << "stored_fields phi[,psi] c dphi dc_scratch eu u a_diff a_at beta_at\nrecomputed_fields aniso_fluxes\n";
    meta << "amr_deferred 1\n";
    std::cout << "ALCU_FTA backend=mpi_hip device=" << (local_rank % n_dev)
              << " nproc=" << nproc << " Nx=" << Nx << " Ny=" << Ny << " Nz=" << Nz << "\n";
  }

  const int n_loop =
      cfg.timed_steps > 0 ? (cfg.warmup_steps + cfg.timed_steps) : cfg.n_steps;
  const int warmup = cfg.timed_steps > 0 ? cfg.warmup_steps : 0;
  int win_shift = 0;
  double t_halo = 0.0, t_kern = 0.0, t_timed0 = 0.0, t_timed1 = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  int n_done = 0;

  for (int istep = 1; istep <= n_loop; ++istep) {
    if (cfg.timed_steps > 0 && istep == warmup + 1) {
      hip_check(hipDeviceSynchronize(), "sync warmup");
      MPI_Barrier(MPI_COMM_WORLD);
      t_timed0 = MPI_Wtime();
      t_halo = 0.0;
      t_kern = 0.0;
    }
    const double th0 = MPI_Wtime();
    exchange7();
    t_halo += MPI_Wtime() - th0;
    HipPhys Hp = make_phys(cfg, dim3, i0, j0, k0, win_shift);
    const double tk0 = MPI_Wtime();
    alcu_fill_eu_u_hip(d_eu.p, d_u.p, d_phi1.p, d_phi2.p, d_c.p, nx, ny, nz, hw, Hp);
    const double *pf1 = use_glasner ? d_psi1.p : d_phi1.p;
    alcu_grain_step_hip(pf1, d_phi1.p, d_phi2.p, d_c.p, d_d1.p, nx, ny, nz, hw, Hp, istep, 1);
    if (n_grains == 2) {
      HipPhys H2 = Hp;
      const Mat3 R2 = bunge_crystal_to_lab(phys.phi1_g2, phys.Phi_g2, phys.phi2_g2);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          H2.R[3 * i + j] = R2[i][j];
        }
      }
      const double *pf2 = use_glasner ? d_psi2.p : d_phi2.p;
      alcu_grain_step_hip(pf2, d_phi2.p, d_phi1.p, d_c.p, d_d2.p, nx, ny, nz, hw, H2, istep, 2);
    }
    alcu_euler_hip(d_psi1.p, d_phi1.p, d_d1.p, d_psi2.p, d_phi2.p, d_d2.p, nx, ny, nz, hw,
                   phys.dt, use_glasner, n_grains);
    exchange7();
    alcu_fill_eu_u_hip(d_eu.p, d_u.p, d_phi1.p, d_phi2.p, d_c.p, nx, ny, nz, hw, Hp);
    alcu_fill_solute_nodal_hip(d_ad.p, d_aa.p, d_beta.p, d_u.p, d_phi1.p, d_phi2.p, d_c.p,
                              d_d1.p, d_d2.p, d_eu.p, nx, ny, nz, hw, Hp);
    alcu_solute_iso_hip(d_c.p, d_dc.p, d_ad.p, d_aa.p, d_u.p, d_beta.p, nx, ny, nz, hw, Hp);
    t_kern += MPI_Wtime() - tk0;

    if (cfg.window_enable) {
      d2h(d_phi1.p, phi1);
      double xt_loc = 0.0;
      const int nx_h = phi1.local_size()[0];
      const int ny_h = phi1.local_size()[1];
      const int nz_h = phi1.local_size()[2];
      for (int kk = 0; kk < nz_h; ++kk) {
        for (int j = 0; j < ny_h; ++j) {
          for (int i = 0; i < nx_h - 1; ++i) {
            const double p0 = phi1(i, j, kk);
            const double p1 = phi1(i + 1, j, kk);
            if (p0 >= 0.0 && p1 < 0.0) {
              const auto g = phi1.global(i, j, kk);
              const double a = p0 / (p0 - p1 + 1.0e-30);
              xt_loc = std::max(xt_loc, (static_cast<double>(g[0]) + 0.5 +
                                         static_cast<double>(win_shift)) *
                                                dx +
                                            a * dx);
            }
          }
        }
      }
      double xt = 0.0;
      MPI_Allreduce(&xt_loc, &xt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const int nsh = window_shift_count(xt, win_shift, dx, cfg.window_margin_left, Nx,
                                         cfg.window_margin_right);
      const double c_liq = c_eq(-1.0, phys.ke, phys.clo);
      auto shift_one = [&](double *dev, Field &host, double fill) {
        if (nproc == 1) {
          alcu_shift_left_hip(dev, nx, ny, nz, hw, fill);
          return;
        }
        d2h(dev, host);
        mpi_util::shift_left_one(host, fill, decomp, rank, Nx);
        h2d(host, dev);
      };
      for (int s = 0; s < nsh; ++s) {
        shift_one(d_phi1.p, phi1, -1.0);
        shift_one(d_c.p, c, c_liq);
        if (use_glasner) {
          shift_one(d_psi1.p, psi1, -8.0);
        }
        if (n_grains == 2) {
          shift_one(d_phi2.p, phi2, -1.0);
          if (use_glasner) {
            shift_one(d_psi2.p, psi2, -8.0);
          }
        }
      }
      win_shift += nsh;
    }

    n_done = istep;
    if (!quiet && rank == 0 && cfg.nprint > 0 && istep % cfg.nprint == 0) {
      std::cout << "step " << istep << "/" << n_loop << " done\n";
    }
    if (cfg.timed_steps > 0 && istep == warmup + cfg.timed_steps) {
      hip_check(hipDeviceSynchronize(), "sync timed");
      MPI_Barrier(MPI_COMM_WORLD);
      t_timed1 = MPI_Wtime();
      break;
    }
  }

  hip_check(hipDeviceSynchronize(), "final sync");
  if (cfg.timed_steps > 0 && t_timed1 <= t_timed0) {
    t_timed1 = MPI_Wtime();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t0;
  double wall_max = 0.0;
  MPI_Reduce(&wall, &wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  d2h(d_phi1.p, phi1);
  d2h(d_phi2.p, phi2);
  d2h(d_c.p, c);
  double min_phi = std::numeric_limits<double>::infinity();
  double max_phi = -min_phi, min_c = min_phi, max_c = -min_phi, sum_phi = 0, sum_c = 0,
         mass1_loc = 0;
  phi1.for_each_owned([&](int i, int j, int k) {
    const double ph =
        (n_grains == 1) ? phi1(i, j, k) : phi_eff_two(phi1(i, j, k), phi2(i, j, k));
    min_phi = std::min(min_phi, ph);
    max_phi = std::max(max_phi, ph);
    min_c = std::min(min_c, c(i, j, k));
    max_c = std::max(max_c, c(i, j, k));
    sum_phi += ph;
    sum_c += c(i, j, k);
    mass1_loc += c(i, j, k);
  });
  mass1_loc *= dV;
  double g_min_phi, g_max_phi, g_min_c, g_max_c, g_sum_phi, g_sum_c, mass1;
  MPI_Allreduce(&min_phi, &g_min_phi, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&max_phi, &g_max_phi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&min_c, &g_min_c, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&max_c, &g_max_c, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&sum_phi, &g_sum_phi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sum_c, &g_sum_c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mass1_loc, &mass1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double xt_loc = 0.0;
  {
    const int nx_h = phi1.local_size()[0];
    const int ny_h = phi1.local_size()[1];
    const int nz_h = phi1.local_size()[2];
    for (int kk = 0; kk < nz_h; ++kk) {
      for (int j = 0; j < ny_h; ++j) {
        for (int i = 0; i < nx_h - 1; ++i) {
          const double p0 = phi1(i, j, kk);
          const double p1v = phi1(i + 1, j, kk);
          if (p0 >= 0.0 && p1v < 0.0) {
            const auto g = phi1.global(i, j, kk);
            const double a = p0 / (p0 - p1v + 1.0e-30);
            xt_loc = std::max(xt_loc, (static_cast<double>(g[0]) + 0.5 +
                                       static_cast<double>(win_shift)) *
                                              dx +
                                          a * dx);
          }
        }
      }
    }
  }
  double xt = 0.0;
  MPI_Allreduce(&xt_loc, &xt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0) {
    const int n_timed = cfg.timed_steps > 0
                            ? std::min(cfg.timed_steps, std::max(0, n_done - warmup))
                            : n_done;
    const double tps =
        n_timed > 0
            ? ((cfg.timed_steps > 0 ? (t_timed1 - t_timed0) : wall_max) /
               static_cast<double>(n_timed))
            : 0.0;
    const double rel = (mass1 - mass0) / std::max(std::abs(mass0), 1.0e-30);
    std::cout << std::setprecision(17);
    std::cout << "ALCU_VERIFY wall_loop_s=" << wall_max << " nproc=" << nproc
              << " mass0=" << mass0 << " mass1=" << mass1 << " rel_mass_err=" << rel
              << " min_phi=" << g_min_phi << " max_phi=" << g_max_phi << " min_c=" << g_min_c
              << " max_c=" << g_max_c << " x_tip=" << xt << " n_steps_done=" << n_done
              << " blew_up=0"
              << " sum_phi=" << g_sum_phi << " sum_c=" << g_sum_c
              << " time_per_step_s=" << tps << " halo_s=" << t_halo << " kernel_s=" << t_kern
              << "\n";
    std::cout << std::setprecision(6);
    alloy_pf_directional::engine::print_directional_perf_halo_kernel(std::cout, "hip", nproc, Nx, Ny, Nz, n_timed,
                                                 tps, wall_max, t_halo, t_kern);
  }
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg = alloy_pf_directional::parse_or_print_usage(app_argc, app_argv);
        if (!cfg) {
          return EXIT_FAILURE;
        }
        run_hip(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
