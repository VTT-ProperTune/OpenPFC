// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_pf_directional_mpi.cpp
 * @brief MPI CPU Al-Cu FTA (regular grid). Halo width 1.
 *        2D (`Nz=1`): `Full2D` (8 in-plane dirs, Ji \(\bar S_{2,1}\)).
 *        3D: 26-neighbor `Full3D`. Persistent: φ/ψ, c, ∂tφ, e^u, u.
 *        Nodal α/β filled each solute step. Recomputed: anisotropy fluxes.
 *
 * Nz=1 + n_dim=2 is the 2D-equivalent path (compare to alloy_pf_directional_openmp).
 * Classic AMR is not implemented.
 */

#include <alloy_pf_directional/cli.hpp>
#include <alloy_pf_directional/engine.hpp>
#include <alloy_pf_directional/isotropic_fd.hpp>
#include <alloy_pf_directional/mpi_support.hpp>
#include <alloy_pf_directional/noise.hpp>
#include <alloy_pf_directional/recompute.hpp>
#include <alloy_pf_directional/window.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Field = pfc::data::Field<double, pfc::HostSpace>;
using alloy_pf_directional::mpi_util::apply_noflux;
using alloy_pf_directional::mpi_util::init_fields;
using alloy_pf_directional::mpi_util::owns_hi;
using alloy_pf_directional::mpi_util::owns_lo;
using alloy_pf_directional::mpi_util::shift_left_one;

void exchange_and_bc(pfc::comm::HaloExchange<pfc::HostSpace, double> &halo, Field &f,
                     bool noflux_x, bool noflux_y, bool noflux_z, int Nx, int Ny, int Nz,
                     bool dim3) {
  halo.exchange();
  apply_noflux(f, noflux_x, noflux_y, noflux_z, Nx, Ny, Nz, dim3);
}

double local_tip(const Field &phi, double dx, int shift_cells) {
  double xt = 0.0;
  const int nx = phi.local_size()[0];
  const int ny = phi.local_size()[1];
  const int nz = phi.local_size()[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        const double p0 = phi(i, j, k);
        const double p1 = phi(i + 1, j, k);
        if (p0 >= 0.0 && p1 < 0.0) {
          const auto g = phi.global(i, j, k);
          const double a = p0 / (p0 - p1 + 1.0e-30);
          xt = std::max(xt, (static_cast<double>(g[0]) + 0.5 + static_cast<double>(shift_cells)) * dx +
                                a * dx);
        }
      }
    }
  }
  return xt;
}

void run_mpi(const alloy_pf_directional::RunConfig &cfg, int rank, int nproc) {
  using namespace alloy_pf_directional;
  const Physics phys = cfg.phys;
  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int Nz = cfg.Nz < 1 ? 1 : cfg.Nz;
  const bool dim3 = (phys.n_dim >= 3 && Nz > 1);
  const double dx = phys.dx;
  const double dt = phys.dt;
  const double inv_dx = 1.0 / dx;
  const double k = phys.ke;
  const double clo = phys.clo;
  const Mat3 R1 = bunge_crystal_to_lab(phys.phi1_g1, phys.Phi_g1, phys.phi2_g1);
  const Mat3 R2 = bunge_crystal_to_lab(phys.phi1_g2, phys.Phi_g2, phys.phi2_g2);
  const bool use_glasner = cfg.use_glasner;
  const bool use_iso = cfg.use_isotropic;
  const bool periodic_y = cfg.periodic_y;
  const bool periodic_z = dim3 && cfg.periodic_z;
  const int n_grains = cfg.n_grains;
  const bool skip_io = env_on("OPENPFC_ALCU_SKIP_PNG", false) ||
                       env_on("OPENPFC_ALCU_SKIP_VTK", false) || cfg.timed_steps > 0;
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
  Field dc = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field eu_f = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field u_f = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field a_diff = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field a_at_f = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  Field beta_at = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
  const bool store_eu = cfg.store_eu;

  pfc::comm::HaloExchangeOptions halo_opt;
  halo_opt.connectivity = pfc::comm::HaloConnectivity::Full;
  halo_opt.directions = dim3 ? pfc::halo::presets::Full3D() : pfc::halo::presets::Full2D();
  auto make_halo = [&](Field &f, int exchange_base) {
    pfc::comm::HaloExchangeOptions opt = halo_opt;
    opt.exchange_base = exchange_base;
    return pfc::comm::HaloExchange<pfc::HostSpace, double>(f, decomp, rank, MPI_COMM_WORLD,
                                                            opt);
  };
  auto h_phi1 = make_halo(phi1, 0);
  auto h_phi2 = make_halo(phi2, 30);
  auto h_psi1 = make_halo(psi1, 60);
  auto h_psi2 = make_halo(psi2, 90);
  auto h_c = make_halo(c, 120);
  auto h_d1 = make_halo(dphi1, 150);
  auto h_d2 = make_halo(dphi2, 180);

  init_fields(phi1, phi2, psi1, psi2, c, cfg, use_glasner, dim3);

  const bool noflux_x = true;
  const bool noflux_y = !periodic_y;
  const bool noflux_z = dim3 && !periodic_z;
  auto bc = [&](auto &halo, Field &f) {
    exchange_and_bc(halo, f, noflux_x, noflux_y, noflux_z, Nx, Ny, Nz, dim3);
  };
  bc(h_phi1, phi1);
  if (n_grains == 2) {
    bc(h_phi2, phi2);
  }
  bc(h_c, c);
  if (use_glasner) {
    bc(h_psi1, psi1);
    if (n_grains == 2) {
      bc(h_psi2, psi2);
    }
  }

  const int nx = phi1.local_size()[0];
  const int ny = phi1.local_size()[1];
  const int nz = phi1.local_size()[2];
  const double dV = dim3 ? dx * dx * dx : dx * dx;
  double mass0_loc = 0.0;
  c.for_each_owned([&](int i, int j, int kcell) { mass0_loc += c(i, j, kcell); });
  mass0_loc *= dV;
  double mass0 = 0.0;
  MPI_Allreduce(&mass0_loc, &mass0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int win_shift = 0;
  const double lab_Lx = cfg.lab_Lx > 0.0 ? cfg.lab_Lx : (static_cast<double>(Nx) - 0.5) * dx;
  BlockSkip skip =
      make_block_skip(nx, ny, nz, cfg.block_skip, cfg.block_skip_tol_phi, cfg.block_skip_tol_c,
                      cfg.block_skip_refresh);
  // Block skip uses 1-based PadField indexing; map 0-based Field → +1 in is_active.
  auto active = [&](int i, int j, int kcell) {
    if (!skip.enabled()) {
      return true;
    }
    return skip.is_active(i + 1, j + 1, kcell + 1);
  };

  if (rank == 0) {
    std::filesystem::create_directories(cfg.output_dir);
    std::ofstream meta(cfg.output_dir + "/meta.txt");
    meta << std::setprecision(16);
    meta << "backend mpi_cpu\nNx " << Nx << "\nNy " << Ny << "\nNz " << Nz << "\nn_dim "
         << phys.n_dim << "\nnproc " << nproc << "\ndt " << dt << "\n";
    meta << "stored_fields phi[,psi] c dphi dc_scratch eu u a_diff a_at beta_at\nrecomputed_fields aniso_fluxes gradients\n";
    meta << "amr_deferred 1\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  const int n_loop =
      cfg.timed_steps > 0 ? (cfg.warmup_steps + cfg.timed_steps) : cfg.n_steps;
  const int warmup = cfg.timed_steps > 0 ? cfg.warmup_steps : 0;
  double t_halo = 0.0, t_kern = 0.0, t_timed0 = 0.0, t_timed1 = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop0 = MPI_Wtime();
  int n_done = 0;
  bool hit_right = false;

  auto fill_eu_u = [&]() {
    const int k0 = dim3 ? -1 : 0;
    const int k1 = dim3 ? nz : nz - 1;
    for (int kcell = k0; kcell <= k1; ++kcell) {
      for (int j = -1; j <= ny; ++j) {
        for (int i = -1; i <= nx; ++i) {
          const double euv = eu_at(phi1, phi2, c, n_grains, k, clo, i, j, kcell);
          eu_f(i, j, kcell) = euv;
          u_f(i, j, kcell) = u_from_eu(euv);
        }
      }
    }
  };

  auto fill_solute_nodal = [&]() {
    const int k0 = dim3 ? -1 : 0;
    const int k1 = dim3 ? nz : nz - 1;
    for (int kcell = k0; kcell <= k1; ++kcell) {
      for (int j = -1; j <= ny; ++j) {
        for (int i = -1; i <= nx; ++i) {
          const double ph = phi_at(phi1, phi2, n_grains, i, j, kcell);
          const double euv = store_eu ? eu_f(i, j, kcell)
                                      : eu_from_phi_c(ph, c(i, j, kcell), k, clo);
          a_diff(i, j, kcell) = phys.DL * c(i, j, kcell) * q_of(ph, k);
          const double dte = dphi1(i, j, kcell) + dphi2(i, j, kcell);
          a_at_f(i, j, kcell) =
              a_at_nodal(ph, euv, dte, phys.a_at, phys.A_trap, phys.W0, clo, k, use_glasner);
          beta_at(i, j, kcell) =
              use_glasner ? beta_glasner_from_phi(ph) : ph;
          if (!store_eu) {
            u_f(i, j, kcell) = u_from_eu(euv);
          }
        }
      }
    }
  };

  for (int istep = 1; istep <= n_loop; ++istep) {
    if (cfg.timed_steps > 0 && istep == warmup + 1) {
      MPI_Barrier(MPI_COMM_WORLD);
      t_timed0 = MPI_Wtime();
      t_halo = 0.0;
      t_kern = 0.0;
    }
    const double t = dt * static_cast<double>(istep - 1);
    const double th0 = MPI_Wtime();
    bc(h_phi1, phi1);
    if (n_grains == 2) {
      bc(h_phi2, phi2);
    }
    bc(h_c, c);
    if (use_glasner) {
      bc(h_psi1, psi1);
      if (n_grains == 2) {
        bc(h_psi2, psi2);
      }
    }
    t_halo += MPI_Wtime() - th0;
    const Field &pf1 = use_glasner ? psi1 : phi1;
    const Field &pf2 = use_glasner ? psi2 : phi2;
    const double tk0 = MPI_Wtime();
    if (store_eu) {
      fill_eu_u();
    }
    if (skip.enabled() && istep % skip.refresh == 0) {
      std::fill(skip.active.begin(), skip.active.end(), 0);
      for (int kcell = 0; kcell < nz; ++kcell) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            const double ph =
                (n_grains == 1) ? phi1(i, j, kcell)
                                : phi_eff_two(phi1(i, j, kcell), phi2(i, j, kcell));
            if (std::abs(ph * ph - 1.0) > skip.tol_phi ||
                std::abs(c(i, j, kcell) - c_eq(ph, k, clo)) > skip.tol_c) {
              const int bi = i / skip.bs;
              const int bj = j / skip.bs;
              const int bk = kcell / skip.bs;
              skip.active[static_cast<std::size_t>(bi + bj * skip.nbx + bk * skip.nbx * skip.nby)] =
                  1;
            }
          }
        }
      }
    }

    auto step_grain = [&](Field &dphi, const Field &pf, const Field &phi_self,
                          const Field &phi_other, const Mat3 &R, int grain_id) {
      for (int kcell = 0; kcell < nz; ++kcell) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            if (!active(i, j, kcell)) {
              dphi(i, j, kcell) = 0.0;
              continue;
            }
            const double gx = 0.5 * inv_dx * (pf(i + 1, j, kcell) - pf(i - 1, j, kcell));
            const double gy = 0.5 * inv_dx * (pf(i, j + 1, kcell) - pf(i, j - 1, kcell));
            const double gz =
                dim3 ? 0.5 * inv_dx * (pf(i, j, kcell + 1) - pf(i, j, kcell - 1)) : 0.0;
            double jxv = 0.0, jyv = 0.0, jzv = 0.0, tau = 0.0, A = 0.0;
            cubic_aniso_from_grad(gx, gy, gz, phys.eps_c, phys.W0, phys.tau0, R, jxv, jyv,
                                  jzv, tau, A);
            (void)jxv;
            (void)jyv;
            (void)jzv;
            const double euv = store_eu ? eu_f(i, j, kcell)
                                        : eu_at(phi1, phi2, c, n_grains, k, clo, i, j, kcell);
            tau = tau_with_u_corr(tau, phys, euv);
            const int gi = phi1.global(i, j, kcell)[0];
            double aniso = (flux_aniso_x(pf, i, j, kcell, nx, inv_dx, dim3, phys, R) -
                            flux_aniso_x(pf, i - 1, j, kcell, nx, inv_dx, dim3, phys, R)) *
                           inv_dx;
            aniso += (flux_aniso_y(pf, i, j, kcell, ny, inv_dx, dim3, periodic_y, phys, R) -
                      flux_aniso_y(pf, i, j - 1, kcell, ny, inv_dx, dim3, periodic_y, phys, R)) *
                     inv_dx;
            if (dim3) {
              aniso +=
                  (flux_aniso_z(pf, i, j, kcell, nz, inv_dx, periodic_z, phys, R) -
                   flux_aniso_z(pf, i, j, kcell - 1, nz, inv_dx, periodic_z, phys, R)) *
                  inv_dx;
            }
            const double ph = phi_self(i, j, kcell);
            const double x = (static_cast<double>(gi) + 0.5 + static_cast<double>(win_shift)) * dx;
            const double therm = thermal_drive(phys, x, t);
            const double grain =
                grain_repulsion(ph, phi_other(i, j, kcell), omega_used(phys, therm));
            double mag2 = gx * gx + gy * gy + gz * gz;
            if (use_iso) {
              const double L_iso = alloy_pf_directional::iso::laplacian_iso(pf, i, j, kcell, dx, dim3);
              const double L_std = alloy_pf_directional::iso::laplacian_std(pf, i, j, kcell, dx, dim3);
              mag2 = alloy_pf_directional::iso::grad2_iso(pf, i, j, kcell, dx, dim3);
              aniso += phys.W0 * phys.W0 * A * A * (L_iso - L_std);
            }
            if (use_glasner) {
              const double W = phys.W0 * A;
              const double sqrt2 = std::sqrt(2.0);
              const double bulk =
                  sqrt2 * ph * (1.0 - W * W * mag2) -
                  sqrt2 * (1.0 - ph * ph) * (phys.lambda / (1.0 - k)) * (euv - 1.0 - therm);
              dphi(i, j, kcell) =
                  (aniso + bulk) / tau + grain_dpsi_dt(grain, tau, ph, dt);
            } else {
              const double bulk = -f_prime(ph) - (phys.lambda / (1.0 - k)) * g_prime(ph) *
                                                     (euv - 1.0 - therm);
              dphi(i, j, kcell) = (aniso + bulk + grain) / tau;
            }
            if (cfg.noise_F0 > 0.0) {
              const int knoise = (Nz == 1) ? 0 : (phi1.global(i, j, kcell)[2] + 1);
              const int inoise = phi1.global(i, j, kcell)[0] + 1;
              const int jnoise = phi1.global(i, j, kcell)[1] + 1;
              const double xi = gaussian_n01(cfg.noise_seed, istep, inoise, jnoise, knoise, grain_id);
              dphi(i, j, kcell) +=
                  use_glasner
                      ? fdt_psi_noise_rate(cfg.noise_F0, phys.W0, phys.n_dim, tau, dt, dV, ph, xi)
                      : fdt_phi_noise_rate(cfg.noise_F0, phys.W0, phys.n_dim, tau, dt, dV, ph, xi);
            }
          }
        }
      }
    };

    step_grain(dphi1, pf1, phi1, phi2, R1, 1);
    if (n_grains == 2) {
      step_grain(dphi2, pf2, phi2, phi1, R2, 2);
    }

    for (int kcell = 0; kcell < nz; ++kcell) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          if (!active(i, j, kcell)) {
            continue;
          }
          if (use_glasner) {
            const double p1_old = phi1(i, j, kcell);
            psi1(i, j, kcell) += dt * dphi1(i, j, kcell);
            psi1(i, j, kcell) = std::max(-8.0, std::min(8.0, psi1(i, j, kcell)));
            phi1(i, j, kcell) = phi_from_psi(psi1(i, j, kcell));
            if (n_grains == 2) {
              dphi1(i, j, kcell) = (phi1(i, j, kcell) - p1_old) / dt;
              const double p2_old = phi2(i, j, kcell);
              psi2(i, j, kcell) += dt * dphi2(i, j, kcell);
              psi2(i, j, kcell) = std::max(-8.0, std::min(8.0, psi2(i, j, kcell)));
              phi2(i, j, kcell) = phi_from_psi(psi2(i, j, kcell));
              dphi2(i, j, kcell) = (phi2(i, j, kcell) - p2_old) / dt;
            } else {
              dphi1(i, j, kcell) = dphi_dpsi_from_phi(p1_old) * dphi1(i, j, kcell);
            }
          } else {
            phi1(i, j, kcell) += dt * dphi1(i, j, kcell);
            if (n_grains == 2) {
              phi2(i, j, kcell) += dt * dphi2(i, j, kcell);
            }
          }
        }
      }
    }
    bc(h_phi1, phi1);
    if (n_grains == 2) {
      bc(h_phi2, phi2);
    }
    bc(h_d1, dphi1);
    if (n_grains == 2) {
      bc(h_d2, dphi2);
    }
    if (use_glasner) {
      bc(h_psi1, psi1);
      if (n_grains == 2) {
        bc(h_psi2, psi2);
      }
    }
    if (store_eu) {
      fill_eu_u();
    }

    if (!use_iso && rank == 0 && istep == 1) {
      std::cerr << "alloy_pf_directional_mpi: ISO=0 is OpenMP-only; MPI keeps Ji isotropic solute\n";
    }
    if (use_iso) {
      fill_solute_nodal();
      for (int kcell = 0; kcell < nz; ++kcell) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            if (!active(i, j, kcell)) {
              continue;
            }
            const double Dd =
                alloy_pf_directional::iso::div_alpha_grad(a_diff, u_f, i, j, kcell, dx, dim3);
            const double Dat =
                alloy_pf_directional::iso::div_alpha_grad(a_at_f, beta_at, i, j, kcell, dx, dim3);
            dc(i, j, kcell) = Dd + Dat;
          }
        }
      }
      for (int kcell = 0; kcell < nz; ++kcell) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            if (!active(i, j, kcell)) {
              continue;
            }
            c(i, j, kcell) += dt * dc(i, j, kcell);
            if (std::isfinite(c(i, j, kcell)) && c(i, j, kcell) < kCMin &&
                c(i, j, kcell) > -kFieldBlowAbs) {
              c(i, j, kcell) = kCMin;
            }
          }
        }
      }
    }
    t_kern += MPI_Wtime() - tk0;

    if (cfg.window_enable) {
      double xt_loc = local_tip(phi1, dx, win_shift);
      double xt = 0.0;
      MPI_Allreduce(&xt_loc, &xt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const int nsh = window_shift_count(xt, win_shift, dx, cfg.window_margin_left, Nx,
                                         cfg.window_margin_right);
      const double c_liq = c_eq(-1.0, k, clo);
      for (int s = 0; s < nsh; ++s) {
        shift_left_one(phi1, -1.0, decomp, rank, Nx);
        shift_left_one(c, c_liq, decomp, rank, Nx);
        if (n_grains == 2) {
          shift_left_one(phi2, -1.0, decomp, rank, Nx);
        }
        if (use_glasner) {
          shift_left_one(psi1, -8.0, decomp, rank, Nx);
          if (n_grains == 2) {
            shift_left_one(psi2, -8.0, decomp, rank, Nx);
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
      MPI_Barrier(MPI_COMM_WORLD);
      t_timed1 = MPI_Wtime();
      break;
    }
    int hit_c = 0;
    if (cfg.stop_on_far_c && owns_hi(c, 0, Nx)) {
      const double tol = kWallCRel * std::max(std::abs(phys.clo), 1.0e-12);
      for (int kcell = 0; kcell < nz && !hit_c; ++kcell) {
        for (int j = 0; j < ny; ++j) {
          if (std::abs(c(nx - 1, j, kcell) - phys.clo) > tol) {
            hit_c = 1;
            break;
          }
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &hit_c, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (hit_c) {
      break;
    }
    if (cfg.stop_on_right && !cfg.window_enable && owns_hi(phi1, 0, Nx)) {
      for (int kcell = 0; kcell < nz && !hit_right; ++kcell) {
        for (int j = 0; j < ny; ++j) {
          const double ph = (n_grains == 1) ? phi1(nx - 1, j, kcell)
                                            : phi_eff_two(phi1(nx - 1, j, kcell),
                                                          phi2(nx - 1, j, kcell));
          if (ph > 0.0) {
            hit_right = true;
          }
        }
      }
    }
    int hit_i = hit_right ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &hit_i, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    hit_right = hit_i != 0;
    if (hit_right) {
      break;
    }
    if (cfg.stop_on_right && cfg.window_enable) {
      double xt_loc = local_tip(phi1, dx, win_shift);
      double xt = 0.0;
      MPI_Allreduce(&xt_loc, &xt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (xt >= lab_Lx - 2.0 * phys.W0) {
        hit_right = true;
        break;
      }
    }
    (void)skip_io;
  }

  if (cfg.timed_steps > 0 && t_timed1 <= t_timed0) {
    t_timed1 = MPI_Wtime();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t_loop0;
  double wall_max = 0.0;
  MPI_Reduce(&wall, &wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double min_phi = std::numeric_limits<double>::infinity();
  double max_phi = -min_phi;
  double min_c = min_phi;
  double max_c = -min_phi;
  double sum_phi = 0.0, sum_c = 0.0, mass1_loc = 0.0;
  phi1.for_each_owned([&](int i, int j, int kcell) {
    const double ph =
        (n_grains == 1) ? phi1(i, j, kcell) : phi_eff_two(phi1(i, j, kcell), phi2(i, j, kcell));
    min_phi = std::min(min_phi, ph);
    max_phi = std::max(max_phi, ph);
    min_c = std::min(min_c, c(i, j, kcell));
    max_c = std::max(max_c, c(i, j, kcell));
    sum_phi += ph;
    sum_c += c(i, j, kcell);
    mass1_loc += c(i, j, kcell);
  });
  mass1_loc *= dV;
  double g_min_phi = 0, g_max_phi = 0, g_min_c = 0, g_max_c = 0, g_sum_phi = 0, g_sum_c = 0,
         mass1 = 0;
  MPI_Allreduce(&min_phi, &g_min_phi, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&max_phi, &g_max_phi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&min_c, &g_min_c, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&max_c, &g_max_c, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&sum_phi, &g_sum_phi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sum_c, &g_sum_c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mass1_loc, &mass1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double xt_loc = local_tip(phi1, dx, win_shift);
  double xt = 0.0;
  MPI_Allreduce(&xt_loc, &xt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0) {
    const int n_timed = cfg.timed_steps > 0
                            ? std::min(cfg.timed_steps, std::max(0, n_done - warmup))
                            : n_done;
    const double tps =
        n_timed > 0
            ? ((cfg.timed_steps > 0 ? (t_timed1 - t_timed0) : wall_max) / static_cast<double>(n_timed))
            : 0.0;
    const double rel_mass = (mass1 - mass0) / std::max(std::abs(mass0), 1.0e-30);
    std::cout << std::setprecision(17);
    std::cout << "ALCU_VERIFY wall_loop_s=" << wall_max << " nproc=" << nproc
              << " mass0=" << mass0 << " mass1=" << mass1 << " rel_mass_err=" << rel_mass
              << " min_phi=" << g_min_phi << " max_phi=" << g_max_phi << " min_c=" << g_min_c
              << " max_c=" << g_max_c << " x_tip=" << xt << " n_steps_done=" << n_done
              << " hit_right=" << (hit_right ? 1 : 0) << " blew_up=0"
              << " sum_phi=" << g_sum_phi << " sum_c=" << g_sum_c
              << " time_per_step_s=" << tps << " halo_s=" << t_halo << " kernel_s=" << t_kern
              << "\n";
    std::cout << std::setprecision(6);
    alloy_pf_directional::engine::print_directional_perf_halo_kernel(std::cout, "mpi", nproc, cfg.Nx, cfg.Ny,
                                                 cfg.Nz, n_timed, tps, wall_max, t_halo,
                                                 t_kern);
    std::ofstream meta(cfg.output_dir + "/meta.txt", std::ios::app);
    meta << std::setprecision(16);
    meta << "n_steps_done " << n_done << "\nx_tip " << xt << "\nsum_phi " << g_sum_phi
         << "\nsum_c " << g_sum_c << "\ntime_per_step_s " << tps << "\n";
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
        if (rank == 0) {
          std::filesystem::create_directories(cfg->output_dir);
          const auto &p = cfg->phys;
          std::cout << std::setprecision(10);
          std::cout << "ALCU_FTA backend=mpi_cpu nproc=" << nproc << " Nx=" << cfg->Nx
                    << " Ny=" << cfg->Ny << " Nz=" << cfg->Nz << " n_dim=" << p.n_dim
                    << " dt=" << p.dt << " out=" << cfg->output_dir << "\n";
        }
        run_mpi(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
