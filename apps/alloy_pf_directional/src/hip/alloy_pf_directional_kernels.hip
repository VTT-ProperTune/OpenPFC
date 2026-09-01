// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_HIP)

#include <alloy_pf_directional/device_math.hpp>
#include <alloy_pf_directional/device_step_hip.hpp>
#include <alloy_pf_directional/noise.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace alloy_pf_directional {
namespace {

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

__device__ inline std::size_t brick_lin(int i, int j, int k, int hw, int nxp, int nyp) noexcept {
  const auto h = static_cast<std::size_t>(hw);
  return (static_cast<std::size_t>(i) + h) +
         (static_cast<std::size_t>(j) + h) * static_cast<std::size_t>(nxp) +
         (static_cast<std::size_t>(k) + h) * static_cast<std::size_t>(nxp) *
             static_cast<std::size_t>(nyp);
}

struct Brick {
  const double *a;
  int hw, nxp, nyp;
  __device__ double operator()(int i, int j, int k) const noexcept {
    return a[brick_lin(i, j, k, hw, nxp, nyp)];
  }
};

__device__ inline void aniso_from_grad(const HipPhys &P, double gx, double gy, double gz,
                                       double &jx, double &jy, double &jz, double &tau,
                                       double &A) {
  jx = jy = jz = tau = A = 0.0;
  dmath::cubic_aniso_from_grad(gx, gy, gz, P.eps_c, P.eps_k, P.W0, P.tau0, P.R, jx, jy, jz,
                               tau, A);
}

__device__ inline double aniso_jx_face(const Brick &pfv, int ii, int j, int k,
                                       const HipPhys &P) {
  const double gxx = P.inv_dx * (pfv(ii + 1, j, k) - pfv(ii, j, k));
  const double gyy =
      0.25 * P.inv_dx *
      (pfv(ii + 1, j + 1, k) + pfv(ii, j + 1, k) - pfv(ii + 1, j - 1, k) - pfv(ii, j - 1, k));
  const double gzz =
      P.dim3 ? 0.25 * P.inv_dx *
                   (pfv(ii + 1, j, k + 1) + pfv(ii, j, k + 1) - pfv(ii + 1, j, k - 1) -
                    pfv(ii, j, k - 1))
             : 0.0;
  double jx = 0, jy = 0, jz = 0, t = 0, A = 0;
  aniso_from_grad(P, gxx, gyy, gzz, jx, jy, jz, t, A);
  return jx;
}

__device__ inline double aniso_jy_face(const Brick &pfv, int i, int jj, int k,
                                       const HipPhys &P) {
  const double gyy = P.inv_dx * (pfv(i, jj + 1, k) - pfv(i, jj, k));
  const double gxx =
      0.25 * P.inv_dx *
      (pfv(i + 1, jj + 1, k) + pfv(i + 1, jj, k) - pfv(i - 1, jj + 1, k) - pfv(i - 1, jj, k));
  const double gzz =
      P.dim3 ? 0.25 * P.inv_dx *
                   (pfv(i, jj + 1, k + 1) + pfv(i, jj, k + 1) - pfv(i, jj + 1, k - 1) -
                    pfv(i, jj, k - 1))
             : 0.0;
  double jx = 0, jy = 0, jz = 0, t = 0, A = 0;
  aniso_from_grad(P, gxx, gyy, gzz, jx, jy, jz, t, A);
  return jy;
}

__device__ inline double aniso_jz_face(const Brick &pfv, int i, int j, int kk,
                                       const HipPhys &P) {
  const double gzz = P.inv_dx * (pfv(i, j, kk + 1) - pfv(i, j, kk));
  const double gxx =
      0.25 * P.inv_dx *
      (pfv(i + 1, j, kk + 1) + pfv(i + 1, j, kk) - pfv(i - 1, j, kk + 1) - pfv(i - 1, j, kk));
  const double gyy =
      0.25 * P.inv_dx *
      (pfv(i, j + 1, kk + 1) + pfv(i, j + 1, kk) - pfv(i, j - 1, kk + 1) - pfv(i, j - 1, kk));
  double jx = 0, jy = 0, jz = 0, t = 0, A = 0;
  aniso_from_grad(P, gxx, gyy, gzz, jx, jy, jz, t, A);
  return jz;
}

__global__ void grain_kernel(const double *pf, const double *phi_self, const double *phi_other,
                             const double *c, double *dphi, int nx, int ny, int nz, int hw,
                             HipPhys P, int istep, int grain_id) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const Brick pfv{pf, hw, nxp, nyp};
  const std::size_t cidx = brick_lin(i, j, k, hw, nxp, nyp);
  const double gx = 0.5 * P.inv_dx * (pfv(i + 1, j, k) - pfv(i - 1, j, k));
  const double gy = 0.5 * P.inv_dx * (pfv(i, j + 1, k) - pfv(i, j - 1, k));
  const double gz = P.dim3 ? 0.5 * P.inv_dx * (pfv(i, j, k + 1) - pfv(i, j, k - 1)) : 0.0;
  double jxv = 0, jyv = 0, jzv = 0, tau = 0, A = 0;
  dmath::cubic_aniso_from_grad(gx, gy, gz, P.eps_c, P.eps_k, P.W0, P.tau0, P.R, jxv, jyv, jzv,
                               tau, A);
  const double phs = phi_self[cidx];
  const double pho = phi_other[cidx];
  const double ph = (P.n_grains == 1) ? phs : dmath::phi_eff_two(phs, pho);
  const double euv = dmath::eu_from_phi_c(ph, c[cidx], P.ke, P.clo);
  tau = dmath::tau_with_u_corr(tau, P.tau_beta, P.tau_a2, P.tau0, euv);
  const double aniso_x = (aniso_jx_face(pfv, i, j, k, P) - aniso_jx_face(pfv, i - 1, j, k, P)) *
                         P.inv_dx;
  const double aniso_y = (aniso_jy_face(pfv, i, j, k, P) - aniso_jy_face(pfv, i, j - 1, k, P)) *
                         P.inv_dx;
  double aniso = aniso_x + aniso_y;
  if (P.dim3) {
    aniso += (aniso_jz_face(pfv, i, j, k, P) - aniso_jz_face(pfv, i, j, k - 1, P)) * P.inv_dx;
  }
  const double L_iso = dmath::laplacian_iso(pfv, i, j, k, P.dx, P.dim3);
  const double L_std = dmath::laplacian_std(pfv, i, j, k, P.dx, P.dim3);
  const double mag2 = dmath::grad2_iso(pfv, i, j, k, P.dx, P.dim3);
  aniso += P.W0 * P.W0 * A * A * (L_iso - L_std);
  const double x =
      (static_cast<double>(P.i0 + i) + 0.5 + static_cast<double>(P.shift_cells)) * P.dx;
  const double therm = dmath::thermal_drive(P.G, P.mle, P.clo, P.x_tl, P.Vp, x,
                                            P.dt * (istep - 1), P.ke, P.delta_iso);
  const double grain = dmath::grain_repulsion(
      phs, pho, dmath::omega_used(P.omega_zhong, P.omega, P.lambda, P.ke, therm));
  if (P.use_glasner) {
    const double W = P.W0 * A;
    const double sqrt2 = sqrt(2.0);
    const double bulk = sqrt2 * phs * (1.0 - W * W * mag2) -
                        sqrt2 * (1.0 - phs * phs) * (P.lambda / (1.0 - P.ke)) * (euv - 1.0 - therm);
    dphi[cidx] = (aniso + bulk) / tau + dmath::grain_dpsi_dt(grain, tau, phs, P.dt);
  } else {
    const double bulk = -dmath::f_prime(phs) -
                        (P.lambda / (1.0 - P.ke)) * dmath::g_prime(phs) * (euv - 1.0 - therm);
    dphi[cidx] = (aniso + bulk + grain) / tau;
  }
  if (P.noise_F0 > 0.0) {
    const int knoise = (nz == 1) ? 0 : (P.k0 + k + 1);
    const double xi =
        gaussian_n01(P.noise_seed, istep, P.i0 + i + 1, P.j0 + j + 1, knoise, grain_id);
    dphi[cidx] += P.use_glasner
                      ? fdt_psi_noise_rate(P.noise_F0, P.W0, P.n_dim, tau, P.dt, P.dV, phs, xi)
                      : fdt_phi_noise_rate(P.noise_F0, P.W0, P.n_dim, tau, P.dt, P.dV, phs, xi);
  }
}

__global__ void euler_kernel(double *psi, double *phi, double *dphi, double *psi2, double *phi2,
                             double *dphi2, int nx, int ny, int nz, int hw, double dt,
                             bool glasner, int n_grains) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const std::size_t cidx = brick_lin(i, j, k, hw, nxp, nyp);
  if (glasner) {
    const double p1_old = phi[cidx];
    psi[cidx] += dt * dphi[cidx];
    psi[cidx] = psi[cidx] < -8.0 ? -8.0 : (psi[cidx] > 8.0 ? 8.0 : psi[cidx]);
    phi[cidx] = dmath::phi_from_psi(psi[cidx]);
    if (n_grains == 2) {
      dphi[cidx] = (phi[cidx] - p1_old) / dt;
      const double p2_old = phi2[cidx];
      psi2[cidx] += dt * dphi2[cidx];
      psi2[cidx] = psi2[cidx] < -8.0 ? -8.0 : (psi2[cidx] > 8.0 ? 8.0 : psi2[cidx]);
      phi2[cidx] = dmath::phi_from_psi(psi2[cidx]);
      dphi2[cidx] = (phi2[cidx] - p2_old) / dt;
    } else {
      dphi[cidx] = dmath::dphi_dpsi_from_phi(p1_old) * dphi[cidx];
    }
  } else {
    phi[cidx] += dt * dphi[cidx];
    if (n_grains == 2) {
      phi2[cidx] += dt * dphi2[cidx];
    }
  }
}

__global__ void fill_eu_u_kernel(double *eu, double *u, const double *phi1, const double *phi2,
                                 const double *c, int nx, int ny, int nz, int hw, HipPhys P) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - hw;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - hw;
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) - hw;
  if (i < -hw || i >= nx + hw || j < -hw || j >= ny + hw || k < -hw || k >= nz + hw) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const std::size_t p = brick_lin(i, j, k, hw, nxp, nyp);
  const double ph = P.n_grains == 1 ? phi1[p] : dmath::phi_eff_two(phi1[p], phi2[p]);
  const double euv = dmath::eu_from_phi_c(ph, c[p], P.ke, P.clo);
  eu[p] = euv;
  u[p] = dmath::u_from_eu(euv);
}

__global__ void fill_solute_nodal_kernel(double *a_diff, double *a_at, double *beta, double *u,
                                         const double *phi1, const double *phi2, const double *c,
                                         const double *dphi1, const double *dphi2, const double *eu,
                                         int nx, int ny, int nz, int hw, HipPhys P) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - hw;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - hw;
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) - hw;
  if (i < -hw || i >= nx + hw || j < -hw || j >= ny + hw || k < -hw || k >= nz + hw) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const std::size_t p = brick_lin(i, j, k, hw, nxp, nyp);
  const double ph = P.n_grains == 1 ? phi1[p] : dmath::phi_eff_two(phi1[p], phi2[p]);
  const double euv = eu ? eu[p] : dmath::eu_from_phi_c(ph, c[p], P.ke, P.clo);
  a_diff[p] = P.DL * c[p] * dmath::q_of(ph, P.ke);
  const double dte = dphi1[p] + dphi2[p];
  a_at[p] = dmath::a_at_nodal(ph, euv, dte, P.a_at, P.A_trap, P.W0, P.clo, P.ke, P.use_glasner);
  beta[p] = P.use_glasner ? dmath::beta_glasner_from_phi(ph) : ph;
  if (u && !eu) {
    u[p] = dmath::u_from_eu(euv);
  }
}

__global__ void solute_kernel(double *dc, const double *a_diff, const double *a_at, const double *u,
                              const double *beta, int nx, int ny, int nz, int hw, HipPhys P) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const Brick ad{a_diff, hw, nxp, nyp};
  const Brick aa{a_at, hw, nxp, nyp};
  const Brick ub{u, hw, nxp, nyp};
  const Brick ba{beta, hw, nxp, nyp};
  const double Dd = dmath::div_alpha_grad(ad, ub, i, j, k, P.dx, P.dim3);
  const double Dat = dmath::div_alpha_grad(aa, ba, i, j, k, P.dx, P.dim3);
  dc[brick_lin(i, j, k, hw, nxp, nyp)] = Dd + Dat;
}

__global__ void add_c_kernel(double *c, const double *dc, int nx, int ny, int nz, int hw,
                             double dt) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  const std::size_t p = brick_lin(i, j, k, hw, nx + 2 * hw, ny + 2 * hw);
  c[p] += dt * dc[p];
  if (isfinite(c[p]) && c[p] < 1.0e-16 && c[p] > -1.0e6) {
    c[p] = 1.0e-16;
  }
}

__global__ void noflux_x_kernel(double *f, int nx, int ny, int nz, int hw, bool lo, bool hi) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (j >= ny + 2 * hw || k >= nz + 2 * hw) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int jj = j - hw;
  const int kk = k - hw;
  if (lo) {
    f[brick_lin(-1, jj, kk, hw, nxp, nyp)] = f[brick_lin(0, jj, kk, hw, nxp, nyp)];
  }
  if (hi) {
    f[brick_lin(nx, jj, kk, hw, nxp, nyp)] = f[brick_lin(nx - 1, jj, kk, hw, nxp, nyp)];
  }
}

__global__ void shift_left_kernel(double *f, int nx, int ny, int nz, int hw, double fill) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (j >= ny || k >= nz) {
    return;
  }
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  for (int i = 0; i < nx - 1; ++i) {
    f[brick_lin(i, j, k, hw, nxp, nyp)] = f[brick_lin(i + 1, j, k, hw, nxp, nyp)];
  }
  f[brick_lin(nx - 1, j, k, hw, nxp, nyp)] = fill;
}

dim3 grid3(int nx, int ny, int nz) {
  constexpr int bx = 8, by = 8, bz = 4;
  return dim3((nx + bx - 1) / bx, (ny + by - 1) / by, (nz + bz - 1) / bz);
}

dim3 grid_pad(int nx, int ny, int nz, int hw) {
  return grid3(nx + 2 * hw, ny + 2 * hw, nz + 2 * hw);
}

} // namespace

void alcu_grain_step_hip(const double *pf, const double *phi_self, const double *phi_other,
                         const double *c, double *dphi, int nx, int ny, int nz, int hw,
                         const HipPhys &P, int istep, int grain_id) {
  grain_kernel<<<grid3(nx, ny, nz), dim3(8, 8, 4)>>>(pf, phi_self, phi_other, c, dphi, nx, ny,
                                                     nz, hw, P, istep, grain_id);
  hip_check(hipGetLastError(), "grain_kernel");
}

void alcu_euler_hip(double *psi, double *phi, double *dphi, double *psi2, double *phi2,
                    double *dphi2, int nx, int ny, int nz, int hw, double dt, bool glasner,
                    int n_grains) {
  euler_kernel<<<grid3(nx, ny, nz), dim3(8, 8, 4)>>>(psi, phi, dphi, psi2, phi2, dphi2, nx, ny,
                                                     nz, hw, dt, glasner, n_grains);
  hip_check(hipGetLastError(), "euler_kernel");
}

void alcu_fill_eu_u_hip(double *eu, double *u, const double *phi1, const double *phi2,
                        const double *c, int nx, int ny, int nz, int hw, const HipPhys &P) {
  fill_eu_u_kernel<<<grid_pad(nx, ny, nz, hw), dim3(8, 8, 4)>>>(eu, u, phi1, phi2, c, nx, ny, nz,
                                                               hw, P);
  hip_check(hipGetLastError(), "fill_eu_u_kernel");
}

void alcu_fill_solute_nodal_hip(double *a_diff, double *a_at, double *beta, double *u,
                                const double *phi1, const double *phi2, const double *c,
                                const double *dphi1, const double *dphi2, const double *eu,
                                int nx, int ny, int nz, int hw, const HipPhys &P) {
  fill_solute_nodal_kernel<<<grid_pad(nx, ny, nz, hw), dim3(8, 8, 4)>>>(
      a_diff, a_at, beta, u, phi1, phi2, c, dphi1, dphi2, eu, nx, ny, nz, hw, P);
  hip_check(hipGetLastError(), "fill_solute_nodal_kernel");
}

void alcu_solute_iso_hip(double *c, double *dc, const double *a_diff, const double *a_at,
                         const double *u, const double *beta, int nx, int ny, int nz, int hw,
                         const HipPhys &P) {
  solute_kernel<<<grid3(nx, ny, nz), dim3(8, 8, 4)>>>(dc, a_diff, a_at, u, beta, nx, ny, nz, hw,
                                                      P);
  hip_check(hipGetLastError(), "solute_kernel");
  add_c_kernel<<<grid3(nx, ny, nz), dim3(8, 8, 4)>>>(c, dc, nx, ny, nz, hw, P.dt);
  hip_check(hipGetLastError(), "add_c_kernel");
}

void alcu_noflux_x_hip(double *f, int nx, int ny, int nz, int hw, bool lo, bool hi) {
  const int jn = ny + 2 * hw;
  const int kn = nz + 2 * hw;
  noflux_x_kernel<<<dim3((jn + 15) / 16, (kn + 15) / 16), dim3(16, 16)>>>(f, nx, ny, nz, hw, lo,
                                                                         hi);
  hip_check(hipGetLastError(), "noflux_x_kernel");
}

void alcu_shift_left_hip(double *f, int nx, int ny, int nz, int hw, double fill) {
  shift_left_kernel<<<dim3((ny + 15) / 16, (nz + 15) / 16), dim3(16, 16)>>>(f, nx, ny, nz, hw,
                                                                           fill);
  hip_check(hipGetLastError(), "shift_left_kernel");
}

} // namespace alloy_pf_directional

#endif
