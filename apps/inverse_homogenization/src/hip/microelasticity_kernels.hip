// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "microelasticity HIP kernels require HIP"
#endif

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

__device__ double2 cscale(double2 a, double s) { return {a.x * s, a.y * s}; }
__device__ double2 cadd(double2 a, double2 b) { return {a.x + b.x, a.y + b.y}; }

__global__ void green_kspace_kernel(double2 *txx, double2 *tyy, double2 *tzz,
                                    double2 *tyz, double2 *txz, double2 *txy,
                                    const double *kx, const double *ky,
                                    const double *kz, const double *gxx,
                                    const double *gyy, const double *gzz,
                                    const double *gyz, const double *gxz,
                                    const double *gxy, std::size_t n,
                                    std::size_t zero_mode, double n_global,
                                    double e0, double e1, double e2, double e3,
                                    double e4, double e5) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i == zero_mode) {
    txx[i] = {n_global * e0, 0.0};
    tyy[i] = {n_global * e1, 0.0};
    tzz[i] = {n_global * e2, 0.0};
    tyz[i] = {n_global * e3, 0.0};
    txz[i] = {n_global * e4, 0.0};
    txy[i] = {n_global * e5, 0.0};
    return;
  }
  const double kxv = kx[i], kyv = ky[i], kzv = kz[i];
  const double2 txxv = txx[i], tyyv = tyy[i], tzzv = tzz[i];
  const double2 tyzv = tyz[i], txzv = txz[i], txyv = txy[i];
  const double2 bx = cadd(cadd(cscale(txxv, kxv), cscale(txyv, kyv)), cscale(txzv, kzv));
  const double2 by = cadd(cadd(cscale(txyv, kxv), cscale(tyyv, kyv)), cscale(tyzv, kzv));
  const double2 bz = cadd(cadd(cscale(txzv, kxv), cscale(tyzv, kyv)), cscale(tzzv, kzv));
  const double2 vx =
      cadd(cadd(cscale(bx, gxx[i]), cscale(by, gxy[i])), cscale(bz, gxz[i]));
  const double2 vy =
      cadd(cadd(cscale(bx, gxy[i]), cscale(by, gyy[i])), cscale(bz, gyz[i]));
  const double2 vz =
      cadd(cadd(cscale(bx, gxz[i]), cscale(by, gyz[i])), cscale(bz, gzz[i]));
  txx[i] = cscale(vx, -kxv);
  tyy[i] = cscale(vy, -kyv);
  tzz[i] = cscale(vz, -kzv);
  tyz[i] = cscale(cadd(cscale(vz, kyv), cscale(vy, kzv)), -0.5);
  txz[i] = cscale(cadd(cscale(vz, kxv), cscale(vx, kzv)), -0.5);
  txy[i] = cscale(cadd(cscale(vy, kxv), cscale(vx, kyv)), -0.5);
}

__device__ void contract(double c11, double c12, double c44, const double e[6],
                         double s[6]) {
  s[0] = c11 * e[0] + c12 * e[1] + c12 * e[2];
  s[1] = c12 * e[0] + c11 * e[1] + c12 * e[2];
  s[2] = c12 * e[0] + c12 * e[1] + c11 * e[2];
  s[3] = 2.0 * c44 * e[3];
  s[4] = 2.0 * c44 * e[4];
  s[5] = 2.0 * c44 * e[5];
}

__device__ void solve_cubic(double c11, double c12, double c44, const double s[6],
                            double e[6]) {
  const double d = c11 - c12;
  const double t = c11 + 2.0 * c12;
  const double tr = s[0] + s[1] + s[2];
  const double shift = c12 * tr / t;
  e[0] = (s[0] - shift) / d;
  e[1] = (s[1] - shift) / d;
  e[2] = (s[2] - shift) / d;
  e[3] = s[3] / (2.0 * c44);
  e[4] = s[4] / (2.0 * c44);
  e[5] = s[5] / (2.0 * c44);
}

__global__ void polarisation_kernel(const double *h, const double *amp,
                                    const double *eps0, const double *eps1,
                                    const double *eps2, const double *eps3,
                                    const double *eps4, const double *eps5,
                                    double *tau0, double *tau1, double *tau2,
                                    double *tau3, double *tau4, double *tau5,
                                    std::size_t n, double cs11, double cs12,
                                    double cs44, double cl11, double cl12,
                                    double cl44, double c011, double c012,
                                    double c044, double p0, double p1, double p2,
                                    double p3, double p4, double p5) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double hv = h[i];
  const double c11 = hv * cs11 + (1.0 - hv) * cl11;
  const double c12 = hv * cs12 + (1.0 - hv) * cl12;
  const double c44 = hv * cs44 + (1.0 - hv) * cl44;
  const double a = amp[i];
  double e[6] = {eps0[i], eps1[i], eps2[i], eps3[i], eps4[i], eps5[i]};
  double ed[6] = {e[0] - a * p0, e[1] - a * p1, e[2] - a * p2,
                  e[3] - a * p3, e[4] - a * p4, e[5] - a * p5};
  double s[6], s0[6];
  contract(c11, c12, c44, ed, s);
  contract(c011, c012, c044, e, s0);
  tau0[i] = s[0] - s0[0];
  tau1[i] = s[1] - s0[1];
  tau2[i] = s[2] - s0[2];
  tau3[i] = s[3] - s0[3];
  tau4[i] = s[4] - s0[4];
  tau5[i] = s[5] - s0[5];
}

__global__ void eyre_milton_local_kernel(
    const double *h, const double *amp, const double *zin0, const double *zin1,
    const double *zin2, const double *zin3, const double *zin4, const double *zin5,
    double *eps0, double *eps1, double *eps2, double *eps3, double *eps4,
    double *eps5, double *zout0, double *zout1, double *zout2, double *zout3,
    double *zout4, double *zout5, std::size_t n, double cs11, double cs12,
    double cs44, double cl11, double cl12, double cl44, double c011, double c012,
    double c044, double p0, double p1, double p2, double p3, double p4, double p5) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double hv = h[i];
  const double c11 = hv * cs11 + (1.0 - hv) * cl11;
  const double c12 = hv * cs12 + (1.0 - hv) * cl12;
  const double c44 = hv * cs44 + (1.0 - hv) * cl44;
  double z[6] = {zin0[i], zin1[i], zin2[i], zin3[i], zin4[i], zin5[i]};
  double w[6] = {eps0[i], eps1[i], eps2[i], eps3[i], eps4[i], eps5[i]};
  double c0w[6];
  contract(c011, c012, c044, w, c0w);
  double y[6];
  for (int c = 0; c < 6; ++c) y[c] = z[c] + 2.0 * c0w[c];
  const double a = amp[i];
  double estar[6] = {a * p0, a * p1, a * p2, a * p3, a * p4, a * p5};
  double cestar[6];
  contract(c11, c12, c44, estar, cestar);
  double rhs[6];
  for (int c = 0; c < 6; ++c) rhs[c] = y[c] + cestar[c];
  double e[6];
  solve_cubic(c11 + c011, c12 + c012, c44 + c044, rhs, e);
  double c0e[6];
  contract(c011, c012, c044, e, c0e);
  zout0[i] = y[0] - 2.0 * c0e[0];
  zout1[i] = y[1] - 2.0 * c0e[1];
  zout2[i] = y[2] - 2.0 * c0e[2];
  zout3[i] = y[3] - 2.0 * c0e[3];
  zout4[i] = y[4] - 2.0 * c0e[4];
  zout5[i] = y[5] - 2.0 * c0e[5];
  eps0[i] = e[0];
  eps1[i] = e[1];
  eps2[i] = e[2];
  eps3[i] = e[3];
  eps4[i] = e[4];
  eps5[i] = e[5];
}

int grid(std::size_t n) {
  return static_cast<int>((n + 255) / 256);
}

} // namespace

namespace pfc::apps {

void hip_green_kspace(void *txx, void *tyy, void *tzz, void *tyz, void *txz,
                      void *txy, const double *kx, const double *ky, const double *kz,
                      const double *gxx, const double *gyy, const double *gzz,
                      const double *gyz, const double *gxz, const double *gxy,
                      std::size_t n, std::size_t zero_mode, double n_global,
                      const double eapp[6]) {
  if (n == 0) return;
  green_kspace_kernel<<<grid(n), 256>>>(
      static_cast<double2 *>(txx), static_cast<double2 *>(tyy),
      static_cast<double2 *>(tzz), static_cast<double2 *>(tyz),
      static_cast<double2 *>(txz), static_cast<double2 *>(txy), kx, ky, kz, gxx,
      gyy, gzz, gyz, gxz, gxy, n, zero_mode, n_global, eapp[0], eapp[1], eapp[2],
      eapp[3], eapp[4], eapp[5]);
  hip_check(hipGetLastError(), "green_kspace launch");
  hip_check(hipDeviceSynchronize(), "green_kspace sync");
}

void hip_polarisation(const double *h, const double *amp, const double *eps[6],
                      double *tau[6], std::size_t n, const double cs[3],
                      const double cl[3], const double c0[3], const double pat[6]) {
  if (n == 0) return;
  polarisation_kernel<<<grid(n), 256>>>(
      h, amp, eps[0], eps[1], eps[2], eps[3], eps[4], eps[5], tau[0], tau[1],
      tau[2], tau[3], tau[4], tau[5], n, cs[0], cs[1], cs[2], cl[0], cl[1], cl[2],
      c0[0], c0[1], c0[2], pat[0], pat[1], pat[2], pat[3], pat[4], pat[5]);
  hip_check(hipGetLastError(), "polarisation launch");
  hip_check(hipDeviceSynchronize(), "polarisation sync");
}

void hip_eyre_milton_local(const double *h, const double *amp, const double *zin[6],
                           double *eps[6], double *zout[6], std::size_t n,
                           const double cs[3], const double cl[3], const double c0[3],
                           const double pat[6]) {
  if (n == 0) return;
  eyre_milton_local_kernel<<<grid(n), 256>>>(
      h, amp, zin[0], zin[1], zin[2], zin[3], zin[4], zin[5], eps[0], eps[1],
      eps[2], eps[3], eps[4], eps[5], zout[0], zout[1], zout[2], zout[3], zout[4],
      zout[5], n, cs[0], cs[1], cs[2], cl[0], cl[1], cl[2], c0[0], c0[1], c0[2],
      pat[0], pat[1], pat[2], pat[3], pat[4], pat[5]);
  hip_check(hipGetLastError(), "eyre_milton_local launch");
  hip_check(hipDeviceSynchronize(), "eyre_milton_local sync");
}

void hip_memcpy_d2d(void *dst, const void *src, std::size_t bytes) {
  hip_check(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice), "d2d");
}

} // namespace pfc::apps
