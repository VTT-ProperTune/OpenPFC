// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file microelasticity_hip_kernels.hip
 * @brief Device twin of the pointwise passes of `microelasticity.hpp`.
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "microelasticity_hip_kernels requires HIP (-DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <openpfc_apps/microelasticity_hip_kernels.hpp>

namespace pfc::apps::hip_detail {
namespace {

constexpr int kBlock = 256;

inline void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "microelasticity HIP error in %s: %s\n", what,
                 hipGetErrorString(e));
    std::abort();
  }
}

enum : int { XX = 0, YY = 1, ZZ = 2, YZ = 3, XZ = 4, XY = 5 };

__device__ inline void contract(const MEStiffness &c, const double e[6],
                                double s[6]) {
  s[XX] = c.c11 * e[XX] + c.c12 * (e[YY] + e[ZZ]);
  s[YY] = c.c11 * e[YY] + c.c12 * (e[XX] + e[ZZ]);
  s[ZZ] = c.c11 * e[ZZ] + c.c12 * (e[XX] + e[YY]);
  s[YZ] = 2.0 * c.c44 * e[YZ];
  s[XZ] = 2.0 * c.c44 * e[XZ];
  s[XY] = 2.0 * c.c44 * e[XY];
}

__device__ inline void solve(const MEStiffness &c, const double s[6],
                             double e[6]) {
  const double d = c.c11 - c.c12;
  const double t = c.c11 + 2.0 * c.c12;
  const double tr = s[XX] + s[YY] + s[ZZ];
  const double shift = c.c12 * tr / t;
  e[XX] = (s[XX] - shift) / d;
  e[YY] = (s[YY] - shift) / d;
  e[ZZ] = (s[ZZ] - shift) / d;
  e[YZ] = s[YZ] / (2.0 * c.c44);
  e[XZ] = s[XZ] / (2.0 * c.c44);
  e[XY] = s[XY] / (2.0 * c.c44);
}

__device__ inline MEStiffness stiffness_at(const MEParams &p, double h) {
  MEStiffness c;
  c.c11 = h * p.c_solid.c11 + (1.0 - h) * p.c_liquid.c11;
  c.c12 = h * p.c_solid.c12 + (1.0 - h) * p.c_liquid.c12;
  c.c44 = h * p.c_solid.c44 + (1.0 - h) * p.c_liquid.c44;
  return c;
}

__device__ inline double ddot6(const double a[6], const double b[6]) {
  return a[XX] * b[XX] + a[YY] * b[YY] + a[ZZ] * b[ZZ] +
         2.0 * (a[YZ] * b[YZ] + a[XZ] * b[XZ] + a[XY] * b[XY]);
}

__global__ void build_polarisation_kernel(const double *h, const double *amp,
                                          MESym6Const eps, MESym6 tau,
                                          MEParams p) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.n) return;
  double e[6];
  double ediff[6];
  for (int c = 0; c < 6; ++c) {
    e[c] = eps.c[c][i];
    ediff[c] = e[c] - amp[i] * p.pattern[c];
  }
  double s[6];
  contract(stiffness_at(p, h[i]), ediff, s);
  double s0[6];
  contract(p.c0, e, s0);
  for (int c = 0; c < 6; ++c) tau.c[c][i] = s[c] - s0[c];
}

__global__ void eyre_milton_local_kernel(const double *h, const double *amp,
                                         MESym6Const zin, MESym6 eps,
                                         MESym6 zout, MEParams p,
                                         double *block_max) {
  __shared__ double sdiff[kBlock];
  __shared__ double sscale[kBlock];
  const int tid = static_cast<int>(threadIdx.x);
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

  double diff = 0.0;
  double scale = 0.0;
  if (i < p.n) {
    double z[6];
    double w[6];
    for (int c = 0; c < 6; ++c) {
      z[c] = zin.c[c][i];
      w[c] = eps.c[c][i];
    }
    const MEStiffness ci_local = stiffness_at(p, h[i]);
    double c0w[6];
    contract(p.c0, w, c0w);
    double y[6];
    for (int c = 0; c < 6; ++c) y[c] = z[c] + 2.0 * c0w[c];

    double estar[6];
    for (int c = 0; c < 6; ++c) estar[c] = amp[i] * p.pattern[c];
    double c_estar[6];
    contract(ci_local, estar, c_estar);
    double rhs[6];
    for (int c = 0; c < 6; ++c) rhs[c] = y[c] + c_estar[c];

    MEStiffness sum;
    sum.c11 = ci_local.c11 + p.c0.c11;
    sum.c12 = ci_local.c12 + p.c0.c12;
    sum.c44 = ci_local.c44 + p.c0.c44;
    double e[6];
    solve(sum, rhs, e);
    double c0e[6];
    contract(p.c0, e, c0e);
    for (int c = 0; c < 6; ++c) {
      const double zn = y[c] - 2.0 * c0e[c];
      diff = fmax(diff, fabs(zn - z[c]));
      scale = fmax(scale, fabs(zn));
      zout.c[c][i] = zn;
      eps.c[c][i] = e[c];
    }
  }

  sdiff[tid] = diff;
  sscale[tid] = scale;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdiff[tid] = fmax(sdiff[tid], sdiff[tid + s]);
      sscale[tid] = fmax(sscale[tid], sscale[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_max[2 * blockIdx.x + 0] = sdiff[0];
    block_max[2 * blockIdx.x + 1] = sscale[0];
  }
}

struct MEComplex {
  double re;
  double im;
};

__device__ inline MEComplex cmul(double a, MEComplex z) {
  return MEComplex{a * z.re, a * z.im};
}
__device__ inline MEComplex cadd(MEComplex a, MEComplex b) {
  return MEComplex{a.re + b.re, a.im + b.im};
}

__global__ void green_multiply_kernel(MESym6 hat, const double *kx,
                                      const double *ky, const double *kz,
                                      MESym6Const g, long long n_outbox,
                                      long long zero_mode) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_outbox) return;
  if (i == zero_mode) return;

  MEComplex *t[6];
  for (int c = 0; c < 6; ++c) t[c] = reinterpret_cast<MEComplex *>(hat.c[c]);

  const double ax = kx[i];
  const double ay = ky[i];
  const double az = kz[i];
  const MEComplex txx = t[XX][i];
  const MEComplex tyy = t[YY][i];
  const MEComplex tzz = t[ZZ][i];
  const MEComplex tyz = t[YZ][i];
  const MEComplex txz = t[XZ][i];
  const MEComplex txy = t[XY][i];

  const MEComplex bx = cadd(cadd(cmul(ax, txx), cmul(ay, txy)), cmul(az, txz));
  const MEComplex by = cadd(cadd(cmul(ax, txy), cmul(ay, tyy)), cmul(az, tyz));
  const MEComplex bz = cadd(cadd(cmul(ax, txz), cmul(ay, tyz)), cmul(az, tzz));

  const MEComplex vx = cadd(cadd(cmul(g.c[XX][i], bx), cmul(g.c[XY][i], by)),
                            cmul(g.c[XZ][i], bz));
  const MEComplex vy = cadd(cadd(cmul(g.c[XY][i], bx), cmul(g.c[YY][i], by)),
                            cmul(g.c[YZ][i], bz));
  const MEComplex vz = cadd(cadd(cmul(g.c[XZ][i], bx), cmul(g.c[YZ][i], by)),
                            cmul(g.c[ZZ][i], bz));

  t[XX][i] = MEComplex{-(ax * vx.re), -(ax * vx.im)};
  t[YY][i] = MEComplex{-(ay * vy.re), -(ay * vy.im)};
  t[ZZ][i] = MEComplex{-(az * vz.re), -(az * vz.im)};
  t[YZ][i] = MEComplex{-0.5 * (ay * vz.re + az * vy.re),
                       -0.5 * (ay * vz.im + az * vy.im)};
  t[XZ][i] = MEComplex{-0.5 * (ax * vz.re + az * vx.re),
                       -0.5 * (ax * vz.im + az * vx.im)};
  t[XY][i] = MEComplex{-0.5 * (ax * vy.re + ay * vx.re),
                       -0.5 * (ax * vy.im + ay * vx.im)};
}

__global__ void green_zero_mode_kernel(MESym6 hat, long long zero_mode,
                                       double a0, double a1, double a2,
                                       double a3, double a4, double a5,
                                       double n_global) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const double applied[6] = {a0, a1, a2, a3, a4, a5};
  for (int c = 0; c < 6; ++c) {
    auto *t = reinterpret_cast<MEComplex *>(hat.c[c]);
    t[zero_mode] = MEComplex{n_global * applied[c], 0.0};
  }
}

__global__ void finalise_kernel(const double *h, const double *amp,
                                const double *dh_dphi, const double *damp_dphi,
                                MESym6Const eps, MESym6 sig, double *f_el,
                                double *dfel_dphi, MEParams p, int want_dfel) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.n) return;

  double ediff[6];
  for (int c = 0; c < 6; ++c) {
    ediff[c] = eps.c[c][i] - amp[i] * p.pattern[c];
  }
  double s[6];
  contract(stiffness_at(p, h[i]), ediff, s);
  for (int c = 0; c < 6; ++c) sig.c[c][i] = s[c];
  f_el[i] = 0.5 * ddot6(ediff, s);

  if (want_dfel != 0) {
    double work = 0.0;
    for (int c = 0; c < 6; ++c) {
      const double w = (c < 3) ? 1.0 : 2.0;
      work += w * s[c] * damp_dphi[i] * p.pattern[c];
    }
    double sc[6];
    contract(p.dc, ediff, sc);
    dfel_dphi[i] = -work + 0.5 * dh_dphi[i] * ddot6(ediff, sc);
  } else {
    dfel_dphi[i] = 0.0;
  }
}

__global__ void fill_kernel(double *dst, double value, long long n) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = value;
}

__global__ void block_sum_kernel(const double *x, double *partial, long long n) {
  __shared__ double s[kBlock];
  const int tid = static_cast<int>(threadIdx.x);
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  s[tid] = (i < n) ? x[i] : 0.0;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (tid < w) s[tid] += s[tid + w];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

__global__ void assemble_kernel(const double *phi, const double *U,
                                const double *theta, double *h, double *amp,
                                double *damp, int nx, int ny, int nz, int hw,
                                long long sy, long long sz, double eps_c,
                                double eps_T, double u_ref, double theta_ref,
                                double *block_sum) {
  __shared__ double s[kBlock];
  const int tid = static_cast<int>(threadIdx.x);
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long n = static_cast<long long>(nx) * ny * nz;
  double local = 0.0;
  if (i < n) {
    const int ix = static_cast<int>(i % nx);
    const long long t = i / nx;
    const int iy = static_cast<int>(t % ny);
    const int iz = static_cast<int>(t / ny);
    const long long pidx = (ix + hw) + (iy + hw) * sy + (iz + hw) * sz;
    const double ph = phi[pidx];
    const double hv = 0.5 * (1.0 + ph);
    const double sv =
        eps_c * (U[pidx] - u_ref) + eps_T * (theta[pidx] - theta_ref);
    h[i] = hv;
    amp[i] = hv * sv;
    damp[i] = 0.5 * sv;
    local = hv * sv;
  }
  s[tid] = local;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (tid < w) s[tid] += s[tid + w];
    __syncthreads();
  }
  if (tid == 0) block_sum[blockIdx.x] = s[0];
}

__global__ void report_stats_kernel(MESym6Const sig, const double *dfel,
                                      double *block, long long n) {
  __shared__ double sp[kBlock];
  __shared__ double sdfel[kBlock];
  __shared__ double svm[kBlock];
  const int tid = static_cast<int>(threadIdx.x);
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  double p = 0.0;
  double adfel = 0.0;
  double vm = 0.0;
  if (i < n) {
    const double sxx = sig.c[XX][i];
    const double syy = sig.c[YY][i];
    const double szz = sig.c[ZZ][i];
    const double syz = sig.c[YZ][i];
    const double sxz = sig.c[XZ][i];
    const double sxy = sig.c[XY][i];
    p = (sxx + syy + szz) / 3.0;
    adfel = fabs(dfel[i]);
    const double a = sxx - syy;
    const double b = syy - szz;
    const double d = szz - sxx;
    vm = sqrt(0.5 * (a * a + b * b + d * d) +
              3.0 * (syz * syz + sxz * sxz + sxy * sxy));
  }
  sp[tid] = p;
  sdfel[tid] = adfel;
  svm[tid] = vm;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (tid < w) {
      sp[tid] += sp[tid + w];
      sdfel[tid] = fmax(sdfel[tid], sdfel[tid + w]);
      svm[tid] = fmax(svm[tid], svm[tid + w]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block[3 * blockIdx.x + 0] = sp[0];
    block[3 * blockIdx.x + 1] = sdfel[0];
    block[3 * blockIdx.x + 2] = svm[0];
  }
}

__global__ void copy_owned_kernel(const double *src, double *dst, int nx, int ny,
                                 int nz, int hw, long long sy, long long sz) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long n = static_cast<long long>(nx) * ny * nz;
  if (i >= n) return;
  const int ix = static_cast<int>(i % nx);
  const long long t = i / nx;
  const int iy = static_cast<int>(t % ny);
  const int iz = static_cast<int>(t / ny);
  dst[(ix + hw) + (iy + hw) * sy + (iz + hw) * sz] = src[i];
}

[[nodiscard]] inline int blocks_for(long long n) {
  if (n <= 0) return 1;
  return static_cast<int>((n + kBlock - 1) / kBlock);
}

} // namespace

int me_local_block_count(long long n) noexcept { return blocks_for(n); }

void me_build_polarisation(const double *h, const double *amp,
                           const MESym6Const &eps, const MESym6 &tau,
                           const MEParams &p) {
  build_polarisation_kernel<<<blocks_for(p.n), kBlock>>>(h, amp, eps, tau, p);
  check(hipGetLastError(), "me_build_polarisation");
}

void me_eyre_milton_local(const double *h, const double *amp,
                          const MESym6Const &zin, const MESym6 &eps,
                          const MESym6 &zout, const MEParams &p,
                          double *block_max, int n_blocks) {
  eyre_milton_local_kernel<<<n_blocks, kBlock>>>(h, amp, zin, eps, zout, p,
                                                 block_max);
  check(hipGetLastError(), "me_eyre_milton_local");
}

void me_green_multiply(const MESym6 &hat, const double *kx, const double *ky,
                       const double *kz, const MESym6Const &g,
                       long long n_outbox, long long zero_mode,
                       const double applied[kSym], double n_global) {
  green_multiply_kernel<<<blocks_for(n_outbox), kBlock>>>(
      hat, kx, ky, kz, g, n_outbox, zero_mode);
  check(hipGetLastError(), "me_green_multiply");
  if (zero_mode >= 0) {
    green_zero_mode_kernel<<<1, 1>>>(hat, zero_mode, applied[0], applied[1],
                                     applied[2], applied[3], applied[4],
                                     applied[5], n_global);
    check(hipGetLastError(), "me_green_multiply zero mode");
  }
}

void me_finalise(const double *h, const double *amp, const double *dh_dphi,
                 const double *damp_dphi, const MESym6Const &eps,
                 const MESym6 &sig, double *f_el, double *dfel_dphi,
                 const MEParams &p, int want_dfel) {
  finalise_kernel<<<blocks_for(p.n), kBlock>>>(h, amp, dh_dphi, damp_dphi, eps,
                                               sig, f_el, dfel_dphi, p,
                                               want_dfel);
  check(hipGetLastError(), "me_finalise");
}

void me_fill(double *dst, double value, long long n) {
  fill_kernel<<<blocks_for(n), kBlock>>>(dst, value, n);
  check(hipGetLastError(), "me_fill");
}

void me_block_sum(const double *x, double *partial, long long n, int n_blocks) {
  block_sum_kernel<<<n_blocks, kBlock>>>(x, partial, n);
  check(hipGetLastError(), "me_block_sum");
}

void me_assemble_from_padded(const double *phi, const double *U,
                             const double *theta, double *h, double *amp,
                             double *damp, int nx, int ny, int nz, int hw,
                             long long sy, long long sz, double eps_c,
                             double eps_T, double u_ref, double theta_ref,
                             double *block_sum, int n_blocks) {
  assemble_kernel<<<n_blocks, kBlock>>>(phi, U, theta, h, amp, damp, nx, ny,
                                        nz, hw, sy, sz, eps_c, eps_T, u_ref,
                                        theta_ref, block_sum);
  check(hipGetLastError(), "me_assemble_from_padded");
}

void me_copy_owned_to_padded(const double *src, double *dst, int nx, int ny,
                             int nz, int hw, long long sy, long long sz) {
  const long long n = static_cast<long long>(nx) * ny * nz;
  copy_owned_kernel<<<blocks_for(n), kBlock>>>(src, dst, nx, ny, nz, hw, sy,
                                               sz);
  check(hipGetLastError(), "me_copy_owned_to_padded");
}

void me_report_stats(const MESym6Const &sig, const double *dfel, double *block,
                      long long n, int n_blocks) {
  report_stats_kernel<<<n_blocks, kBlock>>>(sig, dfel, block, n);
  check(hipGetLastError(), "me_report_stats");
}

} // namespace pfc::apps::hip_detail
