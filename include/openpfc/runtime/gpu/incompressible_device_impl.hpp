// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file incompressible_device_impl.hpp
 * @brief Device twin of `ifrk4_velocity` and `rotational_tendency`.
 *
 * Included only from the HIP and CUDA translation units, which share this
 * implementation. Requires the device compiler (__HIPCC__ or __CUDACC__).
 * One spectral backend per build: the kernels are not inside that split,
 * and hipcc may define __CUDACC__ as well as __HIPCC__, so
 * OpenPFC_ENABLE_HIP_SPECTRAL and OpenPFC_ENABLE_CUDA_SPECTRAL must not
 * both be set. Both backends would otherwise define the same device
 * symbols in one link. Instantiations use `#if defined(__HIPCC__)` then
 * `#elif defined(__CUDACC__)`. The formulas match the host step, including
 * the mask before each tendency and the mean-free rotational term.
 */

#pragma once

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#error "incompressible_device_impl.hpp is compiled by the device compiler"
#endif

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/incompressible_device.hpp>
#include <openpfc/solvers/incompressible/rotational.hpp>

namespace pfc::incompressible {
namespace {

struct Z {
  double re;
  double im;
};

static_assert(sizeof(Z) == sizeof(std::complex<double>),
              "device complex must match std::complex<double>");

[[nodiscard]] Z *as_z(std::complex<double> *p) { return reinterpret_cast<Z *>(p); }

[[nodiscard]] const Z *as_z(const std::complex<double> *p) {
  return reinterpret_cast<const Z *>(p);
}

__device__ Z zadd(Z a, Z b) { return Z{a.re + b.re, a.im + b.im}; }

__device__ Z zsub(Z a, Z b) { return Z{a.re - b.re, a.im - b.im}; }

__device__ Z zmul(double s, Z a) { return Z{s * a.re, s * a.im}; }

__device__ Z z_ik(double k, Z a) { return Z{-k * a.im, k * a.re}; }

__device__ void indices(unsigned long long idx, HatGeom g, int &i, int &j, int &k) {
  const unsigned long long n0 = static_cast<unsigned long long>(g.n0);
  const unsigned long long n1 = static_cast<unsigned long long>(g.n1);
  const unsigned long long plane = n0 * n1;
  const int lk = static_cast<int>(idx / plane);
  const unsigned long long rem = idx - static_cast<unsigned long long>(lk) * plane;
  const int lj = static_cast<int>(rem / n0);
  const int li = static_cast<int>(rem - static_cast<unsigned long long>(lj) * n0);
  i = g.low0 + li;
  j = g.low1 + lj;
  k = g.low2 + lk;
}

__device__ bool keep_mode(double kx, double ky, double kz, HatGeom g) {
  return fabs(kx) < g.cx && fabs(ky) < g.cy && fabs(kz) < g.cz;
}

__device__ void leray_one(Z &u, Z &v, Z &w, double kx, double ky, double kz) {
  const double k2 = kx * kx + ky * ky + kz * kz;
  if (k2 == 0.0) return;
  const Z dot = zadd(zadd(zmul(kx, u), zmul(ky, v)), zmul(kz, w));
  const double inv = 1.0 / k2;
  u = zsub(u, zmul(kx * inv, dot));
  v = zsub(v, zmul(ky * inv, dot));
  w = zsub(w, zmul(kz * inv, dot));
}

__global__ void leray_kernel(Z *u, Z *v, Z *w, HatGeom g, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int i = 0, j = 0, k = 0;
  indices(idx, g, i, j, k);
  const double kx = fft::kspace::k_component_odd(i, g.g0, g.fx);
  const double ky = fft::kspace::k_component_odd(j, g.g1, g.fy);
  const double kz = fft::kspace::k_component_odd(k, g.g2, g.fz);
  leray_one(u[idx], v[idx], w[idx], kx, ky, kz);
}

__global__ void mask_kernel(Z *u, Z *v, Z *w, HatGeom g, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int i = 0, j = 0, k = 0;
  indices(idx, g, i, j, k);
  const double kx = fft::kspace::k_component(i, g.g0, g.fx);
  const double ky = fft::kspace::k_component(j, g.g1, g.fy);
  const double kz = fft::kspace::k_component(k, g.g2, g.fz);
  if (keep_mode(kx, ky, kz, g)) return;
  u[idx] = v[idx] = w[idx] = Z{0.0, 0.0};
}

__global__ void zero_mean_kernel(Z *u, Z *v, Z *w, HatGeom g, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int i = 0, j = 0, k = 0;
  indices(idx, g, i, j, k);
  if (i == 0 && j == 0 && k == 0) u[idx] = v[idx] = w[idx] = Z{0.0, 0.0};
}

__global__ void curl_kernel(const Z *u, const Z *v, const Z *w, Z *ox, Z *oy, Z *oz,
                            HatGeom g, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int i = 0, j = 0, k = 0;
  indices(idx, g, i, j, k);
  const double kx = fft::kspace::k_component_odd(i, g.g0, g.fx);
  const double ky = fft::kspace::k_component_odd(j, g.g1, g.fy);
  const double kz = fft::kspace::k_component_odd(k, g.g2, g.fz);
  ox[idx] = zsub(z_ik(ky, w[idx]), z_ik(kz, v[idx]));
  oy[idx] = zsub(z_ik(kz, u[idx]), z_ik(kx, w[idx]));
  oz[idx] = zsub(z_ik(kx, v[idx]), z_ik(ky, u[idx]));
}

__global__ void cross_kernel(const double *ur, const double *vr, const double *wr,
                             const double *ox, const double *oy, const double *oz,
                             double *tx, double *ty, double *tz,
                             unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  tx[idx] = vr[idx] * oz[idx] - wr[idx] * oy[idx];
  ty[idx] = wr[idx] * ox[idx] - ur[idx] * oz[idx];
  tz[idx] = ur[idx] * oy[idx] - vr[idx] * ox[idx];
}

__global__ void copy3_kernel(const Z *a, const Z *b, const Z *c, Z *ao, Z *bo, Z *co,
                             unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  ao[idx] = a[idx];
  bo[idx] = b[idx];
  co[idx] = c[idx];
}

__global__ void stage1_kernel(const Z *u, const Z *v, const Z *w, const Z *n1u,
                              const Z *n1v, const Z *n1w, const double *ehalf, Z *su,
                              Z *sv, Z *sw, double dt, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double eh = ehalf[idx];
  const double half = 0.5 * dt;
  su[idx] = zmul(eh, zadd(u[idx], zmul(half, n1u[idx])));
  sv[idx] = zmul(eh, zadd(v[idx], zmul(half, n1v[idx])));
  sw[idx] = zmul(eh, zadd(w[idx], zmul(half, n1w[idx])));
}

__global__ void stage2_kernel(const Z *u, const Z *v, const Z *w, const Z *n2u,
                              const Z *n2v, const Z *n2w, const double *ehalf, Z *su,
                              Z *sv, Z *sw, double dt, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double half = 0.5 * dt;
  su[idx] = zadd(zmul(ehalf[idx], u[idx]), zmul(half, n2u[idx]));
  sv[idx] = zadd(zmul(ehalf[idx], v[idx]), zmul(half, n2v[idx]));
  sw[idx] = zadd(zmul(ehalf[idx], w[idx]), zmul(half, n2w[idx]));
}

__global__ void stage3_kernel(const Z *u, const Z *v, const Z *w, const Z *n3u,
                              const Z *n3v, const Z *n3w, const double *edt,
                              const double *ehalf, Z *su, Z *sv, Z *sw, double dt,
                              unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  su[idx] = zadd(zmul(edt[idx], u[idx]), zmul(dt * ehalf[idx], n3u[idx]));
  sv[idx] = zadd(zmul(edt[idx], v[idx]), zmul(dt * ehalf[idx], n3v[idx]));
  sw[idx] = zadd(zmul(edt[idx], w[idx]), zmul(dt * ehalf[idx], n3w[idx]));
}

__global__ void stage4_kernel(Z *u, Z *v, Z *w, const Z *n1u, const Z *n1v,
                              const Z *n1w, const Z *n2u, const Z *n2v, const Z *n2w,
                              const Z *n3u, const Z *n3v, const Z *n3w, const Z *n4u,
                              const Z *n4v, const Z *n4w, const double *edt,
                              const double *ehalf, double dt, unsigned long long n) {
  const unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double dt6 = dt / 6.0;
  const double ed = edt[idx];
  const double eh = ehalf[idx];
  u[idx] = zadd(zmul(ed, u[idx]),
                zmul(dt6, zadd(zadd(zmul(ed, n1u[idx]),
                                    zmul(2.0 * eh, zadd(n2u[idx], n3u[idx]))),
                               n4u[idx])));
  v[idx] = zadd(zmul(ed, v[idx]),
                zmul(dt6, zadd(zadd(zmul(ed, n1v[idx]),
                                    zmul(2.0 * eh, zadd(n2v[idx], n3v[idx]))),
                               n4v[idx])));
  w[idx] = zadd(zmul(ed, w[idx]),
                zmul(dt6, zadd(zadd(zmul(ed, n1w[idx]),
                                    zmul(2.0 * eh, zadd(n2w[idx], n3w[idx]))),
                               n4w[idx])));
}

template <class Kernel, class... Args>
void launch_1d(Kernel kernel, unsigned long long n, Args... args) {
  if (n == 0) return;
  constexpr unsigned threads = 256;
  const unsigned long long blocks64 = (n + threads - 1) / threads;
  if (blocks64 > static_cast<unsigned long long>(~unsigned{0})) {
    throw std::runtime_error("incompressible device: grid is too large to launch");
  }
  kernel<<<static_cast<unsigned>(blocks64), threads>>>(args..., n);
  GPU_CHECK(::pfc::gpuGetLastError());
}

template <class Buf>
void require_size(const Buf &buf, std::size_t n, const char *what) {
  if (buf.size() != n) {
    throw std::invalid_argument(std::string("incompressible device: ") + what);
  }
}

template <class Space>
void copy_velocity(DeviceVelocity<Space> &vel,
                   const typename DeviceVelocity<Space>::Cplx &a,
                   const typename DeviceVelocity<Space>::Cplx &b,
                   const typename DeviceVelocity<Space>::Cplx &c,
                   typename DeviceVelocity<Space>::Cplx &ao,
                   typename DeviceVelocity<Space>::Cplx &bo,
                   typename DeviceVelocity<Space>::Cplx &co) {
  launch_1d(copy3_kernel, vel.n_hat, as_z(a.data()), as_z(b.data()), as_z(c.data()),
            as_z(ao.data()), as_z(bo.data()), as_z(co.data()));
}

template <class Space>
void leray(DeviceVelocity<Space> &vel, typename DeviceVelocity<Space>::Cplx &u,
           typename DeviceVelocity<Space>::Cplx &v,
           typename DeviceVelocity<Space>::Cplx &w) {
  launch_1d(leray_kernel, vel.n_hat, as_z(u.data()), as_z(v.data()), as_z(w.data()),
            vel.geom);
}

template <class Space>
void mask_only(DeviceVelocity<Space> &vel, typename DeviceVelocity<Space>::Cplx &u,
               typename DeviceVelocity<Space>::Cplx &v,
               typename DeviceVelocity<Space>::Cplx &w) {
  launch_1d(mask_kernel, vel.n_hat, as_z(u.data()), as_z(v.data()), as_z(w.data()),
            vel.geom);
}

template <class F>
void timed(StepProfile *profile, double StepProfile::*slot, F &&body) {
  if (profile == nullptr) {
    body();
    return;
  }
  gpuEvent_t start{};
  gpuEvent_t stop{};
  GPU_CHECK(gpuEventCreate(&start));
  GPU_CHECK(gpuEventCreate(&stop));
  GPU_CHECK(gpuEventRecord(start, nullptr));
  body();
  GPU_CHECK(gpuEventRecord(stop, nullptr));
  GPU_CHECK(gpuEventSynchronize(stop));
  float ms = 0.0f;
  GPU_CHECK(gpuEventElapsedTime(&ms, start, stop));
  profile->*slot += static_cast<double>(ms) * 1.0e-3;
  GPU_CHECK(gpuEventDestroy(start));
  GPU_CHECK(gpuEventDestroy(stop));
}

template <class Space>
void project_mask(DeviceVelocity<Space> &vel,
                  typename DeviceVelocity<Space>::Cplx &u,
                  typename DeviceVelocity<Space>::Cplx &v,
                  typename DeviceVelocity<Space>::Cplx &w, bool dealias,
                  StepProfile *profile) {
  timed(profile, &StepProfile::spectral_s, [&] {
    leray(vel, u, v, w);
    if (dealias) mask_only(vel, u, v, w);
  });
}

template <class Space>
void rotational(DeviceVelocity<Space> &vel, fft::IDeviceFFT<Space> &fft,
                bool dealias, typename DeviceVelocity<Space>::Cplx &u,
                typename DeviceVelocity<Space>::Cplx &v,
                typename DeviceVelocity<Space>::Cplx &w,
                typename DeviceVelocity<Space>::Cplx &tu,
                typename DeviceVelocity<Space>::Cplx &tv,
                typename DeviceVelocity<Space>::Cplx &tw, StepProfile *profile) {
  timed(profile, &StepProfile::spectral_s, [&] {
    copy_velocity(vel, u, v, w, vel.uh, vel.vh, vel.wh);
    if (dealias) mask_only(vel, vel.uh, vel.vh, vel.wh);
    launch_1d(curl_kernel, vel.n_hat, as_z(vel.uh.data()), as_z(vel.vh.data()),
              as_z(vel.wh.data()), as_z(vel.ox.data()), as_z(vel.oy.data()),
              as_z(vel.oz.data()), vel.geom);
  });
  timed(profile, &StepProfile::fft_s, [&] {
    fft.backward(vel.uh, vel.ur);
    fft.backward(vel.vh, vel.vr);
    fft.backward(vel.wh, vel.wr);
    fft.backward(vel.ox, vel.oxr);
    fft.backward(vel.oy, vel.oyr);
    fft.backward(vel.oz, vel.ozr);
  });
  timed(profile, &StepProfile::nonlinear_s, [&] {
    launch_1d(cross_kernel, vel.n_real, vel.ur.data(), vel.vr.data(), vel.wr.data(),
              vel.oxr.data(), vel.oyr.data(), vel.ozr.data(), vel.tx.data(),
              vel.ty.data(), vel.tz.data());
  });
  timed(profile, &StepProfile::fft_s, [&] {
    fft.forward(vel.tx, tu);
    fft.forward(vel.ty, tv);
    fft.forward(vel.tz, tw);
  });
  timed(profile, &StepProfile::spectral_s, [&] {
    leray(vel, tu, tv, tw);
    if (dealias) mask_only(vel, tu, tv, tw);
    launch_1d(zero_mean_kernel, vel.n_hat, as_z(tu.data()), as_z(tv.data()),
              as_z(tw.data()), vel.geom);
  });
}

template <class Space> typename DeviceVelocity<Space>::Cplx make_hat(std::size_t n) {
  return typename DeviceVelocity<Space>::Cplx(n);
}

template <class Space>
typename DeviceVelocity<Space>::Real make_real(std::size_t n) {
  return typename DeviceVelocity<Space>::Real(n);
}

} // namespace

template <class Space>
void prepare_device_velocity(DeviceVelocity<Space> &vel,
                             const fft::IDeviceFFT<Space> &fft, std::array<int, 3> n,
                             std::array<double, 3> spacing, double nu, double dt) {
  const auto outbox = fft.get_outbox_bounds();
  const auto inbox = fft.get_inbox_bounds();
  vel.n_hat = fft.size_outbox();
  vel.n_real = fft.size_inbox();
  const auto hat_volume = static_cast<std::size_t>(outbox.size[0]) *
                          static_cast<std::size_t>(outbox.size[1]) *
                          static_cast<std::size_t>(outbox.size[2]);
  const auto real_volume = static_cast<std::size_t>(inbox.size[0]) *
                           static_cast<std::size_t>(inbox.size[1]) *
                           static_cast<std::size_t>(inbox.size[2]);
  if (hat_volume != vel.n_hat || real_volume != vel.n_real) {
    throw std::invalid_argument("incompressible device: box volume mismatch");
  }
  vel.dt = dt;
  vel.geom.low0 = outbox.low[0];
  vel.geom.low1 = outbox.low[1];
  vel.geom.low2 = outbox.low[2];
  vel.geom.n0 = outbox.size[0];
  vel.geom.n1 = outbox.size[1];
  vel.geom.n2 = outbox.size[2];
  vel.geom.g0 = n[0];
  vel.geom.g1 = n[1];
  vel.geom.g2 = n[2];
  vel.geom.fx = two_pi / (spacing[0] * static_cast<double>(n[0]));
  vel.geom.fy = two_pi / (spacing[1] * static_cast<double>(n[1]));
  vel.geom.fz = two_pi / (spacing[2] * static_cast<double>(n[2]));
  vel.geom.cx = (2.0 / 3.0) * (pi / spacing[0]);
  vel.geom.cy = (2.0 / 3.0) * (pi / spacing[1]);
  vel.geom.cz = (2.0 / 3.0) * (pi / spacing[2]);

  std::vector<double> exp_dt;
  std::vector<double> exp_half;
  viscous_exponentials(outbox, n, spacing, nu, dt, exp_dt, exp_half);
  vel.u = make_hat<Space>(vel.n_hat);
  vel.v = make_hat<Space>(vel.n_hat);
  vel.w = make_hat<Space>(vel.n_hat);
  vel.su = make_hat<Space>(vel.n_hat);
  vel.sv = make_hat<Space>(vel.n_hat);
  vel.sw = make_hat<Space>(vel.n_hat);
  vel.n1u = make_hat<Space>(vel.n_hat);
  vel.n1v = make_hat<Space>(vel.n_hat);
  vel.n1w = make_hat<Space>(vel.n_hat);
  vel.n2u = make_hat<Space>(vel.n_hat);
  vel.n2v = make_hat<Space>(vel.n_hat);
  vel.n2w = make_hat<Space>(vel.n_hat);
  vel.n3u = make_hat<Space>(vel.n_hat);
  vel.n3v = make_hat<Space>(vel.n_hat);
  vel.n3w = make_hat<Space>(vel.n_hat);
  vel.n4u = make_hat<Space>(vel.n_hat);
  vel.n4v = make_hat<Space>(vel.n_hat);
  vel.n4w = make_hat<Space>(vel.n_hat);
  vel.uh = make_hat<Space>(vel.n_hat);
  vel.vh = make_hat<Space>(vel.n_hat);
  vel.wh = make_hat<Space>(vel.n_hat);
  vel.ox = make_hat<Space>(vel.n_hat);
  vel.oy = make_hat<Space>(vel.n_hat);
  vel.oz = make_hat<Space>(vel.n_hat);
  vel.exp_dt = make_real<Space>(vel.n_hat);
  vel.exp_half = make_real<Space>(vel.n_hat);
  vel.ur = make_real<Space>(vel.n_real);
  vel.vr = make_real<Space>(vel.n_real);
  vel.wr = make_real<Space>(vel.n_real);
  vel.oxr = make_real<Space>(vel.n_real);
  vel.oyr = make_real<Space>(vel.n_real);
  vel.ozr = make_real<Space>(vel.n_real);
  vel.tx = make_real<Space>(vel.n_real);
  vel.ty = make_real<Space>(vel.n_real);
  vel.tz = make_real<Space>(vel.n_real);
  vel.exp_dt.copy_from_host(exp_dt);
  vel.exp_half.copy_from_host(exp_half);
}

template <class Space>
void upload_device_velocity(DeviceVelocity<Space> &vel,
                            const std::vector<std::complex<double>> &u,
                            const std::vector<std::complex<double>> &v,
                            const std::vector<std::complex<double>> &w) {
  require_size(vel.u, u.size(), "upload size");
  if (v.size() != u.size() || w.size() != u.size()) {
    throw std::invalid_argument("incompressible device: upload components differ");
  }
  vel.u.copy_from_host(u);
  vel.v.copy_from_host(v);
  vel.w.copy_from_host(w);
}

template <class Space>
void download_device_velocity(const DeviceVelocity<Space> &vel,
                              std::vector<std::complex<double>> &u,
                              std::vector<std::complex<double>> &v,
                              std::vector<std::complex<double>> &w) {
  u = vel.u.to_host();
  v = vel.v.to_host();
  w = vel.w.to_host();
}

template <class Space>
void step_device_velocity(DeviceVelocity<Space> &vel, fft::IDeviceFFT<Space> &fft,
                          bool dealias, StepProfile *profile) {
  project_mask(vel, vel.u, vel.v, vel.w, dealias, profile);
  rotational(vel, fft, dealias, vel.u, vel.v, vel.w, vel.n1u, vel.n1v, vel.n1w,
             profile);
  timed(profile, &StepProfile::spectral_s, [&] {
    launch_1d(stage1_kernel, vel.n_hat, as_z(vel.u.data()), as_z(vel.v.data()),
              as_z(vel.w.data()), as_z(vel.n1u.data()), as_z(vel.n1v.data()),
              as_z(vel.n1w.data()), vel.exp_half.data(), as_z(vel.su.data()),
              as_z(vel.sv.data()), as_z(vel.sw.data()), vel.dt);
  });
  project_mask(vel, vel.su, vel.sv, vel.sw, dealias, profile);
  rotational(vel, fft, dealias, vel.su, vel.sv, vel.sw, vel.n2u, vel.n2v, vel.n2w,
             profile);
  timed(profile, &StepProfile::spectral_s, [&] {
    launch_1d(stage2_kernel, vel.n_hat, as_z(vel.u.data()), as_z(vel.v.data()),
              as_z(vel.w.data()), as_z(vel.n2u.data()), as_z(vel.n2v.data()),
              as_z(vel.n2w.data()), vel.exp_half.data(), as_z(vel.su.data()),
              as_z(vel.sv.data()), as_z(vel.sw.data()), vel.dt);
  });
  project_mask(vel, vel.su, vel.sv, vel.sw, dealias, profile);
  rotational(vel, fft, dealias, vel.su, vel.sv, vel.sw, vel.n3u, vel.n3v, vel.n3w,
             profile);
  timed(profile, &StepProfile::spectral_s, [&] {
    launch_1d(stage3_kernel, vel.n_hat, as_z(vel.u.data()), as_z(vel.v.data()),
              as_z(vel.w.data()), as_z(vel.n3u.data()), as_z(vel.n3v.data()),
              as_z(vel.n3w.data()), vel.exp_dt.data(), vel.exp_half.data(),
              as_z(vel.su.data()), as_z(vel.sv.data()), as_z(vel.sw.data()), vel.dt);
  });
  project_mask(vel, vel.su, vel.sv, vel.sw, dealias, profile);
  rotational(vel, fft, dealias, vel.su, vel.sv, vel.sw, vel.n4u, vel.n4v, vel.n4w,
             profile);
  timed(profile, &StepProfile::spectral_s, [&] {
    launch_1d(stage4_kernel, vel.n_hat, as_z(vel.u.data()), as_z(vel.v.data()),
              as_z(vel.w.data()), as_z(vel.n1u.data()), as_z(vel.n1v.data()),
              as_z(vel.n1w.data()), as_z(vel.n2u.data()), as_z(vel.n2v.data()),
              as_z(vel.n2w.data()), as_z(vel.n3u.data()), as_z(vel.n3v.data()),
              as_z(vel.n3w.data()), as_z(vel.n4u.data()), as_z(vel.n4v.data()),
              as_z(vel.n4w.data()), vel.exp_dt.data(), vel.exp_half.data(), vel.dt);
  });
  project_mask(vel, vel.u, vel.v, vel.w, dealias, profile);
}

template <class Space>
std::size_t device_velocity_bytes(const DeviceVelocity<Space> &vel) {
  constexpr std::size_t cplx = sizeof(std::complex<double>);
  constexpr std::size_t real = sizeof(double);
  // 24 complex hats: state, stage field, four tendencies, curl inputs, vorticity.
  // 2 real hats: integrating-factor exponentials. 9 real fields: velocity,
  // vorticity, and the cross product.
  const std::size_t hats =
      vel.u.size() + vel.v.size() + vel.w.size() + vel.su.size() + vel.sv.size() +
      vel.sw.size() + vel.n1u.size() + vel.n1v.size() + vel.n1w.size() +
      vel.n2u.size() + vel.n2v.size() + vel.n2w.size() + vel.n3u.size() +
      vel.n3v.size() + vel.n3w.size() + vel.n4u.size() + vel.n4v.size() +
      vel.n4w.size() + vel.uh.size() + vel.vh.size() + vel.wh.size() +
      vel.ox.size() + vel.oy.size() + vel.oz.size();
  const std::size_t reals = vel.exp_dt.size() + vel.exp_half.size() + vel.ur.size() +
                            vel.vr.size() + vel.wr.size() + vel.oxr.size() +
                            vel.oyr.size() + vel.ozr.size() + vel.tx.size() +
                            vel.ty.size() + vel.tz.size();
  return hats * cplx + reals * real;
}

#if defined(__HIPCC__)
template void prepare_device_velocity<pfc::HIPSpace>(
    DeviceVelocity<pfc::HIPSpace> &, const fft::IDeviceFFT<pfc::HIPSpace> &,
    std::array<int, 3>, std::array<double, 3>, double, double);
template void
upload_device_velocity<pfc::HIPSpace>(DeviceVelocity<pfc::HIPSpace> &,
                                      const std::vector<std::complex<double>> &,
                                      const std::vector<std::complex<double>> &,
                                      const std::vector<std::complex<double>> &);
template void download_device_velocity<pfc::HIPSpace>(
    const DeviceVelocity<pfc::HIPSpace> &, std::vector<std::complex<double>> &,
    std::vector<std::complex<double>> &, std::vector<std::complex<double>> &);
template void step_device_velocity<pfc::HIPSpace>(DeviceVelocity<pfc::HIPSpace> &,
                                                  fft::IDeviceFFT<pfc::HIPSpace> &,
                                                  bool, StepProfile *);
template std::size_t
device_velocity_bytes<pfc::HIPSpace>(const DeviceVelocity<pfc::HIPSpace> &);
#elif defined(__CUDACC__)
template void prepare_device_velocity<pfc::CUDASpace>(
    DeviceVelocity<pfc::CUDASpace> &, const fft::IDeviceFFT<pfc::CUDASpace> &,
    std::array<int, 3>, std::array<double, 3>, double, double);
template void
upload_device_velocity<pfc::CUDASpace>(DeviceVelocity<pfc::CUDASpace> &,
                                       const std::vector<std::complex<double>> &,
                                       const std::vector<std::complex<double>> &,
                                       const std::vector<std::complex<double>> &);
template void download_device_velocity<pfc::CUDASpace>(
    const DeviceVelocity<pfc::CUDASpace> &, std::vector<std::complex<double>> &,
    std::vector<std::complex<double>> &, std::vector<std::complex<double>> &);
template void step_device_velocity<pfc::CUDASpace>(DeviceVelocity<pfc::CUDASpace> &,
                                                   fft::IDeviceFFT<pfc::CUDASpace> &,
                                                   bool, StepProfile *);
template std::size_t
device_velocity_bytes<pfc::CUDASpace>(const DeviceVelocity<pfc::CUDASpace> &);
#endif

} // namespace pfc::incompressible
