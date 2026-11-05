// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file elementwise_ops_gpu.inc
 * @brief Single-source generic GPU elementwise kernels (from vendor .cu/.hip).
 *
 * Do not include this from a host TU. Complex values are interleaved
 * real/imag (`std::complex` layout). Error strings keep the vendor kind.
 *
 * @see runtime/gpu/elementwise_ops_gpu.hpp
 */

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/field/grain_interface.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc {

#if defined(OPENPFC_GPU_API_CUDA)
constexpr const char *kElemGpuKind = "CUDA";
#elif defined(OPENPFC_GPU_API_HIP)
constexpr const char *kElemGpuKind = "HIP";
#endif

template <typename T>
__global__ void multiply_complex_real_kernel(const T *a, const T *b, T *out,
                                             std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const T br = b[i];
    out[2 * i] = a[2 * i] * br;
    out[2 * i + 1] = a[2 * i + 1] * br;
  }
}

template <typename T>
__global__ void combine_two_term_kernel(const T *x0, const T *x1, const T *w0,
                                        const T *w1, T *out, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const T a = w0[i];
    const T b = w1[i];
    out[2 * i] = a * x0[2 * i] + b * x1[2 * i];
    out[2 * i + 1] = a * x0[2 * i + 1] + b * x1[2 * i + 1];
  }
}

template <typename T>
__global__ void axpy_fill_kernel(T *out, const T *x, T alpha, T beta,
                                 std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * x[i] + beta;
  }
}

void check_elem_launch(const char *op) {
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kElemGpuKind) + " " + op +
                             " kernel failed: " + ::pfc::gpuGetErrorString(err));
  }
  // Do not gpuDeviceSynchronize here: SpectralETD queues multiply/combine
  // immediately before a HeFFTe transform on the default stream. A full-device
  // wait between them prevents overlapping those kernels with the FFT reshape.
}

int elem_blocks(std::size_t n) {
  constexpr int threads = 256;
  return static_cast<int>((n + static_cast<std::size_t>(threads) - 1) /
                          static_cast<std::size_t>(threads));
}

template <typename T>
void launch_multiply_complex_real(const T *a, const T *b, T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  multiply_complex_real_kernel<<<elem_blocks(n), threads>>>(a, b, out, n);
  check_elem_launch("multiply_complex_real");
}

template <typename T>
void launch_combine_two_term(const T *x0, const T *x1, const T *w0, const T *w1,
                             T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  combine_two_term_kernel<<<elem_blocks(n), threads>>>(x0, x1, w0, w1, out, n);
  check_elem_launch("combine_two_term");
}

template <typename T>
__global__ void combine_two_term_complex_kernel(const T *x0, const T *x1,
                                                const T *w0, const T *w1, T *out,
                                                std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const T ar = w0[2 * i];
    const T ai = w0[2 * i + 1];
    const T br = w1[2 * i];
    const T bi = w1[2 * i + 1];
    const T x0r = x0[2 * i];
    const T x0i = x0[2 * i + 1];
    const T x1r = x1[2 * i];
    const T x1i = x1[2 * i + 1];
    out[2 * i] = ar * x0r - ai * x0i + br * x1r - bi * x1i;
    out[2 * i + 1] = ar * x0i + ai * x0r + br * x1i + bi * x1r;
  }
}

template <typename T>
void launch_combine_two_term_complex(const T *x0, const T *x1, const T *w0,
                                     const T *w1, T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  combine_two_term_complex_kernel<<<elem_blocks(n), threads>>>(x0, x1, w0, w1, out,
                                                               n);
  check_elem_launch("combine_two_term_complex");
}

template <typename T>
__global__ void combine_two_term_real_kernel(const T *x0, const T *x1, const T *w0,
                                             const T *w1, T *out, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = w0[i] * x0[i] + w1[i] * x1[i];
  }
}

template <typename T>
void launch_combine_two_term_real(const T *x0, const T *x1, const T *w0, const T *w1,
                                  T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  combine_two_term_real_kernel<<<elem_blocks(n), threads>>>(x0, x1, w0, w1, out, n);
  check_elem_launch("combine_two_term_real");
}

template <typename T>
void launch_axpy_fill(T *out, const T *x, T alpha, T beta, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  axpy_fill_kernel<<<elem_blocks(n), threads>>>(out, x, alpha, beta, n);
  check_elem_launch("axpy_fill");
}

__global__ void amplitude_pair_etd1_kernel(const double *re, const double *im,
                                           const double *nre, const double *nim,
                                           const double *e1, const double *p1,
                                           const double *e2, const double *p2,
                                           double *ore, double *oim, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double hr = re[2 * i];
  const double hi = re[2 * i + 1];
  const double kr = im[2 * i];
  const double ki = im[2 * i + 1];
  const double nr = nre[2 * i];
  const double ni = nre[2 * i + 1];
  const double mr = nim[2 * i];
  const double mi = nim[2 * i + 1];
  const double c1r = hr - ki;
  const double c1i = hi + kr;
  const double c2r = hr + ki;
  const double c2i = hi - kr;
  const double n1r = nr - mi;
  const double n1i = ni + mr;
  const double n2r = nr + mi;
  const double n2i = ni - mr;
  const double d1r = e1[i] * c1r + p1[i] * n1r;
  const double d1i = e1[i] * c1i + p1[i] * n1i;
  const double d2r = e2[i] * c2r + p2[i] * n2r;
  const double d2i = e2[i] * c2i + p2[i] * n2i;
  ore[2 * i] = 0.5 * (d1r + d2r);
  ore[2 * i + 1] = 0.5 * (d1i + d2i);
  const double dr = d1r - d2r;
  const double di = d1i - d2i;
  oim[2 * i] = 0.5 * di;
  oim[2 * i + 1] = -0.5 * dr;
}

void launch_amplitude_pair_etd1(const double *re, const double *im,
                                const double *nre, const double *nim,
                                const double *e1, const double *p1, const double *e2,
                                const double *p2, double *ore, double *oim,
                                std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  amplitude_pair_etd1_kernel<<<elem_blocks(n), threads>>>(re, im, nre, nim, e1, p1,
                                                          e2, p2, ore, oim, n);
  check_elem_launch("amplitude_pair_etd1");
}

__device__ int level_set_wrap(int i, int n) {
  const int m = i % n;
  return m < 0 ? m + n : m;
}

__device__ double level_set_at(const double *phi, int i, int j, int k, int nx,
                               int ny, int nz) {
  const int ix = level_set_wrap(i, nx);
  const int jx = level_set_wrap(j, ny);
  const int kx = level_set_wrap(k, nz);
  return phi[static_cast<std::size_t>(ix + nx * (jx + ny * kx))];
}

__device__ double level_set_godunov(double speed, double backward, double forward) {
  const double a = speed >= 0.0 ? (backward > 0.0 ? backward : 0.0)
                                : (backward < 0.0 ? backward : 0.0);
  const double b = speed >= 0.0 ? (forward < 0.0 ? forward : 0.0)
                                : (forward > 0.0 ? forward : 0.0);
  const double aa = a * a;
  const double bb = b * b;
  return aa > bb ? aa : bb;
}

__global__ void level_set_rate_kernel(const double *phi, const double *vx,
                                      const double *vy, const double *vz,
                                      const double *vn, const double *phi0,
                                      double *rate, int nx, int ny, int nz,
                                      double hx, double hy, double hz, int mode) {
  const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz);
  const std::size_t p =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= n) {
    return;
  }
  const int i = static_cast<int>(p % static_cast<std::size_t>(nx));
  const std::size_t tile = p / static_cast<std::size_t>(nx);
  const int j = static_cast<int>(tile % static_cast<std::size_t>(ny));
  const int k = static_cast<int>(tile / static_cast<std::size_t>(ny));
  const double c = level_set_at(phi, i, j, k, nx, ny, nz);
  const double dmx = (c - level_set_at(phi, i - 1, j, k, nx, ny, nz)) / hx;
  const double dpx = (level_set_at(phi, i + 1, j, k, nx, ny, nz) - c) / hx;
  const double dmy =
      ny == 1 ? 0.0 : (c - level_set_at(phi, i, j - 1, k, nx, ny, nz)) / hy;
  const double dpy =
      ny == 1 ? 0.0 : (level_set_at(phi, i, j + 1, k, nx, ny, nz) - c) / hy;
  const double dmz =
      nz == 1 ? 0.0 : (c - level_set_at(phi, i, j, k - 1, nx, ny, nz)) / hz;
  const double dpz =
      nz == 1 ? 0.0 : (level_set_at(phi, i, j, k + 1, nx, ny, nz) - c) / hz;
  double out = 0.0;
  if (mode == 0) {
    const double ux = vx[p] > 0.0 ? vx[p] * dmx : (vx[p] < 0.0 ? vx[p] * dpx : 0.0);
    const double uy = vy[p] > 0.0 ? vy[p] * dmy : (vy[p] < 0.0 ? vy[p] * dpy : 0.0);
    const double uz = vz[p] > 0.0 ? vz[p] * dmz : (vz[p] < 0.0 ? vz[p] * dpz : 0.0);
    out = -(ux + uy + uz);
  } else {
    double speed = 0.0;
    if (mode == 1) {
      speed = vn[p];
    } else {
      const double h = hx < hy ? (hx < hz ? hx : hz) : (hy < hz ? hy : hz);
      const double base = phi0[p];
      speed = base / sqrt(base * base + h * h);
    }
    const double mag = sqrt(level_set_godunov(speed, dmx, dpx) +
                            level_set_godunov(speed, dmy, dpy) +
                            level_set_godunov(speed, dmz, dpz));
    out = mode == 2 ? -speed * mag + speed : -speed * mag;
  }
  rate[p] = out;
}

__global__ void level_set_combine3_kernel(const double *x, const double *y,
                                          const double *z, double *out, double alpha,
                                          double beta, double gamma, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
}

void launch_level_set_rate(const double *phi, const double *vx, const double *vy,
                           const double *vz, const double *vn, const double *phi0,
                           double *rate, int nx, int ny, int nz, double hx,
                           double hy, double hz, int mode) {
  const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz);
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  level_set_rate_kernel<<<elem_blocks(n), threads>>>(phi, vx, vy, vz, vn, phi0, rate,
                                                     nx, ny, nz, hx, hy, hz, mode);
  check_elem_launch("level_set_rate");
}

void launch_level_set_combine3(const double *x, const double *y, const double *z,
                               double *out, double alpha, double beta, double gamma,
                               std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  level_set_combine3_kernel<<<elem_blocks(n), threads>>>(x, y, z, out, alpha, beta,
                                                         gamma, n);
  check_elem_launch("level_set_combine3");
}

__device__ int grain_wrap_coord(int i, int n) {
  if (i < 0) return i + n;
  if (i >= n) return i - n;
  return i;
}

__global__ void grain_count_kernel(const int *ids, int *counts, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(counts + ids[i], 1);
}

__global__ void grain_band_kernel(const int *ids,
                                  const field::grain_interface::BandSite *band,
                                  int n_band, int nx, int ny, int nz,
                                  int *mismatches) {
  const int b =
      static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (b >= n_band) return;
  const int p = band[b].index;
  const int nxny = nx * ny;
  const int k = p / nxny;
  const int rem = p - k * nxny;
  const int j = rem / nx;
  const int i = rem - j * nx;
  const int owner = ids[p];
  int seen[27];
  int nseen = 0;
  seen[nseen++] = owner;
  for (int dk = -1; dk <= 1; ++dk) {
    for (int dj = -1; dj <= 1; ++dj) {
      for (int di = -1; di <= 1; ++di) {
        if (di == 0 && dj == 0 && dk == 0) continue;
        const int ii = grain_wrap_coord(i + di, nx);
        const int jj = grain_wrap_coord(j + dj, ny);
        const int kk = grain_wrap_coord(k + dk, nz);
        const int q = ii + nx * (jj + ny * kk);
        if (q == p) continue;
        const int id = ids[q];
        int known = 0;
        for (int t = 0; t < nseen; ++t) {
          if (seen[t] == id) known = 1;
        }
        if (known == 0) seen[nseen++] = id;
      }
    }
  }
  if (owner != band[b].owner || nseen != band[b].n_ids) atomicAdd(mismatches, 1);
}

void launch_grain_count(const int *ids, int *counts, std::size_t n) {
  if (n == 0) return;
  constexpr int threads = 256;
  grain_count_kernel<<<elem_blocks(n), threads>>>(ids, counts, n);
  check_elem_launch("grain_count");
}

void launch_grain_band(const int *ids, const field::grain_interface::BandSite *band,
                       int n_band, int nx, int ny, int nz, int *mismatches) {
  if (n_band == 0) return;
  constexpr int threads = 256;
  const std::size_t n = static_cast<std::size_t>(n_band);
  grain_band_kernel<<<elem_blocks(n), threads>>>(ids, band, n_band, nx, ny, nz,
                                                 mismatches);
  check_elem_launch("grain_band");
}

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPENPFC_GPU_API_CUDA)
void multiply_complex_real_cuda_impl(const std::complex<double> *a, const double *b,
                                     std::complex<double> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const double *>(a), b,
                               reinterpret_cast<double *>(out), n);
}
void multiply_complex_real_cuda_impl(const std::complex<float> *a, const float *b,
                                     std::complex<float> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const float *>(a), b,
                               reinterpret_cast<float *>(out), n);
}
void combine_two_term_cuda_impl(const std::complex<double> *x0,
                                const std::complex<double> *x1, const double *w0,
                                const double *w1, std::complex<double> *out,
                                std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const double *>(x0),
                          reinterpret_cast<const double *>(x1), w0, w1,
                          reinterpret_cast<double *>(out), n);
}
void combine_two_term_cuda_impl(const std::complex<float> *x0,
                                const std::complex<float> *x1, const float *w0,
                                const float *w1, std::complex<float> *out,
                                std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const float *>(x0),
                          reinterpret_cast<const float *>(x1), w0, w1,
                          reinterpret_cast<float *>(out), n);
}
void combine_two_term_cuda_impl(const std::complex<double> *x0,
                                const std::complex<double> *x1,
                                const std::complex<double> *w0,
                                const std::complex<double> *w1,
                                std::complex<double> *out, std::size_t n) {
  launch_combine_two_term_complex(
      reinterpret_cast<const double *>(x0), reinterpret_cast<const double *>(x1),
      reinterpret_cast<const double *>(w0), reinterpret_cast<const double *>(w1),
      reinterpret_cast<double *>(out), n);
}
void combine_two_term_cuda_impl(const std::complex<float> *x0,
                                const std::complex<float> *x1,
                                const std::complex<float> *w0,
                                const std::complex<float> *w1,
                                std::complex<float> *out, std::size_t n) {
  launch_combine_two_term_complex(
      reinterpret_cast<const float *>(x0), reinterpret_cast<const float *>(x1),
      reinterpret_cast<const float *>(w0), reinterpret_cast<const float *>(w1),
      reinterpret_cast<float *>(out), n);
}
void combine_two_term_cuda_impl(const double *x0, const double *x1, const double *w0,
                                const double *w1, double *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void combine_two_term_cuda_impl(const float *x0, const float *x1, const float *w0,
                                const float *w1, float *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void axpy_fill_cuda_impl(double *out, const double *x, double alpha, double beta,
                         std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void axpy_fill_cuda_impl(float *out, const float *x, float alpha, float beta,
                         std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void amplitude_pair_etd1_cuda_impl(const std::complex<double> *re_hat,
                                   const std::complex<double> *im_hat,
                                   const std::complex<double> *nre_hat,
                                   const std::complex<double> *nim_hat,
                                   const double *exp_plus, const double *phi_plus,
                                   const double *exp_minus, const double *phi_minus,
                                   std::complex<double> *out_re,
                                   std::complex<double> *out_im, std::size_t n) {
  launch_amplitude_pair_etd1(reinterpret_cast<const double *>(re_hat),
                             reinterpret_cast<const double *>(im_hat),
                             reinterpret_cast<const double *>(nre_hat),
                             reinterpret_cast<const double *>(nim_hat), exp_plus,
                             phi_plus, exp_minus, phi_minus,
                             reinterpret_cast<double *>(out_re),
                             reinterpret_cast<double *>(out_im), n);
}
void level_set_rate_cuda_impl(const double *phi, const double *vx, const double *vy,
                              const double *vz, const double *vn, const double *phi0,
                              double *rate, int nx, int ny, int nz, double hx,
                              double hy, double hz, int mode) {
  launch_level_set_rate(phi, vx, vy, vz, vn, phi0, rate, nx, ny, nz, hx, hy, hz,
                        mode);
}
void level_set_combine3_cuda_impl(const double *x, const double *y, const double *z,
                                  double *out, double alpha, double beta,
                                  double gamma, std::size_t n) {
  launch_level_set_combine3(x, y, z, out, alpha, beta, gamma, n);
}
void grain_count_cuda_impl(const int *ids, int *counts, std::size_t n) {
  launch_grain_count(ids, counts, n);
}
void grain_band_cuda_impl(const int *ids,
                          const field::grain_interface::BandSite *band, int n_band,
                          int nx, int ny, int nz, int *mismatches) {
  launch_grain_band(ids, band, n_band, nx, ny, nz, mismatches);
}
#endif

#if defined(OpenPFC_ENABLE_HIP) && defined(OPENPFC_GPU_API_HIP)
void multiply_complex_real_hip_impl(const std::complex<double> *a, const double *b,
                                    std::complex<double> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const double *>(a), b,
                               reinterpret_cast<double *>(out), n);
}
void multiply_complex_real_hip_impl(const std::complex<float> *a, const float *b,
                                    std::complex<float> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const float *>(a), b,
                               reinterpret_cast<float *>(out), n);
}
void combine_two_term_hip_impl(const std::complex<double> *x0,
                               const std::complex<double> *x1, const double *w0,
                               const double *w1, std::complex<double> *out,
                               std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const double *>(x0),
                          reinterpret_cast<const double *>(x1), w0, w1,
                          reinterpret_cast<double *>(out), n);
}
void combine_two_term_hip_impl(const std::complex<float> *x0,
                               const std::complex<float> *x1, const float *w0,
                               const float *w1, std::complex<float> *out,
                               std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const float *>(x0),
                          reinterpret_cast<const float *>(x1), w0, w1,
                          reinterpret_cast<float *>(out), n);
}
void combine_two_term_hip_impl(const std::complex<double> *x0,
                               const std::complex<double> *x1,
                               const std::complex<double> *w0,
                               const std::complex<double> *w1,
                               std::complex<double> *out, std::size_t n) {
  launch_combine_two_term_complex(
      reinterpret_cast<const double *>(x0), reinterpret_cast<const double *>(x1),
      reinterpret_cast<const double *>(w0), reinterpret_cast<const double *>(w1),
      reinterpret_cast<double *>(out), n);
}
void combine_two_term_hip_impl(const std::complex<float> *x0,
                               const std::complex<float> *x1,
                               const std::complex<float> *w0,
                               const std::complex<float> *w1,
                               std::complex<float> *out, std::size_t n) {
  launch_combine_two_term_complex(
      reinterpret_cast<const float *>(x0), reinterpret_cast<const float *>(x1),
      reinterpret_cast<const float *>(w0), reinterpret_cast<const float *>(w1),
      reinterpret_cast<float *>(out), n);
}
void combine_two_term_hip_impl(const double *x0, const double *x1, const double *w0,
                               const double *w1, double *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void combine_two_term_hip_impl(const float *x0, const float *x1, const float *w0,
                               const float *w1, float *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void axpy_fill_hip_impl(double *out, const double *x, double alpha, double beta,
                        std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void axpy_fill_hip_impl(float *out, const float *x, float alpha, float beta,
                        std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void amplitude_pair_etd1_hip_impl(const std::complex<double> *re_hat,
                                  const std::complex<double> *im_hat,
                                  const std::complex<double> *nre_hat,
                                  const std::complex<double> *nim_hat,
                                  const double *exp_plus, const double *phi_plus,
                                  const double *exp_minus, const double *phi_minus,
                                  std::complex<double> *out_re,
                                  std::complex<double> *out_im, std::size_t n) {
  launch_amplitude_pair_etd1(reinterpret_cast<const double *>(re_hat),
                             reinterpret_cast<const double *>(im_hat),
                             reinterpret_cast<const double *>(nre_hat),
                             reinterpret_cast<const double *>(nim_hat), exp_plus,
                             phi_plus, exp_minus, phi_minus,
                             reinterpret_cast<double *>(out_re),
                             reinterpret_cast<double *>(out_im), n);
}
void level_set_rate_hip_impl(const double *phi, const double *vx, const double *vy,
                             const double *vz, const double *vn, const double *phi0,
                             double *rate, int nx, int ny, int nz, double hx,
                             double hy, double hz, int mode) {
  launch_level_set_rate(phi, vx, vy, vz, vn, phi0, rate, nx, ny, nz, hx, hy, hz,
                        mode);
}
void level_set_combine3_hip_impl(const double *x, const double *y, const double *z,
                                 double *out, double alpha, double beta,
                                 double gamma, std::size_t n) {
  launch_level_set_combine3(x, y, z, out, alpha, beta, gamma, n);
}
void grain_count_hip_impl(const int *ids, int *counts, std::size_t n) {
  launch_grain_count(ids, counts, n);
}
void grain_band_hip_impl(const int *ids,
                         const field::grain_interface::BandSite *band, int n_band,
                         int nx, int ny, int nz, int *mismatches) {
  launch_grain_band(ids, band, n_band, nx, ny, nz, mismatches);
}
#endif

} // namespace pfc
