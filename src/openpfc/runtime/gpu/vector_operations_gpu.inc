// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <openpfc/runtime/gpu/vector_operations.hpp>
namespace pfc::solvers::gpu::detail {
__device__ double ieee_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ double ieee_add(double a, double b) { return __dadd_rn(a, b); }
__device__ double ieee_div(double a, double b) { return __ddiv_rn(a, b); }
__global__ void combine_kernel(double *y, double a, const double *x, double b,
                               std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (a == 0 ? 0 : a * x[i]) + (b == 0 ? 0 : b * y[i]);
}
__global__ void dot_kernel(const double *x, const double *y, std::size_t n,
                           double *out) {
  __shared__ double sums[256];
  double sum = 0;
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(gridDim.x) * blockDim.x)
    sum += x[i] * y[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sums[0];
}
__global__ void sum_kernel(double *out, int n) {
  __shared__ double sums[256];
  double sum = 0;
  for (int i = threadIdx.x; i < n; i += 256) sum += out[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[0] = sums[0];
}
__global__ void component_mean(const double *x, std::size_t n, int components,
                               double *out) {
  __shared__ double sums[256];
  double sum = 0;
  for (auto i = std::size_t(threadIdx.x) * components + blockIdx.x; i < n;
       i += 256 * components)
    sum += x[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sums[0] / (n / components);
}
__global__ void mean_correction(double *y, std::size_t n, int components,
                                const double *mean, bool subtract) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] += (subtract ? -1 : 1) * mean[i % components];
}
__global__ void store_if(double *dst, const double *src, const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  *dst = *src;
}
__global__ void set_if(double *dst, double value, const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  *dst = value;
}
__global__ void combine_slot_kernel(double *y, const double *a, double sign,
                                    const double *x, double b, std::size_t n,
                                    const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  double av = a ? ieee_mul(*a, sign) : 0;
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (av == 0 ? 0 : av * x[i]) + (b == 0 ? 0 : b * y[i]);
}
__global__ void scale_inv_kernel(double *y, const double *a, std::size_t n,
                                 const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  double s = ieee_div(1, *a);
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (s == 0 ? 0 : s * y[i]);
}
__global__ void scale_mul_kernel(double *y, const double *a, std::size_t n,
                                 const ScalarPoll *poll) {
  if (poll && (poll->flag || poll->pending)) return;
  double s = *a;
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (s == 0 ? 0 : s * y[i]);
}
__global__ void scale_ratio_kernel(double *y, const double *num, const double *den,
                                   std::size_t n, const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  double s = ieee_div(*num, *den);
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (s == 0 ? 0 : s * y[i]);
}
__global__ void fill_kernel(double *p, std::size_t n, double value) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = value;
}
__global__ void set_double_kernel(double *p, double value) { *p = value; }
__global__ void copy_double_kernel(double *dst, const double *src,
                                   const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  *dst = *src;
}
__global__ void sqrt_slot_kernel(double *p, const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  *p = ::sqrt(*p);
}
__global__ void publish_sqrt_kernel(const double *slot, ScalarPoll *poll) {
  if (poll->flag) return;
  double s = ::sqrt(*slot);
  poll->value = s;
  if (!::isfinite(s)) poll->flag = 3;
}
__global__ void latch_stop_kernel(ScalarPoll *poll, const double *stop) {
  if (poll->flag) return;
  if (poll->value <= *stop) poll->flag = 4;
}
__global__ void cg_alpha_kernel(const double *rho, const double *curv, double *alpha,
                                ScalarPoll *poll) {
  if (poll->flag) return;
  double r = *rho, c = *curv;
  if (!::isfinite(r) || !::isfinite(c)) {
    poll->flag = 2;
    return;
  }
  if (!(r > 0) || !(c > 0)) {
    poll->flag = 1;
    return;
  }
  *alpha = ieee_div(r, c);
}
__global__ void inc_iter_kernel(ScalarPoll *poll) {
  if (poll->flag) return;
  poll->iterations += 1;
}
__global__ void clear_poll_kernel(ScalarPoll *poll) {
  poll->value = 0;
  poll->flag = 0;
  poll->iterations = 0;
  poll->columns = 0;
  poll->pending = 0;
}
__global__ void clear_flag_kernel(ScalarPoll *poll) {
  poll->flag = 0;
  poll->pending = 0;
}
__global__ void clear_restart_kernel(ScalarPoll *poll) {
  poll->flag = 0;
  poll->pending = 0;
  poll->columns = 0;
}
__global__ void add_h_kernel(double *H, const double *coeff, int j, int ld,
                             const ScalarPoll *poll) {
  if (poll && poll->flag) return;
  for (int i = 0; i <= j; ++i) H[i * ld + j] = ieee_add(H[i * ld + j], coeff[i]);
}
__global__ void prepare_arnoldi_kernel(double *H, const double *dot,
                                       const double *before, int j, int ld,
                                       double eps64, double *scale,
                                       ScalarPoll *poll) {
  if (poll->flag) return;
  double hjp = ::sqrt(*dot);
  double b = *before;
  H[(j + 1) * ld + j] = hjp;
  poll->pending = 0;
  if (!::isfinite(b) || !::isfinite(hjp)) {
    poll->flag = 2;
    return;
  }
  if (hjp <= eps64 * b) {
    poll->pending = 1;
    return;
  }
  *scale = ieee_div(1, hjp);
}
__global__ void givens_kernel(double *H, double *cs, double *sn, double *g, int j,
                              int ld, const double *stop, ScalarPoll *poll) {
  if (poll->flag) return;
  for (int i = 0; i < j; ++i) {
    double a = H[i * ld + j];
    double d = H[(i + 1) * ld + j];
    H[i * ld + j] = ieee_add(ieee_mul(cs[i], a), ieee_mul(sn[i], d));
    H[(i + 1) * ld + j] = ieee_add(ieee_mul(-sn[i], a), ieee_mul(cs[i], d));
  }
  double hj = H[j * ld + j];
  double hjp = H[(j + 1) * ld + j];
  double length = ::hypot(hj, hjp);
  if (!(length > 0) || !::isfinite(length)) {
    poll->flag = ::isfinite(length) ? 3 : 2;
    return;
  }
  double c = ieee_div(hj, length);
  double s = ieee_div(hjp, length);
  cs[j] = c;
  sn[j] = s;
  H[j * ld + j] = length;
  H[(j + 1) * ld + j] = 0;
  g[j + 1] = ieee_mul(-s, g[j]);
  g[j] = ieee_mul(g[j], c);
  double next = g[j + 1];
  poll->value = next < 0 ? -next : next;
  poll->iterations += 1;
  poll->columns = j + 1;
  if (poll->pending)
    poll->flag = 1;
  else if (poll->value <= *stop)
    poll->flag = 4;
  poll->pending = 0;
}
__global__ void backsolve_kernel(const double *H, const double *g, double *y,
                                 int columns, int ld) {
  for (int i = columns - 1; i >= 0; --i) {
    double a = g[i];
    for (int j = i + 1; j < columns; ++j)
      a = ieee_add(a, ieee_mul(-H[i * ld + j], y[j]));
    y[i] = ieee_div(a, H[i * ld + i]);
  }
}
#if defined(OPENPFC_GPU_API_CUDA)
#define OP_NAME(name) name##_cuda
#else
#define OP_NAME(name) name##_hip
#endif
void OP_NAME(combine)(double *y, double a, const double *x, double b,
                      std::size_t n) {
  if (!n) return;
  combine_kernel<<<(n + 255) / 256, 256>>>(y, a, x, b, n);
  check(pfc::gpuGetLastError());
}
double OP_NAME(dot)(const double *x, const double *y, std::size_t n, double *out) {
  if (!n) return 0;
  int blocks = std::min<std::size_t>(1024, (n + 255) / 256);
  dot_kernel<<<blocks, 256>>>(x, y, n, out);
  sum_kernel<<<1, 256>>>(out, blocks);
  check(pfc::gpuGetLastError());
  double result;
  check(pfc::gpuMemcpy(&result, out, sizeof(double), pfc::gpuMemcpyDeviceToHost));
  return result;
}
void OP_NAME(mean)(double *y, const double *x, std::size_t n, int components,
                   double *out, bool subtract) {
  if (!n) return;
  component_mean<<<components, 256>>>(x, n, components, out);
  mean_correction<<<(n + 255) / 256, 256>>>(y, n, components, out, subtract);
  check(pfc::gpuGetLastError());
}
void OP_NAME(dot_store)(const double *x, const double *y, std::size_t n,
                        double *partial, double *result, const ScalarPoll *poll) {
  if (!n) {
    set_if<<<1, 1>>>(result, 0, poll);
    check(pfc::gpuGetLastError());
    return;
  }
  int blocks = std::min<std::size_t>(1024, (n + 255) / 256);
  dot_kernel<<<blocks, 256>>>(x, y, n, partial);
  sum_kernel<<<1, 256>>>(partial, blocks);
  store_if<<<1, 1>>>(result, partial, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(combine_slot)(double *y, const double *a, double sign, const double *x,
                           double b, std::size_t n, const ScalarPoll *poll) {
  if (!n) return;
  combine_slot_kernel<<<(n + 255) / 256, 256>>>(y, a, sign, x, b, n, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(scale_inv)(double *y, const double *a, std::size_t n,
                        const ScalarPoll *poll) {
  if (!n) return;
  scale_inv_kernel<<<(n + 255) / 256, 256>>>(y, a, n, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(scale_mul)(double *y, const double *a, std::size_t n,
                        const ScalarPoll *poll) {
  if (!n) return;
  scale_mul_kernel<<<(n + 255) / 256, 256>>>(y, a, n, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(scale_ratio)(double *y, const double *num, const double *den,
                          std::size_t n, const ScalarPoll *poll) {
  if (!n) return;
  scale_ratio_kernel<<<(n + 255) / 256, 256>>>(y, num, den, n, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(fill)(double *p, std::size_t n, double value) {
  if (!n) return;
  fill_kernel<<<(n + 255) / 256, 256>>>(p, n, value);
  check(pfc::gpuGetLastError());
}
void OP_NAME(set_double)(double *p, double value) {
  set_double_kernel<<<1, 1>>>(p, value);
  check(pfc::gpuGetLastError());
}
void OP_NAME(copy_double)(double *dst, const double *src, const ScalarPoll *poll) {
  copy_double_kernel<<<1, 1>>>(dst, src, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(sqrt_slot)(double *p, const ScalarPoll *poll) {
  sqrt_slot_kernel<<<1, 1>>>(p, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(publish_sqrt)(const double *slot, ScalarPoll *poll) {
  publish_sqrt_kernel<<<1, 1>>>(slot, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(latch_stop)(ScalarPoll *poll, const double *stop) {
  latch_stop_kernel<<<1, 1>>>(poll, stop);
  check(pfc::gpuGetLastError());
}
void OP_NAME(cg_alpha)(const double *rho, const double *curv, double *alpha,
                       ScalarPoll *poll) {
  cg_alpha_kernel<<<1, 1>>>(rho, curv, alpha, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(inc_iter)(ScalarPoll *poll) {
  inc_iter_kernel<<<1, 1>>>(poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(clear_poll)(ScalarPoll *poll) {
  clear_poll_kernel<<<1, 1>>>(poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(clear_flag)(ScalarPoll *poll) {
  clear_flag_kernel<<<1, 1>>>(poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(clear_restart)(ScalarPoll *poll) {
  clear_restart_kernel<<<1, 1>>>(poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(add_h)(double *H, const double *coeff, int j, int ld,
                    const ScalarPoll *poll) {
  add_h_kernel<<<1, 1>>>(H, coeff, j, ld, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(prepare_arnoldi)(double *H, const double *dot, const double *before,
                              int j, int ld, double eps64, double *scale,
                              ScalarPoll *poll) {
  prepare_arnoldi_kernel<<<1, 1>>>(H, dot, before, j, ld, eps64, scale, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(givens)(double *H, double *cs, double *sn, double *g, int j, int ld,
                     const double *stop, ScalarPoll *poll) {
  givens_kernel<<<1, 1>>>(H, cs, sn, g, j, ld, stop, poll);
  check(pfc::gpuGetLastError());
}
void OP_NAME(backsolve)(const double *H, const double *g, double *y, int columns,
                        int ld) {
  if (columns < 1) return;
  backsolve_kernel<<<1, 1>>>(H, g, y, columns, ld);
  check(pfc::gpuGetLastError());
}
#undef OP_NAME
} // namespace pfc::solvers::gpu::detail
