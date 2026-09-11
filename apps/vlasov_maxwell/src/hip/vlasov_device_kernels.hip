// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file vlasov_device_kernels.hip
 * @brief The three device kernels of the 1D2V Vlasov step, and the thin
 *        memory wrappers that keep HIP out of every other translation unit.
 *
 * @details
 * This is the only file hipcc compiles. `device_step_hip.hpp` declares what
 * is here and defines the host glue; the two drivers include that header and
 * are built by the ordinary C++ compiler, so `<hip/hip_runtime.h>` appears
 * exactly once in the application. That is not tidiness for its own sake:
 * the drivers use `std::span`, FFTW through `advect.hpp` and the whole
 * `openpfc` header stack, and every one of those is one more thing that can
 * fail to compile under a different front end for reasons that have nothing
 * to do with the physics.
 *
 * ## The gathers, and the arithmetic they are pinned to
 *
 * Both velocity steps are the same operator on different axes: an output
 * cell is a weighted sum of `p` input cells along one axis, with weights
 * that are constant along that axis. The host loop in `advect.hpp` is
 *
 *     row = 0; for m in 0..p-1: row += w[m] * f(source_m)
 *
 * and these kernels are that loop, in that order, with the same weights
 * (computed on the host and uploaded -- see the header for why). Contraction
 * is switched off with `#pragma clang fp contract(off)`, because a fused
 * multiply-add would round once where the host rounds twice and turn an
 * exact agreement into a one-ulp disagreement per stencil point for no gain
 * this kernel can use: it is bandwidth-bound, not flop-bound, so the fused
 * form buys nothing and costs the strongest statement the parity driver can
 * make.
 *
 * ## The reduction, and why one thread owns one `x`
 *
 * The moment pass is a reduction over `(v_x, v_y)` *for each* `x`. The
 * obvious mapping -- one block per `x`, threads spread over the velocity
 * plane -- reduces beautifully and reads terribly: the padded brick is
 * x-fastest, so two cells that differ only in `v_y` are `npx npy` doubles
 * apart, and a wavefront walking the velocity plane at fixed `x` touches 64
 * separate cache lines per access.
 *
 * So the mapping is transposed: **lane = `x`**, and each thread privately
 * accumulates its own `x` over a chunk of the velocity plane. A wavefront is
 * then 64 consecutive `x` cells -- one contiguous 512-byte read -- and there
 * is no intra-block reduction at all, because no two threads ever share an
 * accumulator. The parallelism that the removed tree reduction would have
 * supplied comes instead from splitting the velocity plane into `nsplit`
 * chunks, which is a second grid dimension and a fixed ~131k threads.
 *
 * The second stage sums the `nsplit` partials per `(quantity, x)` in
 * ascending chunk order, one thread each. That is a deterministic order, so
 * the device result is reproducible run to run -- which matters, because a
 * parity test that passes or fails depending on the scheduler is not a test.
 * It is *not* the host's order, and cannot be; the tolerance that follows
 * from that is derived in `vlasov_hip_parity.cpp`.
 *
 * @see vlasov_maxwell/device_step_hip.hpp for the interface and the design
 * @see vlasov_maxwell/advect.hpp, moments.hpp for the host references
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "vlasov_device_kernels.hip requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

// Ask device_step_hip.hpp for the declarations only: the host glue below the
// guard pulls in MPI, FFTW and the openpfc headers, none of which this
// translation unit needs and all of which hipcc would then have to parse.
#define VLASOV_HIP_KERNELS_ONLY 1
#include <vlasov_maxwell/device_step_hip.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace vlasov::hip {
namespace {

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

/// Linear index of an **owned** cell `(i, j, k)` in the padded brick.
__device__ __host__ inline std::size_t brick_lin(int i, int j, int k, int hw,
                                                 int npx, int npy) noexcept {
  const auto h = static_cast<std::size_t>(hw);
  return (static_cast<std::size_t>(i) + h) +
         (static_cast<std::size_t>(j) + h) * static_cast<std::size_t>(npx) +
         (static_cast<std::size_t>(k) + h) * static_cast<std::size_t>(npx) *
             static_cast<std::size_t>(npy);
}

// ---------------------------------------------------------------------------
// Step B: gather along v_x. Rank-local axis, so a departure point outside
// the velocity box contributes a literal zero -- the zero-inflow condition,
// and the reason no halo appears here.
// ---------------------------------------------------------------------------
__global__ void advect_vx_kernel(const double *__restrict__ f,
                                 double *__restrict__ out, int nx, int nvx,
                                 int nvy, int hw, int npx, int npy,
                                 const int *__restrict__ base,
                                 const double *__restrict__ wts, int p,
                                 int first) {
#pragma clang fp contract(off)
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int jo = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z);
  if (i >= nx || jo >= nvx || k >= nvy) return;

  // The step-B shift depends on (x, v_y) and not on v_x, so the coefficient
  // line is indexed by (k, i) and shared by every v_x on it.
  const std::size_t line =
      static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i);
  const int b = base[line];
  const double *w = wts + line * static_cast<std::size_t>(p);

  double acc = 0.0;
  for (int m = 0; m < p; ++m) {
    const int js = jo + b + first + m;
    if (js < 0 || js >= nvx) continue; // zero inflow
    acc += w[m] * f[brick_lin(i, js, k, hw, npx, npy)];
  }
  out[brick_lin(i, jo, k, hw, npx, npy)] = acc;
}

// ---------------------------------------------------------------------------
// Step C: gather along v_y, the distributed axis. Reads the ghost slabs; the
// host has already exchanged them and already proved the stencil fits.
// ---------------------------------------------------------------------------
__global__ void advect_vy_kernel(const double *__restrict__ f,
                                 double *__restrict__ out, int nx, int nvx,
                                 int nvy, int hw, int npx, int npy,
                                 const int *__restrict__ base,
                                 const double *__restrict__ wts, int p,
                                 int first) {
#pragma clang fp contract(off)
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int ko = static_cast<int>(blockIdx.z);
  if (i >= nx || j >= nvx || ko >= nvy) return;

  // The step-C shift depends on (x, v_x), so the coefficient line is (j, i).
  const std::size_t line =
      static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i);
  const int b = base[line];
  const double *w = wts + line * static_cast<std::size_t>(p);

  double acc = 0.0;
  for (int m = 0; m < p; ++m) {
    const int ks = ko + b + first + m;
    acc += w[m] * f[brick_lin(i, j, ks, hw, npx, npy)];
  }
  out[brick_lin(i, j, ko, hw, npx, npy)] = acc;
}

// ---------------------------------------------------------------------------
// Phase D: the velocity-space reduction.
// ---------------------------------------------------------------------------

/// Threads the first reduction stage aims to launch, independent of `N_x`.
/// Fixing the thread count rather than the chunk size keeps the scratch
/// allocation at a constant ~11 MB across every resolution in the study.
constexpr long long kMomentThreads = 131072;

[[nodiscard]] int moment_splits(const DeviceGeometry &g) {
  const long long plane =
      static_cast<long long>(g.nvx) * static_cast<long long>(g.nvy);
  if (plane <= 0 || g.nx <= 0) return 1;
  long long s = kMomentThreads / static_cast<long long>(g.nx);
  s = std::max<long long>(1, std::min<long long>(s, plane));
  return static_cast<int>(s);
}

__global__ void moments_stage1_kernel(const double *__restrict__ f, int nx,
                                      int nvx, int nvy, int hw, int npx, int npy,
                                      double v_max, double dvx, double dvy,
                                      int vy_offset, int nvy_global, double edge,
                                      int nsplit, double *__restrict__ partial) {
#pragma clang fp contract(off)
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int s = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= nx || s >= nsplit) return;

  const long long plane_cells =
      static_cast<long long>(nvx) * static_cast<long long>(nvy);
  const long long begin =
      plane_cells * static_cast<long long>(s) / static_cast<long long>(nsplit);
  const long long end = plane_cells * static_cast<long long>(s + 1) /
                        static_cast<long long>(nsplit);

  double qn = 0.0, qfx = 0.0, qfy = 0.0, qv2 = 0.0;
  double qshell = 0.0, ql1 = 0.0, ql2 = 0.0, qS = 0.0;
  double qmin = __builtin_huge_val();
  double qmax = -__builtin_huge_val();
  double qface = 0.0;

  for (long long t = begin; t < end; ++t) {
    const int j = static_cast<int>(t / static_cast<long long>(nvy));
    const int k = static_cast<int>(t % static_cast<long long>(nvy));
    const double val = f[brick_lin(i, j, k, hw, npx, npy)];
    // Same closed forms as SimParams::vx_of / vy_of, so the coordinates the
    // device weights by are bitwise the coordinates the host weights by.
    const double vx = -v_max + (static_cast<double>(j) + 0.5) * dvx;
    const int kg = k + vy_offset;
    const double vy = -v_max + (static_cast<double>(kg) + 0.5) * dvy;

    qn += val;
    qfx += vx * val;
    qfy += vy * val;
    qv2 += (vx * vx + vy * vy) * val;
    if (fabs(vx) > edge || fabs(vy) > edge) qshell += val;
    ql1 += fabs(val);
    ql2 += val * val;
    if (val > 0.0) qS -= val * log(val);
    qmin = fmin(qmin, val);
    qmax = fmax(qmax, val);
    if (j == 0 || j == nvx - 1 || kg == 0 || kg == nvy_global - 1) {
      qface = fmax(qface, fabs(val));
    }
  }

  const std::size_t stride =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(nsplit);
  const std::size_t at =
      static_cast<std::size_t>(s) * static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i);
  partial[static_cast<std::size_t>(kMomN) * stride + at] = qn;
  partial[static_cast<std::size_t>(kMomFluxX) * stride + at] = qfx;
  partial[static_cast<std::size_t>(kMomFluxY) * stride + at] = qfy;
  partial[static_cast<std::size_t>(kMomV2) * stride + at] = qv2;
  partial[static_cast<std::size_t>(kMomShell) * stride + at] = qshell;
  partial[static_cast<std::size_t>(kMomL1) * stride + at] = ql1;
  partial[static_cast<std::size_t>(kMomL2Sq) * stride + at] = ql2;
  partial[static_cast<std::size_t>(kMomEntropy) * stride + at] = qS;
  partial[static_cast<std::size_t>(kMomFMin) * stride + at] = qmin;
  partial[static_cast<std::size_t>(kMomFMax) * stride + at] = qmax;
  partial[static_cast<std::size_t>(kMomFFace) * stride + at] = qface;
}

/// Second stage: one thread per `(quantity, x)`, summing the chunks in
/// ascending order. Fixed order, hence reproducible; see the file comment.
__global__ void moments_stage2_kernel(const double *__restrict__ partial,
                                      double *__restrict__ out, int nx,
                                      int nsplit) {
#pragma clang fp contract(off)
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int q = static_cast<int>(blockIdx.y);
  if (i >= nx || q >= kMomQuantities) return;

  const std::size_t stride =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(nsplit);
  const double *src = partial + static_cast<std::size_t>(q) * stride;

  if (q == kMomFMin) {
    double acc = __builtin_huge_val();
    for (int s = 0; s < nsplit; ++s) {
      acc = fmin(acc, src[static_cast<std::size_t>(s) * static_cast<std::size_t>(nx) +
                          static_cast<std::size_t>(i)]);
    }
    out[static_cast<std::size_t>(q) * static_cast<std::size_t>(nx) +
        static_cast<std::size_t>(i)] = acc;
    return;
  }
  if (q == kMomFMax || q == kMomFFace) {
    double acc = (q == kMomFMax) ? -__builtin_huge_val() : 0.0;
    for (int s = 0; s < nsplit; ++s) {
      acc = fmax(acc, src[static_cast<std::size_t>(s) * static_cast<std::size_t>(nx) +
                          static_cast<std::size_t>(i)]);
    }
    out[static_cast<std::size_t>(q) * static_cast<std::size_t>(nx) +
        static_cast<std::size_t>(i)] = acc;
    return;
  }
  double acc = 0.0;
  for (int s = 0; s < nsplit; ++s) {
    acc += src[static_cast<std::size_t>(s) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(i)];
  }
  out[static_cast<std::size_t>(q) * static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i)] = acc;
}

/// Launch shape for a gather: 64 lanes along the contiguous `x` axis (one
/// full wavefront on CDNA), four rows of the second axis, one block per
/// plane of the third.
void gather_launch(int nx, int nvx, int nvy, dim3 *grid, dim3 *block) {
  block->x = 64;
  block->y = 4;
  block->z = 1;
  grid->x = (static_cast<unsigned>(nx) + block->x - 1) / block->x;
  grid->y = (static_cast<unsigned>(nvx) + block->y - 1) / block->y;
  grid->z = static_cast<unsigned>(std::max(nvy, 1));
}

} // namespace

// ---------------------------------------------------------------------------
// Memory and control
// ---------------------------------------------------------------------------

int device_count() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 0;
  return n;
}

void bind_local_device(int local_rank) {
  const int n = device_count();
  if (n < 1) {
    throw std::runtime_error(
        "bind_local_device: no HIP device is visible to this rank. On LUMI-G "
        "do not set ROCR_VISIBLE_DEVICES; every GCD must stay visible and the "
        "binding is done here.");
  }
  hip_check(hipSetDevice(local_rank % n), "hipSetDevice");
}

const char *device_name() {
  static char name[256] = {0};
  if (name[0] != '\0') return name;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return "unknown";
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return "unknown";
  std::snprintf(name, sizeof(name), "%s (gcnArch %s)", prop.name,
                prop.gcnArchName);
  return name;
}

double *device_alloc(std::size_t n_doubles) {
  void *p = nullptr;
  hip_check(hipMalloc(&p, n_doubles * sizeof(double)), "hipMalloc(double)");
  return static_cast<double *>(p);
}

int *device_alloc_int(std::size_t n_ints) {
  void *p = nullptr;
  hip_check(hipMalloc(&p, n_ints * sizeof(int)), "hipMalloc(int)");
  return static_cast<int *>(p);
}

void device_free(void *p) noexcept {
  if (p != nullptr) (void)hipFree(p);
}

void device_upload(double *dst_dev, const double *src_host, std::size_t n) {
  hip_check(hipMemcpy(dst_dev, src_host, n * sizeof(double),
                      hipMemcpyHostToDevice),
            "hipMemcpy H2D");
}

void device_upload_int(int *dst_dev, const int *src_host, std::size_t n) {
  hip_check(hipMemcpy(dst_dev, src_host, n * sizeof(int), hipMemcpyHostToDevice),
            "hipMemcpy H2D (int)");
}

void device_download(double *dst_host, const double *src_dev, std::size_t n) {
  hip_check(hipMemcpy(dst_host, src_dev, n * sizeof(double),
                      hipMemcpyDeviceToHost),
            "hipMemcpy D2H");
}

void device_zero(double *dst_dev, std::size_t n) {
  hip_check(hipMemset(dst_dev, 0, n * sizeof(double)), "hipMemset");
}

void device_synchronize() {
  hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

// ---------------------------------------------------------------------------
// Kernel launchers
// ---------------------------------------------------------------------------

void advect_vx_gather_hip(const double *f_dev, double *out_dev,
                          const DeviceGeometry &g, const int *base_dev,
                          const double *wts_dev, int interp_order, int first) {
  dim3 grid{};
  dim3 block{};
  gather_launch(g.nx, g.nvx, g.nvy, &grid, &block);
  hipLaunchKernelGGL(advect_vx_kernel, grid, block, 0, 0, f_dev, out_dev, g.nx,
                     g.nvx, g.nvy, g.halo, g.npx(), g.npy(), base_dev, wts_dev,
                     interp_order, first);
  hip_check(hipGetLastError(), "advect_vx_gather_hip launch");
}

void advect_vy_gather_hip(const double *f_dev, double *out_dev,
                          const DeviceGeometry &g, const int *base_dev,
                          const double *wts_dev, int interp_order, int first) {
  dim3 grid{};
  dim3 block{};
  gather_launch(g.nx, g.nvx, g.nvy, &grid, &block);
  hipLaunchKernelGGL(advect_vy_kernel, grid, block, 0, 0, f_dev, out_dev, g.nx,
                     g.nvx, g.nvy, g.halo, g.npx(), g.npy(), base_dev, wts_dev,
                     interp_order, first);
  hip_check(hipGetLastError(), "advect_vy_gather_hip launch");
}

std::size_t moments_scratch_doubles(const DeviceGeometry &g) {
  return static_cast<std::size_t>(kMomQuantities) *
         static_cast<std::size_t>(g.nx) *
         static_cast<std::size_t>(moment_splits(g));
}

void moments_hip(const double *f_dev, const DeviceGeometry &g, double v_max,
                 double dvx, double dvy, int vy_offset, int nvy_global,
                 double edge, double *scratch_dev, double *out_dev) {
  const int nsplit = moment_splits(g);

  dim3 block1{64, 4, 1};
  dim3 grid1{(static_cast<unsigned>(g.nx) + block1.x - 1) / block1.x,
             (static_cast<unsigned>(nsplit) + block1.y - 1) / block1.y, 1};
  hipLaunchKernelGGL(moments_stage1_kernel, grid1, block1, 0, 0, f_dev, g.nx,
                     g.nvx, g.nvy, g.halo, g.npx(), g.npy(), v_max, dvx, dvy,
                     vy_offset, nvy_global, edge, nsplit, scratch_dev);
  hip_check(hipGetLastError(), "moments_stage1 launch");

  dim3 block2{64, 1, 1};
  dim3 grid2{(static_cast<unsigned>(g.nx) + block2.x - 1) / block2.x,
             static_cast<unsigned>(kMomQuantities), 1};
  hipLaunchKernelGGL(moments_stage2_kernel, grid2, block2, 0, 0, scratch_dev,
                     out_dev, g.nx, nsplit);
  hip_check(hipGetLastError(), "moments_stage2 launch");
}

} // namespace vlasov::hip
