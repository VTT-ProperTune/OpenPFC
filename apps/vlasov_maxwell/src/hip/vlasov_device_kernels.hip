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
#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif

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
/// Fixing the thread count rather than the chunk size keeps the launch shape
/// sane across three orders of magnitude of phase-space size.
constexpr long long kMomentThreads = 131072;
/// Fewest velocity cells one thread should be given. Below this the
/// partial-sum array the first stage writes -- eleven doubles per
/// `(x, chunk)` -- grows to the size of the brick being reduced, and the
/// reduction stops measuring the brick and starts measuring its own
/// scratch. At `64^3` before this clamp, phase D was *slower* than at
/// `128^3` for exactly that reason.
constexpr long long kMomentMinCellsPerThread = 16;
/// Chunks reduced together inside one block, along the second block
/// dimension. Each block folds them into one partial, which divides the
/// scratch traffic by this factor at the cost of one small shared-memory
/// tree per quantity.
constexpr unsigned kMomentGroup = 8;

[[nodiscard]] int moment_splits(const DeviceGeometry &g) {
  const long long plane =
      static_cast<long long>(g.nvx) * static_cast<long long>(g.nvy);
  if (plane <= 0 || g.nx <= 0) return 1;
  long long s = kMomentThreads / static_cast<long long>(g.nx);
  s = std::min<long long>(s, std::max<long long>(1, plane / kMomentMinCellsPerThread));
  s = std::max<long long>(1, std::min<long long>(s, plane));
  return static_cast<int>(s);
}

/// Partial groups the first stage writes, i.e. `ceil(nsplit / kMomentGroup)`.
[[nodiscard]] int moment_groups(const DeviceGeometry &g) {
  const unsigned n = static_cast<unsigned>(moment_splits(g));
  return static_cast<int>((n + kMomentGroup - 1) / kMomentGroup);
}

__global__ void moments_stage1_kernel(const double *__restrict__ f, int nx,
                                      int nvx, int nvy, int hw, int npx, int npy,
                                      double v_max, double dvx, double dvy,
                                      int vy_offset, int nvy_global, double edge,
                                      int nsplit, int ngroups,
                                      double *__restrict__ partial) {
#pragma clang fp contract(off)
  extern __shared__ double sh[];
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int i = static_cast<int>(blockIdx.x * blockDim.x) + tx;
  const int s = static_cast<int>(blockIdx.y * blockDim.y) + ty;
  // No early return: every thread has to reach the __syncthreads() below.
  const bool active = (i < nx) && (s < nsplit);

  double q[kMomQuantities];
  for (int t = 0; t < kMomQuantities; ++t) q[t] = 0.0;
  q[kMomFMin] = __builtin_huge_val();
  q[kMomFMax] = -__builtin_huge_val();

  if (active) {
    const long long plane_cells =
        static_cast<long long>(nvx) * static_cast<long long>(nvy);
    const long long begin =
        plane_cells * static_cast<long long>(s) / static_cast<long long>(nsplit);
    const long long end = plane_cells * static_cast<long long>(s + 1) /
                          static_cast<long long>(nsplit);
    // v_x innermost, not v_y: at fixed x, consecutive v_x are npx doubles
    // apart and consecutive v_y are npx*npy apart, so walking v_x keeps a
    // thread's successive reads inside a few kilobytes instead of half a
    // megabyte. The reduction's value does not depend on the order; its
    // speed does, by a factor this file would rather not pay.
    for (long long t = begin; t < end; ++t) {
      const int k = static_cast<int>(t / static_cast<long long>(nvx));
      const int j = static_cast<int>(t % static_cast<long long>(nvx));
      const double val = f[brick_lin(i, j, k, hw, npx, npy)];
      // Same closed forms as SimParams::vx_of / vy_of, so the coordinates
      // the device weights by are bitwise the coordinates the host weights
      // by.
      const double vx = -v_max + (static_cast<double>(j) + 0.5) * dvx;
      const int kg = k + vy_offset;
      const double vy = -v_max + (static_cast<double>(kg) + 0.5) * dvy;

      q[kMomN] += val;
      q[kMomFluxX] += vx * val;
      q[kMomFluxY] += vy * val;
      q[kMomV2] += (vx * vx + vy * vy) * val;
      if (fabs(vx) > edge || fabs(vy) > edge) q[kMomShell] += val;
      q[kMomL1] += fabs(val);
      q[kMomL2Sq] += val * val;
      if (val > 0.0) q[kMomEntropy] -= val * log(val);
      q[kMomFMin] = fmin(q[kMomFMin], val);
      q[kMomFMax] = fmax(q[kMomFMax], val);
      if (j == 0 || j == nvx - 1 || kg == 0 || kg == nvy_global - 1) {
        q[kMomFFace] = fmax(q[kMomFFace], fabs(val));
      }
    }
  }

  // Fold the block's chunks together, one quantity at a time through a
  // single shared buffer. The tree shape is fixed by blockDim, so this is
  // reproducible run to run -- which matters, because a parity test that
  // passes or fails depending on the scheduler is not a test.
  const int lane = ty * static_cast<int>(blockDim.x) + tx;
  for (int t = 0; t < kMomQuantities; ++t) {
    sh[lane] = q[t];
    __syncthreads();
    for (unsigned half = blockDim.y >> 1; half > 0; half >>= 1) {
      if (static_cast<unsigned>(ty) < half) {
        const int other = lane + static_cast<int>(half * blockDim.x);
        if (t == kMomFMin) {
          sh[lane] = fmin(sh[lane], sh[other]);
        } else if (t == kMomFMax || t == kMomFFace) {
          sh[lane] = fmax(sh[lane], sh[other]);
        } else {
          sh[lane] += sh[other];
        }
      }
      __syncthreads();
    }
    if (ty == 0 && i < nx) {
      const std::size_t stride =
          static_cast<std::size_t>(nx) * static_cast<std::size_t>(ngroups);
      partial[static_cast<std::size_t>(t) * stride +
              static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(i)] = sh[tx];
    }
    __syncthreads();
  }
}

/// Second stage: one thread per `(quantity, x)`, combining the block
/// partials in ascending order. Fixed order, hence reproducible; see the
/// file comment.
__global__ void moments_stage2_kernel(const double *__restrict__ partial,
                                      double *__restrict__ out, int nx,
                                      int ngroups) {
#pragma clang fp contract(off)
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int q = static_cast<int>(blockIdx.y);
  if (i >= nx || q >= kMomQuantities) return;

  const std::size_t stride =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ngroups);
  const double *src = partial + static_cast<std::size_t>(q) * stride;

  double acc = 0.0;
  if (q == kMomFMin) acc = __builtin_huge_val();
  if (q == kMomFMax) acc = -__builtin_huge_val();
  for (int g = 0; g < ngroups; ++g) {
    const double v =
        src[static_cast<std::size_t>(g) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(i)];
    if (q == kMomFMin) {
      acc = fmin(acc, v);
    } else if (q == kMomFMax || q == kMomFFace) {
      acc = fmax(acc, v);
    } else {
      acc += v;
    }
  }
  out[static_cast<std::size_t>(q) * static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i)] = acc;
}

// ---------------------------------------------------------------------------
// Lagrange weights, one line per thread.
//
// The host computes the *departure offset* of each advected line -- an
// integer cell and a fraction in [0,1) -- and uploads it; this kernel turns
// the fraction into `p` weights. The split is deliberate. The offset costs a
// handful of flops per line and needs the fields, the halo guard and the
// collective that goes with it, all of which belong on the host. The weights
// cost `p(p-1)` *divisions* per line, which is what actually hurts: measured
// on LUMI-C at 256^3 the host spent 5.7 ms per Strang step on nothing but
// those divisions, more than every device kernel in this file put together.
//
// Moving them here costs nothing in fidelity. The expression is the one
// `vlasov::lagrange_weights` evaluates, in the same order, out of the same
// `frac`, and it contains only subtraction, division and multiplication --
// operations IEEE-754 specifies exactly, with no fused form available to
// either compiler. So the weights are bitwise the host's.
// ---------------------------------------------------------------------------
__global__ void lagrange_weights_kernel(const double *__restrict__ frac,
                                        double *__restrict__ wts,
                                        long long n_lines, int p, int first) {
#pragma clang fp contract(off)
  const long long line =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (line >= n_lines) return;
  const double f = frac[line];
  double *w = wts + line * static_cast<long long>(p);
  for (int m = 0; m < p; ++m) {
    const double tm = static_cast<double>(first + m);
    double prod = 1.0;
    for (int l = 0; l < p; ++l) {
      if (l == m) continue;
      const double tl = static_cast<double>(first + l);
      prod *= (f - tl) / (tm - tl);
    }
    w[m] = prod;
  }
}

// ---------------------------------------------------------------------------
// Phase A on the device: the exact spectral translation along x.
//
// Every line of constant (v_x, v_y) is translated by v_x dt / dx cells, which
// depends on v_x alone. On a periodic axis that is a diagonal operator in
// Fourier space, so the whole phase is: pack the owned brick into an
// unpadded, x-contiguous buffer; one batched real-to-complex transform of
// N_vx N_vy lines; one multiply by a per-(v_x, mode) unit phasor; one batched
// complex-to-real transform back; unpack.
//
// The pack is not an accident of laziness. hipFFT's advanced layout can
// express a constant distance between batch members, and the padded brick has
// two different distances (npx between v_x lines, npx*npy between v_y planes),
// so transforming in place would mean N_vy separate plan executions -- at
// 256^3 that is 256 launches of ~5 us each, 1.3 ms of pure launch latency,
// against roughly 0.5 ms for the four extra streaming passes the pack costs.
//
// The 1/N normalisation is folded into the phase table rather than applied in
// a fifth pass. That changes the rounding by one ulp relative to the host's
// "multiply, then scale", which is irrelevant next to the difference between
// FFTW and rocFFT that this phase already carries; see vlasov_hip_parity.cpp
// for the tolerance and its derivation.
// ---------------------------------------------------------------------------

/// `out[(k nvx + j) nx + i] = brick(i, j, k)`.
__global__ void pack_x_kernel(const double *__restrict__ f,
                              double *__restrict__ out, int nx, int nvx,
                              int nvy, int hw, int npx, int npy) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z);
  if (i >= nx || j >= nvx || k >= nvy) return;
  const std::size_t dst =
      (static_cast<std::size_t>(k) * static_cast<std::size_t>(nvx) +
       static_cast<std::size_t>(j)) *
          static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i);
  out[dst] = f[brick_lin(i, j, k, hw, npx, npy)];
}

/// The inverse of @ref pack_x_kernel.
__global__ void unpack_x_kernel(const double *__restrict__ in,
                                double *__restrict__ f, int nx, int nvx,
                                int nvy, int hw, int npx, int npy) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int k = static_cast<int>(blockIdx.z);
  if (i >= nx || j >= nvx || k >= nvy) return;
  const std::size_t src =
      (static_cast<std::size_t>(k) * static_cast<std::size_t>(nvx) +
       static_cast<std::size_t>(j)) *
          static_cast<std::size_t>(nx) +
      static_cast<std::size_t>(i);
  f[brick_lin(i, j, k, hw, npx, npy)] = in[src];
}

/**
 * Phase table, `nvx x nk` unit phasors scaled by `1/N_x`.
 *
 * The argument is reduced with `fmod` before the trigonometry, exactly as
 * `XShiftPlan::prepare` does, so accuracy does not decay with the number of
 * box lengths travelled. Mode 0 is forced to exactly `1/N_x` because it is
 * the particle number and must not acquire a phase; the Nyquist mode of an
 * even-length real transform keeps only its real part, or the result stops
 * being real.
 */
__global__ void x_phase_kernel(double *__restrict__ phase, int nx, int nk,
                               int nvx, double v_max, double dvx, double dt,
                               double dx) {
  const int m = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (m >= nk || j >= nvx) return;
  const double two_pi = 6.283185307179586476925286766559;
  const double vx = -v_max + (static_cast<double>(j) + 0.5) * dvx;
  const double delta = vx * dt / dx;
  const double n = static_cast<double>(nx);
  const double arg = fmod(static_cast<double>(m) * delta, n);
  const double theta = -two_pi * arg / n;
  double pr = cos(theta);
  double pim = sin(theta);
  if (m == 0) {
    pr = 1.0;
    pim = 0.0;
  }
  if ((nx % 2) == 0 && m == nk - 1) pim = 0.0;
  const double inv = 1.0 / n;
  const std::size_t at =
      2 * (static_cast<std::size_t>(j) * static_cast<std::size_t>(nk) +
           static_cast<std::size_t>(m));
  phase[at] = pr * inv;
  phase[at + 1] = pim * inv;
}

/// Multiply every transformed line by its phasor. `b = k nvx + j`, so the
/// phase row is `b % nvx`.
__global__ void x_multiply_kernel(double *__restrict__ spec,
                                  const double *__restrict__ phase, int nx,
                                  int nk, int nvx, long long batch) {
#pragma clang fp contract(off)
  const int m = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const long long b =
      static_cast<long long>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (m >= nk || b >= batch) return;
  const int j = static_cast<int>(b % static_cast<long long>(nvx));
  const std::size_t at =
      2 * (static_cast<std::size_t>(b) * static_cast<std::size_t>(nk) +
           static_cast<std::size_t>(m));
  const std::size_t pat =
      2 * (static_cast<std::size_t>(j) * static_cast<std::size_t>(nk) +
           static_cast<std::size_t>(m));
  const double re = spec[at];
  const double im = spec[at + 1];
  const double pr = phase[pat];
  const double pim = phase[pat + 1];
  const double outr = re * pr - im * pim;
  double outi = re * pim + im * pr;
  // A real signal has a real DC and, for even N, a real Nyquist coefficient.
  // Force it: the inverse transform assumes Hermitian input and silently
  // ignores anything else, so a stray imaginary part would be a difference
  // between this path and the host that no test could see.
  if (m == 0) outi = 0.0;
  if ((nx % 2) == 0 && m == nk - 1) outi = 0.0;
  spec[at] = outr;
  spec[at + 1] = outi;
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
         static_cast<std::size_t>(moment_groups(g));
}

void moments_hip(const double *f_dev, const DeviceGeometry &g, double v_max,
                 double dvx, double dvy, int vy_offset, int nvy_global,
                 double edge, double *scratch_dev, double *out_dev) {
  const int nsplit = moment_splits(g);
  const int ngroups = moment_groups(g);

  dim3 block1{64, kMomentGroup, 1};
  dim3 grid1{(static_cast<unsigned>(g.nx) + block1.x - 1) / block1.x,
             static_cast<unsigned>(ngroups), 1};
  const std::size_t shbytes =
      static_cast<std::size_t>(block1.x) * block1.y * sizeof(double);
  hipLaunchKernelGGL(moments_stage1_kernel, grid1, block1, shbytes, 0, f_dev,
                     g.nx, g.nvx, g.nvy, g.halo, g.npx(), g.npy(), v_max, dvx,
                     dvy, vy_offset, nvy_global, edge, nsplit, ngroups,
                     scratch_dev);
  hip_check(hipGetLastError(), "moments_stage1 launch");

  dim3 block2{64, 1, 1};
  dim3 grid2{(static_cast<unsigned>(g.nx) + block2.x - 1) / block2.x,
             static_cast<unsigned>(kMomQuantities), 1};
  hipLaunchKernelGGL(moments_stage2_kernel, grid2, block2, 0, 0, scratch_dev,
                     out_dev, g.nx, ngroups);
  hip_check(hipGetLastError(), "moments_stage2 launch");
}


void lagrange_weights_hip(const double *frac_dev, double *wts_dev,
                          std::size_t n_lines, int interp_order, int first) {
  if (n_lines == 0) return;
  // Named dim3 variables, not brace-initialisers: hipLaunchKernelGGL is a
  // macro and the commas inside `dim3{a, b, c}` would be read as argument
  // separators.
  dim3 block{256, 1, 1};
  dim3 grid{static_cast<unsigned>((n_lines + 255) / 256), 1, 1};
  hipLaunchKernelGGL(lagrange_weights_kernel, grid, block, 0, 0, frac_dev,
                     wts_dev, static_cast<long long>(n_lines), interp_order,
                     first);
  hip_check(hipGetLastError(), "lagrange_weights_hip launch");
}

// ---------------------------------------------------------------------------
// The x-shift plan.
// ---------------------------------------------------------------------------

struct XShiftDevicePlan {
  DeviceGeometry g{};
  int nk{0};
  long long batch{0};
  double *xbuf{nullptr};  ///< unpadded, x-contiguous copy of the owned brick
  double *spec{nullptr};  ///< interleaved complex, `batch * nk`
  double *phase{nullptr}; ///< interleaved complex, `nvx * nk`
  hipfftHandle fwd{};
  hipfftHandle bwd{};
  bool planned{false};
  /// Cached so the phase table is only rebuilt when the shift changes.
  double last_dt{0.0};
  bool have_phase{false};
};

namespace {

void fft_check(hipfftResult r, const char *what) {
  if (r != HIPFFT_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": hipFFT error " +
                             std::to_string(static_cast<int>(r)));
  }
}

} // namespace

bool x_shift_device_available() { return true; }

XShiftDevicePlan *x_shift_plan_create(const DeviceGeometry &g) {
  if (g.nx < 2) {
    throw std::runtime_error(
        "x_shift_plan_create: need at least two x cells to transform");
  }
  auto *p = new XShiftDevicePlan();
  p->g = g;
  p->nk = g.nx / 2 + 1;
  p->batch = static_cast<long long>(g.nvx) * static_cast<long long>(g.nvy);
  try {
    p->xbuf = device_alloc(g.owned_cells());
    p->spec = device_alloc(2 * static_cast<std::size_t>(p->batch) *
                           static_cast<std::size_t>(p->nk));
    p->phase = device_alloc(2 * static_cast<std::size_t>(g.nvx) *
                            static_cast<std::size_t>(p->nk));
    int n = g.nx;
    int inembed[1] = {g.nx};
    int onembed[1] = {p->nk};
    fft_check(hipfftPlanMany(&p->fwd, 1, &n, inembed, 1, g.nx, onembed, 1,
                             p->nk, HIPFFT_D2Z,
                             static_cast<int>(p->batch)),
              "hipfftPlanMany D2Z");
    fft_check(hipfftPlanMany(&p->bwd, 1, &n, onembed, 1, p->nk, inembed, 1,
                             g.nx, HIPFFT_Z2D, static_cast<int>(p->batch)),
              "hipfftPlanMany Z2D");
    p->planned = true;
  } catch (...) {
    x_shift_plan_destroy(p);
    throw;
  }
  return p;
}

void x_shift_plan_destroy(XShiftDevicePlan *p) {
  if (p == nullptr) return;
  if (p->planned) {
    (void)hipfftDestroy(p->fwd);
    (void)hipfftDestroy(p->bwd);
  }
  device_free(p->xbuf);
  device_free(p->spec);
  device_free(p->phase);
  delete p;
}

void advect_x_hip(XShiftDevicePlan *p, double *brick_dev, double v_max,
                  double dvx, double dt, double dx) {
  if (p == nullptr) throw std::runtime_error("advect_x_hip: null plan");
  const DeviceGeometry &g = p->g;

  dim3 block{64, 4, 1};
  dim3 grid{(static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
            (static_cast<unsigned>(g.nvx) + block.y - 1) / block.y,
            static_cast<unsigned>(std::max(g.nvy, 1))};

  if (!p->have_phase || p->last_dt != dt) {
    dim3 pb{64, 4, 1};
    dim3 pg{(static_cast<unsigned>(p->nk) + pb.x - 1) / pb.x,
            (static_cast<unsigned>(g.nvx) + pb.y - 1) / pb.y, 1};
    hipLaunchKernelGGL(x_phase_kernel, pg, pb, 0, 0, p->phase, g.nx, p->nk,
                       g.nvx, v_max, dvx, dt, dx);
    hip_check(hipGetLastError(), "x_phase_kernel launch");
    p->last_dt = dt;
    p->have_phase = true;
  }

  hipLaunchKernelGGL(pack_x_kernel, grid, block, 0, 0, brick_dev, p->xbuf,
                     g.nx, g.nvx, g.nvy, g.halo, g.npx(), g.npy());
  hip_check(hipGetLastError(), "pack_x_kernel launch");

  fft_check(hipfftExecD2Z(p->fwd, p->xbuf,
                          reinterpret_cast<hipfftDoubleComplex *>(p->spec)),
            "hipfftExecD2Z");

  dim3 mb{64, 4, 1};
  dim3 mg{(static_cast<unsigned>(p->nk) + mb.x - 1) / mb.x,
          static_cast<unsigned>((p->batch + mb.y - 1) / mb.y), 1};
  hipLaunchKernelGGL(x_multiply_kernel, mg, mb, 0, 0, p->spec, p->phase, g.nx,
                     p->nk, g.nvx, p->batch);
  hip_check(hipGetLastError(), "x_multiply_kernel launch");

  fft_check(hipfftExecZ2D(p->bwd,
                          reinterpret_cast<hipfftDoubleComplex *>(p->spec),
                          p->xbuf),
            "hipfftExecZ2D");

  hipLaunchKernelGGL(unpack_x_kernel, grid, block, 0, 0, p->xbuf, brick_dev,
                     g.nx, g.nvx, g.nvy, g.halo, g.npx(), g.npy());
  hip_check(hipGetLastError(), "unpack_x_kernel launch");
}

} // namespace vlasov::hip
