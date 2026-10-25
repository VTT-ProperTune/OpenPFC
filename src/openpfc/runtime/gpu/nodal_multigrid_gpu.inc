// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file nodal_multigrid_gpu.inc
 * @brief Device kernels of `pfc::solvers::NodalMultigrid` (included from .cu/.hip).
 *
 * Node kernels use a 3-D launch with x fastest and call the point functions of
 * nodal_multigrid_kernels.hpp shared with the host. The owned dot product sums
 * fixed per-block partials in a fixed order, so it is deterministic.
 */

#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/nodal_multigrid_gpu.hpp>

namespace pfc::solvers::gpu::nmg {
namespace {

namespace k = ::pfc::solvers::nmg;

inline void check(const char *what) {
  ::pfc::gpuError_t e = ::pfc::gpuGetLastError();
  if (e != ::pfc::gpuSuccess)
    throw std::runtime_error(std::string("NodalMultigrid ") + what + ": " +
                             ::pfc::gpuGetErrorString(e));
}

struct Region {
  int n[3];
};

inline dim3 block3() { return dim3(64, 4, 1); }
inline dim3 grid3(const int n[3]) {
  return dim3(static_cast<unsigned>((n[0] + 63) / 64),
              static_cast<unsigned>((n[1] + 3) / 4), static_cast<unsigned>(n[2]));
}

#define OPENPFC_NMG_NODE(R)                                                         \
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);            \
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);            \
  const int kk = static_cast<int>(blockIdx.z);                                      \
  if (i >= (R).n[0] || j >= (R).n[1] || kk >= (R).n[2]) return;

constexpr int kThreads = 256;
constexpr int kDotBlocks = 1024;

__global__ void fill_kernel(double *p, std::size_t n, double v) {
  for (std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n;
       t += std::size_t(gridDim.x) * blockDim.x)
    p[t] = v;
}
__global__ void axpy_owned_kernel(Level L, double a, const double *x, double *y) {
  OPENPFC_NMG_NODE(L)
  const std::size_t s = k::vindex(L, i, j, kk);
  y[s] += a * x[s];
}
__global__ void add_owned_kernel(Level L, double *y, double v) {
  OPENPFC_NMG_NODE(L)
  y[k::vindex(L, i, j, kk)] += v;
}
__global__ void hash_owned_kernel(Level L, double *x) {
  OPENPFC_NMG_NODE(L)
  x[k::vindex(L, i, j, kk)] =
      k::hash_value(L.low[0] + i, L.low[1] + j, L.low[2] + kk);
}
__global__ void copy_owned_kernel(Level L, const double *src, int sh, double *dst,
                                  int dh) {
  OPENPFC_NMG_NODE(L)
  dst[k::index(L, dh, i, j, kk)] = src[k::index(L, sh, i, j, kk)];
}
__global__ void dot_kernel(Level L, const double *x, const double *y, double *out) {
  __shared__ double sums[kThreads];
  const std::size_t n = k::owned_count(L);
  double s = 0.0;
  for (std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n;
       t += std::size_t(gridDim.x) * blockDim.x) {
    int i, j, kk;
    k::owned_ijk(L, t, i, j, kk);
    const std::size_t q = k::vindex(L, i, j, kk);
    s += x[q] * y[q];
  }
  sums[threadIdx.x] = s;
  __syncthreads();
  for (int stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sums[0];
}
__global__ void sum_kernel(double *p, int n) {
  __shared__ double sums[kThreads];
  double s = 0.0;
  for (int i = threadIdx.x; i < n; i += kThreads) s += p[i];
  sums[threadIdx.x] = s;
  __syncthreads();
  for (int stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) p[0] = sums[0];
}
__global__ void apply_kernel(Level L, Operator A, const double *u, double *out) {
  OPENPFC_NMG_NODE(L)
  out[k::vindex(L, i, j, kk)] = k::apply(L, A, u, i, j, kk);
}
// First nonperiodic GPU entry point: a rank-owned constant Q1 matrix-free
// action. Halo transport remains owned by OpenPFC's device HaloExchange.
__global__ void dirichlet_apply_kernel(Level L, k::dirichlet::Weights w,
                                       const double *u, double *out) {
  OPENPFC_NMG_NODE(L)
  out[k::index(L, k::dirichlet::kHalo, i, j, kk)] =
      k::dirichlet::action(L, w, u, i, j, kk);
}
// The field and device buffer ownership stay with OpenPFC. This kernel
// implements the same damped-Jacobi point update used by the host V-cycle;
// input and output buffers must not alias.
__global__ void dirichlet_jacobi_kernel(Level L, k::dirichlet::Weights w,
                                        const double *old_u, const double *rhs,
                                        double *new_u, double scale) {
  OPENPFC_NMG_NODE(L)
  const auto index = k::index(L, k::dirichlet::kHalo, i, j, kk);
  new_u[index] =
      k::dirichlet::jacobi(L, w, old_u, old_u[index], rhs[index], scale, i, j, kk);
}
__global__ void dirichlet_restrict_kernel(Level F, const double *fine, Level C,
                                          double *coarse) {
  OPENPFC_NMG_NODE(C)
  coarse[k::index(C, k::dirichlet::kHalo, i, j, kk)] =
      k::dirichlet::restrict_pt(F, fine, C.low[0] + i, C.low[1] + j, C.low[2] + kk);
}
__global__ void dirichlet_prolong_kernel(Level F, double *fine, Level C,
                                         const double *coarse) {
  OPENPFC_NMG_NODE(F)
  const auto index = k::index(F, k::dirichlet::kHalo, i, j, kk);
  fine[index] +=
      k::dirichlet::prolong_p(C, coarse, F.low[0] + i, F.low[1] + j, F.low[2] + kk);
}
__global__ void scaled_apply_kernel(Level L, Operator A, const double *u,
                                    double *out) {
  OPENPFC_NMG_NODE(L)
  out[k::vindex(L, i, j, kk)] =
      k::apply(L, A, u, i, j, kk) / k::diagonal(L, A, i, j, kk);
}
__global__ void residual_kernel(Level L, Operator A, const double *f,
                                const double *u, double *r, int zero_u) {
  OPENPFC_NMG_NODE(L)
  k::residual(L, A, f, u, r, i, j, kk, zero_u);
}
__global__ void chebyshev_kernel(Level L, Operator A, const double *f,
                                 const double *u, const double *u_prev, double *out,
                                 double c1, double c2, int zero) {
  OPENPFC_NMG_NODE(L)
  k::chebyshev(L, A, f, u, u_prev, out, c1, c2, i, j, kk, zero);
}
__global__ void gauss_seidel_kernel(Level L, Operator A, const double *f, double *u,
                                    int colour) {
  OPENPFC_NMG_NODE(L)
  k::gauss_seidel(L, A, f, u, colour, i, j, kk);
}
__global__ void restrict_residual_kernel(Level F, const double *rf, Level G,
                                         double *fg, Region R, int o0, int o1,
                                         int o2) {
  OPENPFC_NMG_NODE(R)
  const int off[3] = {o0, o1, o2};
  k::restrict_residual(F, rf, G, fg, off, i, j, kk);
}
__global__ void prolong_kernel(Level F, const double *uf, double *out, Level G,
                               const double *ug, int o0, int o1, int o2) {
  OPENPFC_NMG_NODE(F)
  const int off[3] = {o0, o1, o2};
  k::prolong(F, uf, out, G, ug, off, i, j, kk);
}
__global__ void restrict_coefficient_kernel(Level F, const double *af, Level G,
                                            double *ag, Region R, int o0, int o1,
                                            int o2) {
  OPENPFC_NMG_NODE(R)
  const int off[3] = {o0, o1, o2};
  k::restrict_coefficient(F, af, G, ag, off, i, j, kk);
}
__global__ void galerkin_kernel(Level F, Operator Af, Level G, double *sg,
                                std::size_t gplane, Region R, int o0, int o1,
                                int o2) {
  OPENPFC_NMG_NODE(R)
  const int off[3] = {o0, o1, o2};
  k::galerkin(F, Af, G, sg, gplane, off, i, j, kk);
}
#undef OPENPFC_NMG_NODE

// ---- element-level rows marching along z (fine levels) -------------------
//
// Each thread owns one (i, j) column of KC nodes and keeps the 3 x 3 x 3 window
// of u and the 2 x 2 x 2 element coefficients in registers, so each new plane
// costs 9 u and 4 coefficient loads instead of 27 and 8.
constexpr int KC = 32;

// Epilogues get the row action, the diagonal and the centre value of u.
struct ApplyEpi {
  double *out;
  __device__ void operator()(std::size_t s, double au, double, double) const {
    out[s] = au;
  }
};
struct ResidualEpi {
  const double *f;
  double *r;
  __device__ void operator()(std::size_t s, double au, double, double) const {
    r[s] = f[s] - au;
  }
};
struct ScaledEpi {
  double *out;
  __device__ void operator()(std::size_t s, double au, double d, double) const {
    out[s] = au / d;
  }
};
struct ChebyshevEpi {
  const double *f, *u_prev;
  double *out;
  double c1, c2;
  __device__ void operator()(std::size_t s, double au, double d, double uc) const {
    const double up = u_prev ? u_prev[s] : 0.0;
    out[s] = uc + c1 * (uc - up) + c2 * (f[s] - au) / d;
  }
};

template <class Epi>
__global__ void column_kernel(Level L, const double *__restrict__ a,
                              const double *__restrict__ u, Epi epi) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= L.n[0] || j >= L.n[1]) return;
  const int k0 = static_cast<int>(blockIdx.z) * KC;
  const int k1 = min(k0 + KC, L.n[2]);
  const std::ptrdiff_t uy = L.n[0] + 2 * k::kVectorHalo,
                       uz = uy * (L.n[1] + 2 * k::kVectorHalo);
  const std::ptrdiff_t ay = L.n[0] + 2 * k::kCoefficientHalo,
                       az = ay * (L.n[1] + 2 * k::kCoefficientHalo);
  const double *pu = u + k::vindex(L, i, j, k0);
  const double *pa = a + k::index(L, k::kCoefficientHalo, i, j, k0);
  double win[3][3][3]; // [oz][oy][ox] at the current node
  double e[2][2][2];   // [sz][sy][sx]
#pragma unroll
  for (int oz = 0; oz < 2; ++oz)
#pragma unroll
    for (int oy = 0; oy < 3; ++oy)
#pragma unroll
      for (int ox = 0; ox < 3; ++ox)
        win[oz + 1][oy][ox] = pu[(ox - 1) + (oy - 1) * uy + (oz - 1) * uz];
#pragma unroll
  for (int sy = 0; sy < 2; ++sy)
#pragma unroll
    for (int sx = 0; sx < 2; ++sx) e[0][sy][sx] = pa[-sx - sy * ay - az];
  for (int k = k0; k < k1; ++k) {
#pragma unroll
    for (int oy = 0; oy < 3; ++oy)
#pragma unroll
      for (int ox = 0; ox < 3; ++ox) {
        win[0][oy][ox] = win[1][oy][ox];
        win[1][oy][ox] = win[2][oy][ox];
        win[2][oy][ox] = pu[(ox - 1) + (oy - 1) * uy + uz];
      }
#pragma unroll
    for (int sy = 0; sy < 2; ++sy)
#pragma unroll
      for (int sx = 0; sx < 2; ++sx) {
        e[1][sy][sx] = e[0][sy][sx];
        e[0][sy][sx] = pa[-sx - sy * ay];
      }
    // Element by element: element (i - s) couples node i to the offsets o with
    // o_d in {0, 1} for s_d = 0 and {-1, 0} for s_d = 1.
    double au = 0.0;
#pragma unroll
    for (int sz = 0; sz < 2; ++sz)
#pragma unroll
      for (int sy = 0; sy < 2; ++sy)
#pragma unroll
        for (int sx = 0; sx < 2; ++sx) {
          double t = 0.0;
#pragma unroll
          for (int cz = 0; cz < 2; ++cz)
#pragma unroll
            for (int cy = 0; cy < 2; ++cy)
#pragma unroll
              for (int cx = 0; cx < 2; ++cx) {
                const int ox = cx - sx + 1, oy = cy - sy + 1, oz = cz - sz + 1;
                t += L.ke[ox + 3 * oy + 9 * oz] * win[oz][oy][ox];
              }
          au += e[sz][sy][sx] * t;
        }
    if (L.sigma != 0.0) {
      // The assembled mass is V (1/6, 2/3, 1/6) per axis: apply it separably.
      double m = 0.0;
#pragma unroll
      for (int oz = 0; oz < 3; ++oz) {
        double p = 0.0;
#pragma unroll
        for (int oy = 0; oy < 3; ++oy) {
          const double r = (win[oz][oy][0] + win[oz][oy][2]) * (1.0 / 6.0) +
                           win[oz][oy][1] * (2.0 / 3.0);
          p += (oy == 1 ? 2.0 / 3.0 : 1.0 / 6.0) * r;
        }
        m += (oz == 1 ? 2.0 / 3.0 : 1.0 / 6.0) * p;
      }
      au += L.sigma * L.mass[13] * (27.0 / 8.0) * m;
    }
    const double esum = e[0][0][0] + e[0][0][1] + e[0][1][0] + e[0][1][1] +
                        e[1][0][0] + e[1][0][1] + e[1][1][0] + e[1][1][1];
    epi(k::vindex(L, i, j, k), au, L.ke[13] * esum + L.sigma * L.mass[13],
        win[1][1][1]);
    pu += uz;
    pa += az;
  }
}

template <class Epi>
void columns(const Level &L, const double *a, const double *u, Epi epi) {
  const dim3 block(64, 4, 1);
  const dim3 grid(static_cast<unsigned>((L.n[0] + 63) / 64),
                  static_cast<unsigned>((L.n[1] + 3) / 4),
                  static_cast<unsigned>((L.n[2] + KC - 1) / KC));
  column_kernel<<<grid, block>>>(L, a, u, epi);
}

Region half(const Level &F) { return Region{{F.n[0] / 2, F.n[1] / 2, F.n[2] / 2}}; }

double *dot_scratch() {
  static double *p = nullptr;
  if (!p) {
    void *v = nullptr;
    if (::pfc::gpuMalloc(&v, kDotBlocks * sizeof(double)) != ::pfc::gpuSuccess)
      throw std::runtime_error("NodalMultigrid: dot scratch allocation failed");
    p = static_cast<double *>(v);
  }
  return p;
}

} // namespace

#if defined(OPENPFC_GPU_API_CUDA)
#define OPENPFC_NMG_NAME(name) name##_cuda
#else
#define OPENPFC_NMG_NAME(name) name##_hip
#endif

void OPENPFC_NMG_NAME(fill)(double *p, std::size_t n, double v) {
  const std::size_t b = (n + kThreads - 1) / kThreads;
  fill_kernel<<<static_cast<int>(b < 65535u * 16u ? (b ? b : 1) : 65535u * 16u),
                kThreads>>>(p, n, v);
  check("fill");
}
void OPENPFC_NMG_NAME(axpy_owned)(const Level &L, double a, const double *x,
                                  double *y) {
  axpy_owned_kernel<<<grid3(L.n), block3()>>>(L, a, x, y);
  check("axpy_owned");
}
void OPENPFC_NMG_NAME(add_owned)(const Level &L, double *y, double v) {
  add_owned_kernel<<<grid3(L.n), block3()>>>(L, y, v);
  check("add_owned");
}
double OPENPFC_NMG_NAME(dot_owned)(const Level &L, const double *x,
                                   const double *y) {
  double *s = dot_scratch();
  dot_kernel<<<kDotBlocks, kThreads>>>(L, x, y, s);
  check("dot");
  sum_kernel<<<1, kThreads>>>(s, kDotBlocks);
  check("dot sum");
  double r = 0.0;
  if (::pfc::gpuMemcpy(&r, s, sizeof(double), ::pfc::gpuMemcpyDeviceToHost) !=
      ::pfc::gpuSuccess)
    throw std::runtime_error("NodalMultigrid dot: device to host copy failed");
  return r;
}
void OPENPFC_NMG_NAME(hash_owned)(const Level &L, double *x) {
  hash_owned_kernel<<<grid3(L.n), block3()>>>(L, x);
  check("hash_owned");
}
void OPENPFC_NMG_NAME(copy_owned)(const Level &L, const double *src, int src_halo,
                                  double *dst, int dst_halo) {
  copy_owned_kernel<<<grid3(L.n), block3()>>>(L, src, src_halo, dst, dst_halo);
  check("copy_owned");
}
void OPENPFC_NMG_NAME(apply)(const Level &L, const Operator &A, const double *u,
                             double *out) {
  if (!L.stencil)
    columns(L, A.a, u, ApplyEpi{out});
  else
    apply_kernel<<<grid3(L.n), block3()>>>(L, A, u, out);
  check("apply");
}
void OPENPFC_NMG_NAME(dirichlet_apply)(const Level &L,
                                       const k::dirichlet::Weights &w,
                                       const double *u, double *out) {
  dirichlet_apply_kernel<<<grid3(L.n), block3()>>>(L, w, u, out);
  check("dirichlet_apply");
}
void OPENPFC_NMG_NAME(dirichlet_jacobi)(const Level &L,
                                        const k::dirichlet::Weights &w,
                                        const double *old_u, const double *rhs,
                                        double *new_u, double scale) {
  dirichlet_jacobi_kernel<<<grid3(L.n), block3()>>>(L, w, old_u, rhs, new_u, scale);
  check("dirichlet_jacobi");
}
void OPENPFC_NMG_NAME(dirichlet_restrict)(const Level &F, const double *fine,
                                          const Level &C, double *coarse) {
  dirichlet_restrict_kernel<<<grid3(C.n), block3()>>>(F, fine, C, coarse);
  check("dirichlet_restrict");
}
void OPENPFC_NMG_NAME(dirichlet_prolong)(const Level &F, double *fine,
                                         const Level &C, const double *coarse) {
  dirichlet_prolong_kernel<<<grid3(F.n), block3()>>>(F, fine, C, coarse);
  check("dirichlet_prolong");
}
void OPENPFC_NMG_NAME(scaled_apply)(const Level &L, const Operator &A,
                                    const double *u, double *out) {
  if (!L.stencil)
    columns(L, A.a, u, ScaledEpi{out});
  else
    scaled_apply_kernel<<<grid3(L.n), block3()>>>(L, A, u, out);
  check("scaled_apply");
}
void OPENPFC_NMG_NAME(residual)(const Level &L, const Operator &A, const double *f,
                                const double *u, double *r, int zero_u) {
  if (!L.stencil && !zero_u)
    columns(L, A.a, u, ResidualEpi{f, r});
  else
    residual_kernel<<<grid3(L.n), block3()>>>(L, A, f, u, r, zero_u);
  check("residual");
}
void OPENPFC_NMG_NAME(chebyshev)(const Level &L, const Operator &A, const double *f,
                                 const double *u, const double *u_prev, double *out,
                                 double c1, double c2, int zero) {
  if (!L.stencil && !zero)
    columns(L, A.a, u, ChebyshevEpi{f, u_prev, out, c1, c2});
  else
    chebyshev_kernel<<<grid3(L.n), block3()>>>(L, A, f, u, u_prev, out, c1, c2,
                                               zero);
  check("chebyshev");
}
void OPENPFC_NMG_NAME(gauss_seidel)(const Level &L, const Operator &A,
                                    const double *f, double *u, int colour) {
  gauss_seidel_kernel<<<grid3(L.n), block3()>>>(L, A, f, u, colour);
  check("gauss_seidel");
}
void OPENPFC_NMG_NAME(restrict_residual)(const Level &F, const double *rf,
                                         const Level &G, double *fg,
                                         const int off[3]) {
  const Region R = half(F);
  restrict_residual_kernel<<<grid3(R.n), block3()>>>(F, rf, G, fg, R, off[0], off[1],
                                                     off[2]);
  check("restrict_residual");
}
void OPENPFC_NMG_NAME(prolong)(const Level &F, const double *uf, double *out,
                               const Level &G, const double *ug, const int off[3]) {
  prolong_kernel<<<grid3(F.n), block3()>>>(F, uf, out, G, ug, off[0], off[1],
                                           off[2]);
  check("prolong");
}
void OPENPFC_NMG_NAME(restrict_coefficient)(const Level &F, const double *af,
                                            const Level &G, double *ag,
                                            const int off[3]) {
  const Region R = half(F);
  restrict_coefficient_kernel<<<grid3(R.n), block3()>>>(F, af, G, ag, R, off[0],
                                                        off[1], off[2]);
  check("restrict_coefficient");
}
void OPENPFC_NMG_NAME(galerkin)(const Level &F, const Operator &Af, const Level &G,
                                double *sg, std::size_t gplane, const int off[3]) {
  const Region R = half(F);
  galerkin_kernel<<<grid3(R.n), block3()>>>(F, Af, G, sg, gplane, R, off[0], off[1],
                                            off[2]);
  check("galerkin");
}
#undef OPENPFC_NMG_NAME

} // namespace pfc::solvers::gpu::nmg
