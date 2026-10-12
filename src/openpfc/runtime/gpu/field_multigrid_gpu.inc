// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file field_multigrid_gpu.inc
 * @brief Device kernels of `pfc::solvers::FieldMultigrid` (included from .cu/.hip).
 *
 * Every kernel is a grid-stride loop over the owned cells of one level that
 * calls the point function of field_multigrid_kernels.hpp shared with the host.
 */

#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/field_multigrid_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::solvers::gpu::fmg {
namespace {

namespace k = ::pfc::solvers::fmg;

constexpr int kThreads = 256;
inline int blocks_for(std::size_t n) {
  const std::size_t b = (n + kThreads - 1) / kThreads;
  return static_cast<int>(b < 65535u * 16u ? (b ? b : 1) : 65535u * 16u);
}
inline void check(const char *what) {
  ::pfc::gpuError_t e = ::pfc::gpuGetLastError();
  if (e != ::pfc::gpuSuccess)
    throw std::runtime_error(std::string("FieldMultigrid ") + what + ": " +
                             ::pfc::gpuGetErrorString(e));
}

#define OPENPFC_FMG_LOOP(n)                                                            \
  for (std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < (n);  \
       t += std::size_t(gridDim.x) * blockDim.x)

__global__ void fill_kernel(double *p, std::size_t n, double v) {
  OPENPFC_FMG_LOOP(n) p[t] = v;
}
__global__ void fill_owned_kernel(Level L, double *p, double v) {
  OPENPFC_FMG_LOOP(k::owned_count(L)) {
    int i, j, kk;
    k::owned_ijk(L, t, i, j, kk);
    p[k::index(L, i, j, kk)] = v;
  }
}
__global__ void copy_owned_kernel(Level L, const double *s, double *d) {
  OPENPFC_FMG_LOOP(k::owned_count(L)) {
    int i, j, kk;
    k::owned_ijk(L, t, i, j, kk);
    k::store(L, d, i, j, kk, s[k::index(L, i, j, kk)]);
  }
}
__global__ void apply_kernel(Level L, Coefficients C, const double *u, double *out) {
  OPENPFC_FMG_LOOP(k::owned_count(L)) {
    int i, j, kk;
    k::owned_ijk(L, t, i, j, kk);
    out[k::index(L, i, j, kk)] = k::apply(L, C, u, i, j, kk);
  }
}
__global__ void smooth_kernel(Level L, Coefficients C, const double *f, double *u,
                              int colour) {
  OPENPFC_FMG_LOOP(k::owned_count(L)) {
    int i, j, kk;
    k::owned_ijk(L, t, i, j, kk);
    k::smooth(L, C, f, u, i, j, kk, colour);
  }
}
__global__ void residual_kernel(Level L, Coefficients C, const double *f, const double *u,
                                double *r) {
  OPENPFC_FMG_LOOP(k::owned_count(L)) {
    int i, j, kk;
    k::owned_ijk(L, t, i, j, kk);
    k::residual(L, C, f, u, r, i, j, kk);
  }
}
__global__ void restrict_coefficients_kernel(Level F, Coefficients Cf, Level G, double *c,
                                             double *dx, double *dy, double *dz,
                                             double *m) {
  OPENPFC_FMG_LOOP(k::owned_count(G)) {
    int i, j, kk;
    k::owned_ijk(G, t, i, j, kk);
    k::restrict_coefficients(F, Cf, G, c, dx, dy, dz, m, i, j, kk);
  }
}
__global__ void restrict_residual_kernel(Level F, const double *rf, const double *mf,
                                         Level G, const double *mg, double *fg) {
  OPENPFC_FMG_LOOP(k::owned_count(G)) {
    int i, j, kk;
    k::owned_ijk(G, t, i, j, kk);
    k::restrict_residual(F, rf, mf, G, mg, fg, i, j, kk);
  }
}
__global__ void prolong_add_kernel(Level F, const double *mf, double *uf, Level G,
                                   const double *ug) {
  OPENPFC_FMG_LOOP(k::owned_count(F)) {
    int i, j, kk;
    k::owned_ijk(F, t, i, j, kk);
    k::prolong_add(F, mf, uf, G, ug, i, j, kk);
  }
}
#undef OPENPFC_FMG_LOOP

} // namespace

#if defined(OPENPFC_GPU_API_CUDA)
#define OPENPFC_FMG_NAME(name) name##_cuda
#else
#define OPENPFC_FMG_NAME(name) name##_hip
#endif

void OPENPFC_FMG_NAME(fill)(double *p, std::size_t n, double v) {
  fill_kernel<<<blocks_for(n), kThreads>>>(p, n, v);
  check("fill");
}
void OPENPFC_FMG_NAME(fill_owned)(const Level &L, double *p, double v) {
  fill_owned_kernel<<<blocks_for(k::owned_count(L)), kThreads>>>(L, p, v);
  check("fill_owned");
}
void OPENPFC_FMG_NAME(copy_owned)(const Level &L, const double *s, double *d) {
  copy_owned_kernel<<<blocks_for(k::owned_count(L)), kThreads>>>(L, s, d);
  check("copy_owned");
}
void OPENPFC_FMG_NAME(apply)(const Level &L, const Coefficients &C, const double *u,
                             double *out) {
  apply_kernel<<<blocks_for(k::owned_count(L)), kThreads>>>(L, C, u, out);
  check("apply");
}
void OPENPFC_FMG_NAME(smooth)(const Level &L, const Coefficients &C, const double *f,
                              double *u, int colour) {
  smooth_kernel<<<blocks_for(k::owned_count(L)), kThreads>>>(L, C, f, u, colour);
  check("smooth");
}
void OPENPFC_FMG_NAME(residual)(const Level &L, const Coefficients &C, const double *f,
                                const double *u, double *r) {
  residual_kernel<<<blocks_for(k::owned_count(L)), kThreads>>>(L, C, f, u, r);
  check("residual");
}
void OPENPFC_FMG_NAME(restrict_coefficients)(const Level &F, const Coefficients &Cf,
                                             const Level &G, double *c, double *dx,
                                             double *dy, double *dz, double *m) {
  restrict_coefficients_kernel<<<blocks_for(k::owned_count(G)), kThreads>>>(
      F, Cf, G, c, dx, dy, dz, m);
  check("restrict_coefficients");
}
void OPENPFC_FMG_NAME(restrict_residual)(const Level &F, const double *rf,
                                         const double *mf, const Level &G,
                                         const double *mg, double *fg) {
  restrict_residual_kernel<<<blocks_for(k::owned_count(G)), kThreads>>>(F, rf, mf, G, mg,
                                                                         fg);
  check("restrict_residual");
}
void OPENPFC_FMG_NAME(prolong_add)(const Level &F, const double *mf, double *uf,
                                   const Level &G, const double *ug) {
  prolong_add_kernel<<<blocks_for(k::owned_count(F)), kThreads>>>(F, mf, uf, G, ug);
  check("prolong_add");
}
#undef OPENPFC_FMG_NAME

} // namespace pfc::solvers::gpu::fmg
