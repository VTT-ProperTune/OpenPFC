// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_hip_kernels.hip
 * @brief HIP kernels for equations (1)-(4); line-by-line twin of `step.hpp`.
 *
 * @details
 * ## The contract this file is written to
 *
 * Every expression below is transcribed from the corresponding stage of
 * `alloy_dendrite::Stepper`, in the same order, with the same intermediate
 * variables and the same parenthesisation. That is not stylistic tidiness: a
 * parity test that compares the two paths is only a test of the *port* if the
 * two paths are meant to produce the same floating-point result. Reassociate
 * `w2 * gx + flux` into `fma(w2, gx, flux)` and the difference is a few ULP
 * and harmless; reassociate `phi - phi^3` into `phi * (1 - phi^2)` and the
 * difference is still a few ULP but the test no longer distinguishes a
 * rounding difference from a transcription error. So: same algebra, and the
 * residual is measured rather than assumed.
 *
 * ## Templated on the dimension, like the host
 *
 * `Dim` is a template parameter here for the same reason it is one on
 * `Stepper`: a 2-D slab must not read a `z` halo that was never exchanged
 * (`Axes2D()` skips the `±z` faces). Making the dimension a template
 * parameter means the `z` loads do not exist in the 2-D kernel at all, rather
 * than existing and being multiplied by zero.
 *
 * ## Why the launchers take `int dim` and dispatch
 *
 * The header cannot be a template (it must be includable by a translation
 * unit that is not compiled by `hipcc`), so the two instantiations are
 * selected here. The cost is one predictable branch per kernel launch.
 *
 * @see device_step_hip.hpp -- the descriptors and the launch surface
 * @see ../../include/alloy_dendrite/step.hpp -- the host twin
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "alloy_dendrite_hip_kernels requires HIP (-DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <alloy_dendrite/device_step_hip.hpp>

namespace alloy_dendrite::hip {
namespace {

/// Thread-block shape. 64 threads along x is one AMD wavefront, so a row of
/// the padded brick is loaded by one wavefront and the stencil's `±k` loads
/// stay inside the same cache lines.
constexpr int kBlockX = 64;
constexpr int kBlockY = 4;

inline void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "alloy_dendrite HIP error in %s: %s\n", what,
                 hipGetErrorString(e));
    std::abort();
  }
}

/// Linear index of owned cell `(i, j, k)` in the padded brick.
__device__ inline long long lin(const DeviceGeom &g, int i, int j, int k) noexcept {
  return static_cast<long long>(i + g.hw) +
         static_cast<long long>(j + g.hw) * g.sy +
         static_cast<long long>(k + g.hw) * g.sz;
}

/// Central D1 along a stride, integer weights summed then scaled -- the exact
/// shape of `pfc::field::fd::apply_d1_along` followed by `FDGradient`'s
/// `m_s?1 *`.
__device__ inline double d1(const DeviceStencil &s, const double *u, long long c,
                            long long stride, double scale) noexcept {
  double acc = 0.0;
  for (int k = 1; k <= s.hw1; ++k) {
    const long long ks = static_cast<long long>(k) * stride;
    acc += s.c1[k] * (u[c + ks] - u[c - ks]);
  }
  return scale * acc;
}

/// Central D2 along a stride; twin of `apply_d2_along` plus `m_s?2 *`.
__device__ inline double d2(const DeviceStencil &s, const double *u, long long c,
                            long long stride, double scale) noexcept {
  double acc = s.c2[0] * u[c];
  for (int k = 1; k <= s.hw2; ++k) {
    const long long ks = static_cast<long long>(k) * stride;
    acc += s.c2[k] * (u[c - ks] + u[c + ks]);
  }
  return scale * acc;
}

/// `P(phi) = ((1+k) - (1-k) phi) / 2`; twin of `solute_prefactor`.
__device__ inline double prefactor(double k, double phi) noexcept {
  return 0.5 * ((1.0 + k) - (1.0 - k) * phi);
}

/// `q(phi) = (1 - phi) / 2`; twin of `solute_mobility`.
__device__ inline double mobility(double phi) noexcept { return 0.5 * (1.0 - phi); }

/// Equation (1) evaluated from a raw gradient; twin of `evaluate_anisotropy`.
template <int Dim> struct Anisotropy {
  double a_s{1.0};
  double W{1.0};
  double tau{1.0};
  double fx{0.0}, fy{0.0}, fz{0.0};
};

template <int Dim>
__device__ inline Anisotropy<Dim> anisotropy(const DeviceParams &p, double gx,
                                             double gy, double gz) noexcept {
  Anisotropy<Dim> out;
  out.W = p.W0;
  out.tau = p.tau0;
  if (p.eps4 == 0.0) {
    return out;
  }
  const double g2 = gx * gx + gy * gy + (Dim == 3 ? gz * gz : 0.0);
  if (g2 < p.grad_floor2) {
    return out;
  }
  const double g4sum =
      gx * gx * gx * gx + gy * gy * gy * gy + (Dim == 3 ? gz * gz * gz * gz : 0.0);
  const double inv_g2 = 1.0 / g2;
  const double inv_g4 = inv_g2 * inv_g2;
  const double s = g4sum * inv_g4;
  out.a_s = p.aniso_amp * (1.0 + p.aniso_slope * s);
  out.W = p.W0 * out.a_s;
  out.tau = p.tau0 * out.a_s * out.a_s;
  const double pre = 4.0 * p.aniso_ampslope * p.W0 * p.W0 * out.a_s * inv_g4;
  out.fx = pre * (gx * gx * gx * g2 - g4sum * gx);
  out.fy = pre * (gy * gy * gy * g2 - g4sum * gy);
  if constexpr (Dim == 3) {
    out.fz = pre * (gz * gz * gz * g2 - g4sum * gz);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <int Dim>
__global__ void seed_psi_kernel(DeviceFields f, DeviceGeom g, double k) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nx || j >= g.ny) return;
  const int kz_hi = (Dim == 3) ? g.nz : 1;
  for (int kz = 0; kz < kz_hi; ++kz) {
    const long long c = lin(g, i, j, kz);
    f.psi[c] = prefactor(k, f.phi[c]) * f.U[c];
  }
}

template <int Dim>
__global__ void stage_a_kernel(DeviceFields f, DeviceGeom g, DeviceStencil s,
                               DeviceParams p) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nx || j >= g.ny) return;
  const int kz_hi = (Dim == 3) ? g.nz : 1;
  for (int kz = 0; kz < kz_hi; ++kz) {
    const long long c = lin(g, i, j, kz);

    const double gx = d1(s, f.phi, c, 1, s.s1x);
    const double gy = d1(s, f.phi, c, g.sy, s.s1y);
    const double gz = (Dim == 3) ? d1(s, f.phi, c, g.sz, s.s1z) : 0.0;
    f.gx[c] = gx;
    f.gy[c] = gy;
    if constexpr (Dim == 3) {
      f.gz[c] = gz;
    }

    const Anisotropy<Dim> a = anisotropy<Dim>(p, gx, gy, gz);
    const double w2 = a.W * a.W;
    f.tau[c] = a.tau;
    f.Fx[c] = w2 * gx + a.fx;
    f.Fy[c] = w2 * gy + a.fy;
    if constexpr (Dim == 3) {
      f.Fz[c] = w2 * gz + a.fz;
    }

    if (p.thermal != 0) {
      double lap = d2(s, f.theta, c, 1, s.s2x) + d2(s, f.theta, c, g.sy, s.s2y);
      if constexpr (Dim == 3) {
        lap += d2(s, f.theta, c, g.sz, s.s2z);
      }
      f.lap_theta[c] = lap;
    } else {
      f.lap_theta[c] = 0.0;
    }
  }
}

template <int Dim>
__global__ void stage_b_kernel(DeviceFields f, DeviceGeom g, DeviceStencil s,
                               DeviceParams p) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nx || j >= g.ny) return;
  const int kz_hi = (Dim == 3) ? g.nz : 1;
  const double lam = p.lambda;
  const double mc = p.M_c;
  const double lam_el = p.lambda_el;
  const double *dfel = f.dfel;
  for (int kz = 0; kz < kz_hi; ++kz) {
    const long long c = lin(g, i, j, kz);
    double div = d1(s, f.Fx, c, 1, s.s1x) + d1(s, f.Fy, c, g.sy, s.s1y);
    if constexpr (Dim == 3) {
      div += d1(s, f.Fz, c, g.sz, s.s1z);
    }
    const double ph = f.phi[c];
    const double well = ph - ph * ph * ph;
    const double gwell = (1.0 - ph * ph) * (1.0 - ph * ph);
    double rhs = div + well - lam * gwell * (f.U[c] + mc * f.theta[c]);
    // ---- ELASTIC HOOK ---------------------------------------------------
    // The device twin of the hook in `Stepper::stage_b_`. `dfel` is a
    // device-resident field written by whoever solved equations (5)-(7);
    // `alloy_dendrite_coupled_cost.cpp` fills it by a host round-trip
    // through `openpfc_apps/microelasticity.hpp`, which is host-only.
    // Only owned cells are read, so the field needs no halo exchange.
    if (dfel != nullptr && lam_el != 0.0) {
      rhs -= lam_el * gwell * dfel[c];
    }
    // ---------------------------------------------------------------------
    f.dphidt[c] = rhs / f.tau[c];
  }
}

template <int Dim>
__global__ void stage_c_kernel(DeviceFields f, DeviceGeom g, DeviceStencil s,
                               DeviceParams p) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nx || j >= g.ny) return;
  const int kz_hi = (Dim == 3) ? g.nz : 1;
  const double k = p.k;
  const double at = p.at;
  for (int kz = 0; kz < kz_hi; ++kz) {
    const long long c = lin(g, i, j, kz);
    const double gux = d1(s, f.U, c, 1, s.s1x);
    const double guy = d1(s, f.U, c, g.sy, s.s1y);
    const double guz = (Dim == 3) ? d1(s, f.U, c, g.sz, s.s1z) : 0.0;
    const double q = p.D_l * mobility(f.phi[c]);
    const double gx = f.gx[c];
    const double gy = f.gy[c];
    const double gz = (Dim == 3) ? f.gz[c] : 0.0;
    const double g2 = gx * gx + gy * gy + (Dim == 3 ? gz * gz : 0.0);
    const double inv_gn = (g2 > p.grad_floor2) ? 1.0 / sqrt(g2) : 0.0;
    const double amp = at * (1.0 + (1.0 - k) * f.U[c]) * f.dphidt[c] * inv_gn;
    f.Jx[c] = q * gux + amp * gx;
    f.Jy[c] = q * guy + amp * gy;
    if constexpr (Dim == 3) {
      f.Jz[c] = q * guz + amp * gz;
    }
  }
}

template <int Dim>
__global__ void stage_d_kernel(DeviceFields f, DeviceGeom g, DeviceStencil s,
                               DeviceParams p, double dt) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nx || j >= g.ny) return;
  const int kz_hi = (Dim == 3) ? g.nz : 1;
  const double k = p.k;
  const double dth = p.D_th;
  for (int kz = 0; kz < kz_hi; ++kz) {
    const long long c = lin(g, i, j, kz);
    double divJ = d1(s, f.Jx, c, 1, s.s1x) + d1(s, f.Jy, c, g.sy, s.s1y);
    if constexpr (Dim == 3) {
      divJ += d1(s, f.Jz, c, g.sz, s.s1z);
    }
    const double rate = f.dphidt[c];
    const double src = (p.spec_source != 0)
                           ? 0.5 * (1.0 + (1.0 - k) * f.U[c]) * rate
                           : 0.5 * rate;
    const double psi_new = f.psi[c] + dt * (divJ + src);
    const double phi_new = f.phi[c] + dt * rate;
    if (p.evolve_theta != 0) {
      f.theta[c] += dt * (dth * f.lap_theta[c] + 0.5 * rate);
    }
    f.psi[c] = psi_new;
    f.phi[c] = phi_new;
    f.U[c] = psi_new / prefactor(k, phi_new);
  }
}

/// Launch geometry: one thread per `(i, j)` column, the `z` extent walked
/// inside the kernel. A column loop keeps the `z` stencil loads in registers
/// across `k` for the 3-D case and costs nothing in 2-D, where the loop runs
/// once.
inline dim3 grid_for(const DeviceGeom &g) {
  return dim3(static_cast<unsigned>((g.nx + kBlockX - 1) / kBlockX),
              static_cast<unsigned>((g.ny + kBlockY - 1) / kBlockY), 1u);
}

} // namespace

void alloy_seed_psi_hip(const DeviceFields &f, const DeviceGeom &g, double k,
                        int dim) {
  const dim3 block(kBlockX, kBlockY, 1);
  const dim3 grid = grid_for(g);
  if (dim == 3) {
    seed_psi_kernel<3><<<grid, block>>>(f, g, k);
  } else {
    seed_psi_kernel<2><<<grid, block>>>(f, g, k);
  }
  check(hipGetLastError(), "alloy_seed_psi_hip");
}

void alloy_stage_a_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, int dim) {
  const dim3 block(kBlockX, kBlockY, 1);
  const dim3 grid = grid_for(g);
  if (dim == 3) {
    stage_a_kernel<3><<<grid, block>>>(f, g, s, p);
  } else {
    stage_a_kernel<2><<<grid, block>>>(f, g, s, p);
  }
  check(hipGetLastError(), "alloy_stage_a_hip");
}

void alloy_stage_b_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, int dim) {
  const dim3 block(kBlockX, kBlockY, 1);
  const dim3 grid = grid_for(g);
  if (dim == 3) {
    stage_b_kernel<3><<<grid, block>>>(f, g, s, p);
  } else {
    stage_b_kernel<2><<<grid, block>>>(f, g, s, p);
  }
  check(hipGetLastError(), "alloy_stage_b_hip");
}

void alloy_stage_c_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, int dim) {
  const dim3 block(kBlockX, kBlockY, 1);
  const dim3 grid = grid_for(g);
  if (dim == 3) {
    stage_c_kernel<3><<<grid, block>>>(f, g, s, p);
  } else {
    stage_c_kernel<2><<<grid, block>>>(f, g, s, p);
  }
  check(hipGetLastError(), "alloy_stage_c_hip");
}

void alloy_stage_d_hip(const DeviceFields &f, const DeviceGeom &g,
                       const DeviceStencil &s, const DeviceParams &p, double dt,
                       int dim) {
  const dim3 block(kBlockX, kBlockY, 1);
  const dim3 grid = grid_for(g);
  if (dim == 3) {
    stage_d_kernel<3><<<grid, block>>>(f, g, s, p, dt);
  } else {
    stage_d_kernel<2><<<grid, block>>>(f, g, s, p, dt);
  }
  check(hipGetLastError(), "alloy_stage_d_hip");
}

} // namespace alloy_dendrite::hip
