// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file incompressible_device.hpp
 * @brief Device IFRK4 for the periodic rotational Navier–Stokes term.
 *
 * The step keeps the velocity, the Runge–Kutta stages, and the real-space
 * products in device memory. HeFFTe's device FFT is the only transform.
 * Exponentials are uploaded once, before the first step. Diagnostics copy
 * the hats back at a sample, not inside a stage.
 *
 * Runtime owns this header. It names device buffers, so kernel code must
 * not include it. The names are `pfc::incompressible`, the same
 * namespace as the host step. This directory is the device realization.
 * The declarations are compiled only when
 * OpenPFC_ENABLE_CUDA_SPECTRAL or OpenPFC_ENABLE_HIP_SPECTRAL is set.
 * Both the .cu and the .hip unit include one implementation header, and
 * that header requires __CUDACC__ or __HIPCC__. One spectral backend per
 * build: the kernels are not split by those macros, and hipcc may define
 * __CUDACC__ together with __HIPCC__, so enabling both
 * OpenPFC_ENABLE_CUDA_SPECTRAL and OpenPFC_ENABLE_HIP_SPECTRAL would
 * define the same device symbols in one link. Explicit instantiations
 * stay `#if defined(__HIPCC__)` then `#elif defined(__CUDACC__)`.
 */

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::incompressible {

/// Local outbox geometry. Wavenumbers use the global grid.
struct HatGeom {
  int low0{0};
  int low1{0};
  int low2{0};
  int n0{0};
  int n1{0};
  int n2{0};
  int g0{0};
  int g1{0};
  int g2{0};
  double fx{0.0};
  double fy{0.0};
  double fz{0.0};
  double cx{0.0};
  double cy{0.0};
  double cz{0.0};
};

template <class Space> struct DeviceVelocity {
  using FFT = fft::IDeviceFFT<Space>;
  using Real = typename FFT::RealBuffer;
  using Cplx = typename FFT::ComplexBuffer;

  HatGeom geom{};
  double dt{0.0};
  std::size_t n_hat{0};
  std::size_t n_real{0};
  Cplx u, v, w;
  Real exp_dt, exp_half;
  Cplx su, sv, sw;
  Cplx n1u, n1v, n1w;
  Cplx n2u, n2v, n2w;
  Cplx n3u, n3v, n3w;
  Cplx n4u, n4v, n4w;
  Cplx uh, vh, wh;
  Cplx ox, oy, oz;
  Real ur, vr, wr;
  Real oxr, oyr, ozr;
  Real tx, ty, tz;
};

template <class Space>
void prepare_device_velocity(DeviceVelocity<Space> &vel,
                             const fft::IDeviceFFT<Space> &fft, std::array<int, 3> n,
                             std::array<double, 3> spacing, double nu, double dt);

template <class Space>
void upload_device_velocity(DeviceVelocity<Space> &vel,
                            const std::vector<std::complex<double>> &u,
                            const std::vector<std::complex<double>> &v,
                            const std::vector<std::complex<double>> &w);

template <class Space>
void download_device_velocity(const DeviceVelocity<Space> &vel,
                              std::vector<std::complex<double>> &u,
                              std::vector<std::complex<double>> &v,
                              std::vector<std::complex<double>> &w);

/// Seconds inside one profiled step. A null profile does not synchronize.
struct StepProfile {
  double spectral_s{0.0};
  double fft_s{0.0};
  double nonlinear_s{0.0};
};

/// One integrating-factor RK4 step. `dealias` keeps the 2/3 mask.
/// A non-null `profile` inserts a device sync between spectral work, FFTs,
/// and the real-space product so those three can be timed.
template <class Space>
void step_device_velocity(DeviceVelocity<Space> &vel, fft::IDeviceFFT<Space> &fft,
                          bool dealias, StepProfile *profile = nullptr);

/// Bytes resident for one rank, including scratch used inside the step.
template <class Space>
[[nodiscard]] std::size_t device_velocity_bytes(const DeviceVelocity<Space> &vel);

} // namespace pfc::incompressible

#endif
