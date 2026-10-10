// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fft_single_device.hpp
 * @brief One-rank device FFT: a single vendor 3D r2c/c2r plan as an
 *        `IDeviceFFT<MemorySpace>` (cuFFT for CUDA, rocFFT for HIP).
 *
 * @details
 * On one rank heFFTe has nothing to redistribute, but its device backends
 * still split the 3D transform: with the default options strided dimensions
 * get one vendor call per plane (about 150 kernel launches per spectral ETD
 * step on a 43 x 819 x 16 box), with `use_reorder` most of the GPU time goes
 * to transpose kernels. One vendor 3D plan does the transform in a few
 * kernels (issue #382: 0.75 -> 0.22 ms per step on an H100).
 *
 * Semantics match `FFT_Impl` with the heFFTe backends for one rank:
 * - real inbox = the whole domain [0, nx) x [0, ny) x [0, nz),
 * - complex outbox = [0, nx/2] x [0, ny) x [0, nz) (r2c along x),
 * - both stored with x fastest,
 * - forward unscaled, backward scaled by 1/(nx ny nz),
 * - the backward input is preserved (copied to a workspace first: out-of-
 *   place multi-dimensional C2R may overwrite its input).
 * All work is issued on the legacy default stream, like the heFFTe path.
 *
 * Use through `GPUSpectralStack` with `DeviceFFTChoice::single_device`, or
 * directly. Only valid for one rank (the caller's decomposition must cover
 * the whole domain).
 */

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
#include <cuda_runtime_api.h>
#include <cufft.h>
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <hip/hip_runtime_api.h>
#include <rocfft/rocfft.h>
#endif

namespace pfc::fft {

namespace detail {

template <class MemorySpace> struct single_device_vendor;

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
template <> struct single_device_vendor<CUDASpace> {
  struct plans {
    cufftHandle fwd = 0, bwd = 0;
  };
  static void check(cufftResult r, const char *what) {
    if (r != CUFFT_SUCCESS)
      throw std::runtime_error(std::string("SingleDeviceFFT<CUDA>: ") + what + " failed, cufftResult " +
                               std::to_string(static_cast<int>(r)));
  }
  static plans create(const std::array<int, 3> &n) {
    plans p;
    check(cufftPlan3d(&p.fwd, n[2], n[1], n[0], CUFFT_D2Z), "cufftPlan3d D2Z");
    check(cufftPlan3d(&p.bwd, n[2], n[1], n[0], CUFFT_Z2D), "cufftPlan3d Z2D");
    return p;
  }
  static void destroy(plans &p) noexcept {
    if (p.fwd) cufftDestroy(p.fwd);
    if (p.bwd) cufftDestroy(p.bwd);
    p = {};
  }
  static void forward(plans &p, const double *in, std::complex<double> *out) {
    check(cufftExecD2Z(p.fwd, const_cast<double *>(in), reinterpret_cast<cufftDoubleComplex *>(out)),
          "cufftExecD2Z");
  }
  static void backward(plans &p, std::complex<double> *in, double *out) {
    check(cufftExecZ2D(p.bwd, reinterpret_cast<cufftDoubleComplex *>(in), out), "cufftExecZ2D");
  }
  static void copy(void *dst, const void *src, std::size_t bytes) {
    if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, nullptr) != cudaSuccess)
      throw std::runtime_error("SingleDeviceFFT<CUDA>: cudaMemcpyAsync failed");
  }
  static void scale(double *x, double a, std::size_t n) { pfc::axpy_fill_cuda_impl(x, x, a, 0.0, n); }
};
#endif

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
template <> struct single_device_vendor<HIPSpace> {
  struct plans {
    rocfft_plan fwd = nullptr, bwd = nullptr;
    rocfft_execution_info info = nullptr;
    void *work = nullptr;
  };
  static void check(rocfft_status s, const char *what) {
    if (s != rocfft_status_success)
      throw std::runtime_error(std::string("SingleDeviceFFT<HIP>: ") + what + " failed, rocfft_status " +
                               std::to_string(static_cast<int>(s)));
  }
  static plans create(const std::array<int, 3> &n) {
    check(rocfft_setup(), "rocfft_setup");
    plans p;
    // rocFFT lengths are fastest first; default strides are contiguous.
    const std::size_t len[3] = {static_cast<std::size_t>(n[0]), static_cast<std::size_t>(n[1]),
                                static_cast<std::size_t>(n[2])};
    check(rocfft_plan_create(&p.fwd, rocfft_placement_notinplace, rocfft_transform_type_real_forward,
                             rocfft_precision_double, 3, len, 1, nullptr),
          "rocfft_plan_create (forward)");
    check(rocfft_plan_create(&p.bwd, rocfft_placement_notinplace, rocfft_transform_type_real_inverse,
                             rocfft_precision_double, 3, len, 1, nullptr),
          "rocfft_plan_create (inverse)");
    std::size_t wf = 0, wb = 0;
    check(rocfft_plan_get_work_buffer_size(p.fwd, &wf), "rocfft_plan_get_work_buffer_size");
    check(rocfft_plan_get_work_buffer_size(p.bwd, &wb), "rocfft_plan_get_work_buffer_size");
    check(rocfft_execution_info_create(&p.info), "rocfft_execution_info_create");
    const std::size_t w = wf > wb ? wf : wb;
    if (w > 0) {
      if (hipMalloc(&p.work, w) != hipSuccess) throw std::runtime_error("SingleDeviceFFT<HIP>: hipMalloc failed");
      check(rocfft_execution_info_set_work_buffer(p.info, p.work, w), "rocfft_execution_info_set_work_buffer");
    }
    return p;
  }
  static void destroy(plans &p) noexcept {
    if (p.fwd) rocfft_plan_destroy(p.fwd);
    if (p.bwd) rocfft_plan_destroy(p.bwd);
    if (p.info) rocfft_execution_info_destroy(p.info);
    if (p.work) (void)hipFree(p.work);
    p = {};
  }
  static void forward(plans &p, const double *in, std::complex<double> *out) {
    void *ib[1] = {const_cast<double *>(in)};
    void *ob[1] = {out};
    check(rocfft_execute(p.fwd, ib, ob, p.info), "rocfft_execute (forward)");
  }
  static void backward(plans &p, std::complex<double> *in, double *out) {
    void *ib[1] = {in};
    void *ob[1] = {out};
    check(rocfft_execute(p.bwd, ib, ob, p.info), "rocfft_execute (inverse)");
  }
  static void copy(void *dst, const void *src, std::size_t bytes) {
    if (hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, nullptr) != hipSuccess)
      throw std::runtime_error("SingleDeviceFFT<HIP>: hipMemcpyAsync failed");
  }
  static void scale(double *x, double a, std::size_t n) { pfc::axpy_fill_hip_impl(x, x, a, 0.0, n); }
};
#endif

} // namespace detail

/**
 * @brief One-rank device FFT through one vendor 3D plan per direction.
 * @tparam MemorySpace `CUDASpace` or `HIPSpace`.
 */
template <class MemorySpace> class SingleDeviceFFT final : public IDeviceFFT<MemorySpace> {
  using vendor = detail::single_device_vendor<MemorySpace>;

public:
  using typename IDeviceFFT<MemorySpace>::RealBuffer;
  using typename IDeviceFFT<MemorySpace>::ComplexBuffer;

  /// @param size global grid size {nx, ny, nz} (x fastest), each >= 1.
  explicit SingleDeviceFFT(const std::array<int, 3> &size) : m_n(size) {
    for (int v : size)
      if (v < 1) throw std::invalid_argument("SingleDeviceFFT: grid sizes must be positive");
    m_plans = vendor::create(m_n);
    m_work = ComplexBuffer(size_outbox());
  }
  SingleDeviceFFT(const SingleDeviceFFT &) = delete;
  SingleDeviceFFT &operator=(const SingleDeviceFFT &) = delete;
  SingleDeviceFFT(SingleDeviceFFT &&) = delete;
  SingleDeviceFFT &operator=(SingleDeviceFFT &&) = delete;
  ~SingleDeviceFFT() override { vendor::destroy(m_plans); }

  void forward(const RealBuffer &in, ComplexBuffer &out) override {
    require(in.size() == size_inbox() && out.size() == size_outbox(), "forward");
    vendor::forward(m_plans, in.data(), out.data());
  }

  void backward(const ComplexBuffer &in, RealBuffer &out) override {
    require(in.size() == size_outbox() && out.size() == size_inbox(), "backward");
    vendor::copy(m_work.data(), in.data(), size_outbox() * sizeof(std::complex<double>));
    vendor::backward(m_plans, m_work.data(), out.data());
    vendor::scale(out.data(), 1.0 / static_cast<double>(size_inbox()), size_inbox());
  }

  void reset_fft_time() override {}
  [[nodiscard]] double get_fft_time() const override { return 0.0; }
  [[nodiscard]] std::size_t size_inbox() const override {
    return static_cast<std::size_t>(m_n[0]) * m_n[1] * m_n[2];
  }
  [[nodiscard]] std::size_t size_outbox() const override {
    return static_cast<std::size_t>(m_n[0] / 2 + 1) * m_n[1] * m_n[2];
  }
  [[nodiscard]] std::size_t size_workspace() const override { return size_outbox(); }
  [[nodiscard]] std::size_t get_allocated_memory_bytes() const override {
    return size_outbox() * sizeof(std::complex<double>);
  }
  [[nodiscard]] Box3i get_inbox_bounds() const override {
    return Box3i::from_bounds({0, 0, 0}, {m_n[0] - 1, m_n[1] - 1, m_n[2] - 1});
  }
  [[nodiscard]] Box3i get_outbox_bounds() const override {
    return Box3i::from_bounds({0, 0, 0}, {m_n[0] / 2, m_n[1] - 1, m_n[2] - 1});
  }

private:
  static void require(bool ok, const char *what) {
    if (!ok) throw std::invalid_argument(std::string("SingleDeviceFFT::") + what + ": buffer size mismatch");
  }
  std::array<int, 3> m_n;
  typename vendor::plans m_plans{};
  ComplexBuffer m_work;
};

} // namespace pfc::fft

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL || OpenPFC_ENABLE_HIP_SPECTRAL
