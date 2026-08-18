// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_heffte_backend.hpp
 * @brief HeFFTe-backed FFT_Impl template (include only where HeFFTe is required)
 *
 * @details
 * Workspace ownership is backend-specialized via `detail::FftWorkspaceStorage`:
 * - FFTW owns only the host `m_wrk` buffer used by the `std::vector` transform
 *   path.
 * - GPU backends (`cufft` / `rocfft`) lazily allocate the device workspace
 *   for each precision on first use (ADR 0006: only instantiate what runs).
 */

#pragma once

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/heap_concept.hpp>

#include <heffte.h>
#include <mpi.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(OpenPFC_ENABLE_CUDA)
static_assert(pfc::fft::HeapBackend<heffte::backend::cufft>,
              "cuFFT backend must satisfy HeapBackend concept");
#endif
#if defined(OpenPFC_ENABLE_HIP)
static_assert(pfc::fft::HeapBackend<heffte::backend::rocfft>,
              "rocFFT backend must satisfy HeapBackend concept");
#endif

namespace pfc {
namespace fft {
namespace detail {

/**
 * @brief Throw if a transform buffer extent does not match the FFT contract.
 * @param actual Caller buffer size
 * @param expected Required size (`size_inbox()` or `size_outbox()`)
 * @param context Message prefix naming the overload and buffer role, ending with
 *        `"size "` (e.g. `"FFT_Impl::forward: real buffer size "`)
 * @param expected_label Name of the expected extent accessor for the message
 * @throws std::invalid_argument when `actual != expected`
 */
inline void require_equal_size(std::size_t actual, std::size_t expected,
                               const char *context,
                               const char *expected_label) {
  if (actual != expected) [[unlikely]] {
    throw std::invalid_argument(std::string(context) + std::to_string(actual) +
                                " != " + expected_label + "() " +
                                std::to_string(expected));
  }
}

/**
 * @brief Backend-gated HeFFTe workspace ownership for `FFT_Impl`.
 *
 * Primary template is intentionally incomplete; specialize per backend family.
 */
template <typename BackendTag> struct FftWorkspaceStorage;

/**
 * @brief FFTW: host workspace only (`std::vector`-backed HeFFTe container).
 */
template <> struct FftWorkspaceStorage<heffte::backend::fftw> {
  using workspace_type = typename heffte::fft3d_r2c<
      heffte::backend::fftw>::template buffer_container<std::complex<double>>;

  workspace_type m_wrk;

  explicit FftWorkspaceStorage(std::size_t n) : m_wrk(n) {}

  auto *data_wrk() noexcept { return m_wrk.data(); }

  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return m_wrk.size() * sizeof(typename workspace_type::value_type);
  }
};

/**
 * @brief GPU backends: lazy per-precision device workspaces (no unused `m_wrk`).
 *
 * `FFT_Impl` is not templated on `RealType`, so float and double overloads
 * share one instance. Allocate a precision only when that overload first
 * runs (ADR 0006).
 */
template <typename BackendTag>
  requires HeapBackend<BackendTag>
struct FftWorkspaceStorage<BackendTag> {
  using gpu_workspace_type = typename heffte::fft3d_r2c<
      BackendTag>::template buffer_container<std::complex<double>>;
  using gpu_workspace_float = typename heffte::fft3d_r2c<
      BackendTag>::template buffer_container<std::complex<float>>;

  std::size_t m_n = 0;
  std::unique_ptr<gpu_workspace_type> m_gpu_wrk_double;
  std::unique_ptr<gpu_workspace_float> m_gpu_wrk_float;

  explicit FftWorkspaceStorage(std::size_t n) : m_n(n) {}

  auto *data_gpu_double() {
    if (!m_gpu_wrk_double) {
      m_gpu_wrk_double = std::make_unique<gpu_workspace_type>(m_n);
    }
    return m_gpu_wrk_double->data();
  }
  auto *data_gpu_float() {
    if (!m_gpu_wrk_float) {
      m_gpu_wrk_float = std::make_unique<gpu_workspace_float>(m_n);
    }
    return m_gpu_wrk_float->data();
  }

  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    std::size_t n = 0;
    if (m_gpu_wrk_double) {
      n += m_gpu_wrk_double->size() *
           sizeof(typename gpu_workspace_type::value_type);
    }
    if (m_gpu_wrk_float) {
      n += m_gpu_wrk_float->size() *
           sizeof(typename gpu_workspace_float::value_type);
    }
    return n;
  }
};

/**
 * @brief Detect `IDeviceFFT` buffer aliases so `FFT_Impl` can declare the
 *        override without naming a missing nested type on `IHostFFT`.
 */
template <typename I, typename = void> struct device_fft_buffers {
  struct unused_real;
  struct unused_complex;
  using RealBuffer = unused_real;
  using ComplexBuffer = unused_complex;
  static constexpr bool value = false;
};

template <typename I>
struct device_fft_buffers<
    I, std::void_t<typename I::RealBuffer, typename I::ComplexBuffer>> {
  using RealBuffer = typename I::RealBuffer;
  using ComplexBuffer = typename I::ComplexBuffer;
  static constexpr bool value = true;
};

} // namespace detail

/**
 * @brief FFT class template for distributed-memory parallel Fourier transforms
 *
 * @tparam BackendTag HeFFTe backend tag (heffte::backend::fftw or
 * heffte::backend::cufft / rocfft)
 *
 * Workspace buffers are owned by `detail::FftWorkspaceStorage<BackendTag>` so
 * unused twin host/device allocations are not constructed.
 */
template <typename BackendTag = heffte::backend::fftw,
          typename Interface = IHostFFT>
struct FFT_Impl : Interface {

  using fft_type = heffte::fft3d_r2c<BackendTag>;
  const fft_type m_fft;
  double m_fft_time = 0.0;

  detail::FftWorkspaceStorage<BackendTag> m_ws;

  FFT_Impl(fft_type fft)
      : m_fft(std::move(fft)), m_ws(m_fft.size_workspace()) {}

  /**
   * @brief Forward transform via `DataBuffer` (GPU backends, any RealType).
   * @throws std::invalid_argument if `in.size() != size_inbox()` or
   *         `out.size() != size_outbox()`
   */
  template <typename RealBackendTag, typename ComplexBackendTag, typename RealType>
    requires HeapBackend<BackendTag>
  void forward(const core::DataBuffer<RealBackendTag, RealType> &in,
               core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &out) {
    forward_device_(in, out);
  }

  /// `IDeviceFFT` double-buffer override (non-template wins over the template).
  /// Constraint is discarded on `IHostFFT`; `override` is omitted because
  /// Cray GNU rejects `override` plus a trailing `requires` on a
  /// non-template member.
  void forward(const typename detail::device_fft_buffers<Interface>::RealBuffer &in,
               typename detail::device_fft_buffers<Interface>::ComplexBuffer &out)
    requires detail::device_fft_buffers<Interface>::value
  {
    forward_device_(in, out);
  }

  /**
   * @brief Forward transform via host vectors (FFTW / `IHostFFT` only).
   * @throws std::invalid_argument if `in.size() != size_inbox()` or
   *         `out.size() != size_outbox()`
   */
  void forward(const RealVector &in, ComplexVector &out)
    requires std::is_base_of_v<IHostFFT, Interface>
  {
    detail::require_equal_size(
        in.size(), size_inbox(),
        "FFT_Impl::forward: real buffer size ", "size_inbox");
    detail::require_equal_size(
        out.size(), size_outbox(),
        "FFT_Impl::forward: complex buffer size ", "size_outbox");
    m_fft_time -= MPI_Wtime();
    m_fft.forward(in.data(), out.data(), m_ws.data_wrk());
    m_fft_time += MPI_Wtime();
  }

  /**
   * @brief Backward transform via `DataBuffer` (GPU backends, any RealType).
   * @throws std::invalid_argument if `in.size() != size_outbox()` or
   *         `out.size() != size_inbox()`
   */
  template <typename ComplexBackendTag, typename RealBackendTag, typename RealType>
    requires HeapBackend<BackendTag>
  void
  backward(const core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &in,
           core::DataBuffer<RealBackendTag, RealType> &out) {
    backward_device_(in, out);
  }

  /// `IDeviceFFT` double-buffer override.
  void
  backward(const typename detail::device_fft_buffers<Interface>::ComplexBuffer &in,
           typename detail::device_fft_buffers<Interface>::RealBuffer &out)
    requires detail::device_fft_buffers<Interface>::value
  {
    backward_device_(in, out);
  }

  /**
   * @brief Backward transform via host vectors (FFTW / `IHostFFT` only).
   * @throws std::invalid_argument if `in.size() != size_outbox()` or
   *         `out.size() != size_inbox()`
   */
  void backward(const ComplexVector &in, RealVector &out)
    requires std::is_base_of_v<IHostFFT, Interface>
  {
    detail::require_equal_size(
        in.size(), size_outbox(),
        "FFT_Impl::backward: complex buffer size ", "size_outbox");
    detail::require_equal_size(
        out.size(), size_inbox(),
        "FFT_Impl::backward: real buffer size ", "size_inbox");
    m_fft_time -= MPI_Wtime();
    m_fft.backward(in.data(), out.data(), m_ws.data_wrk(), heffte::scale::full);
    m_fft_time += MPI_Wtime();
  }

  void reset_fft_time() override { m_fft_time = 0.0; }

  double get_fft_time() const override { return m_fft_time; }

  size_t size_inbox() const override { return m_fft.size_inbox(); }

  size_t size_outbox() const override { return m_fft.size_outbox(); }

  size_t size_workspace() const override { return m_fft.size_workspace(); }

  size_t get_allocated_memory_bytes() const override {
    return m_ws.allocated_bytes();
  }

  Box3i get_inbox_bounds() const override {
    const auto &in = m_fft.inbox();
    return Box3i{in.low, in.high, in.size};
  }

  Box3i get_outbox_bounds() const override {
    const auto &out = m_fft.outbox();
    return Box3i{out.low, out.high, out.size};
  }

private:
  template <typename RealBackendTag, typename ComplexBackendTag, typename RealType>
  void forward_device_(
      const core::DataBuffer<RealBackendTag, RealType> &in,
      core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &out) {
    static_assert(std::is_same_v<RealBackendTag, ComplexBackendTag>,
                  "Input and output must use the same backend");
    detail::require_equal_size(in.size(), size_inbox(),
                               "FFT_Impl::forward: real buffer size ",
                               "size_inbox");
    detail::require_equal_size(out.size(), size_outbox(),
                               "FFT_Impl::forward: complex buffer size ",
                               "size_outbox");
    m_fft_time -= MPI_Wtime();
    if constexpr (std::is_same_v<RealType, double>) {
      m_fft.forward(in.data(), out.data(), m_ws.data_gpu_double());
    } else if constexpr (std::is_same_v<RealType, float>) {
      m_fft.forward(in.data(), out.data(), m_ws.data_gpu_float());
    }
    m_fft_time += MPI_Wtime();
  }

  template <typename ComplexBackendTag, typename RealBackendTag, typename RealType>
  void backward_device_(
      const core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &in,
      core::DataBuffer<RealBackendTag, RealType> &out) {
    static_assert(std::is_same_v<ComplexBackendTag, RealBackendTag>,
                  "Input and output must use the same backend");
    detail::require_equal_size(in.size(), size_outbox(),
                               "FFT_Impl::backward: complex buffer size ",
                               "size_outbox");
    detail::require_equal_size(out.size(), size_inbox(),
                               "FFT_Impl::backward: real buffer size ",
                               "size_inbox");
    m_fft_time -= MPI_Wtime();
    if constexpr (std::is_same_v<RealType, double>) {
      m_fft.backward(in.data(), out.data(), m_ws.data_gpu_double(),
                     heffte::scale::full);
    } else if constexpr (std::is_same_v<RealType, float>) {
      m_fft.backward(in.data(), out.data(), m_ws.data_gpu_float(),
                     heffte::scale::full);
    }
    m_fft_time += MPI_Wtime();
  }
};

template <typename BackendTag, typename Interface>
inline const auto &
get_fft_object(const FFT_Impl<BackendTag, Interface> &fft) noexcept {
  return fft.m_fft;
}

} // namespace fft
} // namespace pfc
