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
 * - GPU backends (`cufft` / `rocfft`) own only the dual-precision device
 *   workspaces used by the `DataBuffer` path. Both float and double remain
 *   because `FFT_Impl` is not templated on `RealType` and both overloads share
 *   one instance — no idle host/device twin is allocated.
 */

#pragma once

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/heap_concept.hpp>

#include <heffte.h>
#include <mpi.h>

#include <cstddef>
#include <stdexcept>
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
 * @brief GPU backends: dual-precision device workspaces only (no unused `m_wrk`).
 *
 * Both precisions stay owned because float and double `DataBuffer` overloads
 * share one `FFT_Impl` instance.
 */
template <typename BackendTag>
  requires HeapBackend<BackendTag>
struct FftWorkspaceStorage<BackendTag> {
  using gpu_workspace_type = typename heffte::fft3d_r2c<
      BackendTag>::template buffer_container<std::complex<double>>;
  using gpu_workspace_float = typename heffte::fft3d_r2c<
      BackendTag>::template buffer_container<std::complex<float>>;

  gpu_workspace_type m_gpu_wrk_double;
  gpu_workspace_float m_gpu_wrk_float;

  explicit FftWorkspaceStorage(std::size_t n)
      : m_gpu_wrk_double(n), m_gpu_wrk_float(n) {}

  auto *data_gpu_double() noexcept { return m_gpu_wrk_double.data(); }
  auto *data_gpu_float() noexcept { return m_gpu_wrk_float.data(); }

  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return m_gpu_wrk_double.size() *
               sizeof(typename gpu_workspace_type::value_type) +
           m_gpu_wrk_float.size() *
               sizeof(typename gpu_workspace_float::value_type);
  }
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
template <typename BackendTag = heffte::backend::fftw> struct FFT_Impl : IFFT {

  using fft_type = heffte::fft3d_r2c<BackendTag>;
  const fft_type m_fft;
  double m_fft_time = 0.0;

  detail::FftWorkspaceStorage<BackendTag> m_ws;

  FFT_Impl(fft_type fft)
      : m_fft(std::move(fft)), m_ws(m_fft.size_workspace()) {}

  template <typename RealBackendTag, typename ComplexBackendTag, typename RealType>
  void forward(const core::DataBuffer<RealBackendTag, RealType> &in,
               core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &out) {
    static_assert(std::is_same_v<RealBackendTag, ComplexBackendTag>,
                  "Input and output must use the same backend");
    if constexpr (HeapBackend<BackendTag>) {
      m_fft_time -= MPI_Wtime();
      if constexpr (std::is_same_v<RealType, double>) {
        m_fft.forward(in.data(), out.data(), m_ws.data_gpu_double());
      } else if constexpr (std::is_same_v<RealType, float>) {
        m_fft.forward(in.data(), out.data(), m_ws.data_gpu_float());
      }
      m_fft_time += MPI_Wtime();
    } else {
      throw std::runtime_error(
          "FFTW FFT requires std::vector, not DataBuffer. Use forward(RealVector, "
          "ComplexVector) instead.");
    }
  }

  void forward(const RealVector &in, ComplexVector &out) override {
    if constexpr (std::is_same_v<BackendTag, heffte::backend::fftw>) {
      m_fft_time -= MPI_Wtime();
      m_fft.forward(in.data(), out.data(), m_ws.data_wrk());
      m_fft_time += MPI_Wtime();
    } else {
      throw std::runtime_error(
          "GPU FFT requires DataBuffer, not std::vector. Use forward(DataBuffer, "
          "DataBuffer) instead.");
    }
  }

  template <typename ComplexBackendTag, typename RealBackendTag, typename RealType>
  void
  backward(const core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &in,
           core::DataBuffer<RealBackendTag, RealType> &out) {
    static_assert(std::is_same_v<ComplexBackendTag, RealBackendTag>,
                  "Input and output must use the same backend");
    if constexpr (HeapBackend<BackendTag>) {
      m_fft_time -= MPI_Wtime();
      if constexpr (std::is_same_v<RealType, double>) {
        m_fft.backward(in.data(), out.data(), m_ws.data_gpu_double(),
                       heffte::scale::full);
      } else if constexpr (std::is_same_v<RealType, float>) {
        m_fft.backward(in.data(), out.data(), m_ws.data_gpu_float(),
                       heffte::scale::full);
      }
      m_fft_time += MPI_Wtime();
    } else {
      throw std::runtime_error(
          "FFTW FFT requires std::vector, not DataBuffer. Use "
          "backward(ComplexVector, RealVector) instead.");
    }
  }

  void backward(const ComplexVector &in, RealVector &out) override {
    if constexpr (std::is_same_v<BackendTag, heffte::backend::fftw>) {
      m_fft_time -= MPI_Wtime();
      m_fft.backward(in.data(), out.data(), m_ws.data_wrk(), heffte::scale::full);
      m_fft_time += MPI_Wtime();
    } else {
      throw std::runtime_error(
          "GPU FFT requires DataBuffer, not std::vector. Use backward(DataBuffer, "
          "DataBuffer) instead.");
    }
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
};

template <typename BackendTag>
inline const auto &get_fft_object(const FFT_Impl<BackendTag> &fft) noexcept {
  return fft.m_fft;
}

} // namespace fft
} // namespace pfc
