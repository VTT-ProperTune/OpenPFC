// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_heffte_backend.hpp
 * @brief HeFFTe-backed FFT_Impl template (include only where HeFFTe is required)
 */

#pragma once

#include <openpfc/kernel/fft/fft_interface.hpp>

#include <heffte.h>
#include <mpi.h>

#include <stdexcept>
#include <type_traits>

namespace pfc {
namespace fft {

/**
 * @brief FFT class template for distributed-memory parallel Fourier transforms
 *
 * @tparam BackendTag HeFFTe backend tag (heffte::backend::fftw or
 * heffte::backend::cufft / rocfft)
 */
template <typename BackendTag = heffte::backend::fftw> struct FFT_Impl : IFFT {

  using fft_type = heffte::fft3d_r2c<BackendTag>;
  const fft_type m_fft;
  double m_fft_time = 0.0;

  using workspace_type = typename heffte::fft3d_r2c<
      BackendTag>::template buffer_container<std::complex<double>>;
  workspace_type m_wrk;

  FFT_Impl(fft_type fft) : m_fft(std::move(fft)), m_wrk(m_fft.size_workspace()) {}

  template <typename RealBackendTag, typename ComplexBackendTag, typename RealType>
  void forward(const core::DataBuffer<RealBackendTag, RealType> &in,
               core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &out) {
    static_assert(std::is_same_v<RealBackendTag, ComplexBackendTag>,
                  "Input and output must use the same backend");
    m_fft_time -= MPI_Wtime();
    auto wrk = typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
        std::complex<RealType>>(m_fft.size_workspace());
    m_fft.forward(in.data(), out.data(), wrk.data());
    m_fft_time += MPI_Wtime();
  }

  void forward(const RealVector &in, ComplexVector &out) override {
    if constexpr (std::is_same_v<BackendTag, heffte::backend::fftw>) {
      m_fft_time -= MPI_Wtime();
      m_fft.forward(in.data(), out.data(), m_wrk.data());
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
    m_fft_time -= MPI_Wtime();
    auto wrk = typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
        std::complex<RealType>>(m_fft.size_workspace());
    m_fft.backward(in.data(), out.data(), wrk.data(), heffte::scale::full);
    m_fft_time += MPI_Wtime();
  }

  void backward(const ComplexVector &in, RealVector &out) override {
    if constexpr (std::is_same_v<BackendTag, heffte::backend::fftw>) {
      m_fft_time -= MPI_Wtime();
      m_fft.backward(in.data(), out.data(), m_wrk.data(), heffte::scale::full);
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
    return m_wrk.size() * sizeof(typename workspace_type::value_type);
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
