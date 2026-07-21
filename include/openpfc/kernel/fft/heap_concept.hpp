// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file heap_concept.hpp
 * @brief C++20 concept unifying GPU workspace *types* across HeFFTe
 *        device backends.
 *
 * @details
 * `HeapBackend` is a compile-time contract on dual-precision
 * `buffer_container` type availability for GPU tags
 * (`heffte::backend::cufft`, `heffte::backend::rocfft`): both
 * `std::complex<double>` and `std::complex<float>` containers must exist, be
 * constructible from a size, and expose `size()`/`data()`.
 *
 * It does **not** require every `FFT_Impl` instantiation to eagerly allocate
 * unused workspaces. Eager ownership is backend-specialized in
 * `fft_heffte_backend.hpp` via `detail::FftWorkspaceStorage`:
 * - FFTW: host `m_wrk` only (this concept does not apply).
 * - GPU: per-precision device workspaces only (no idle host `m_wrk`).
 *
 * The CPU backend (`heffte::backend::fftw`) is excluded on purpose -- it uses
 * `std::vector`-backed host memory, not a device-allocating container.
 *
 * @see fft_heffte_backend.hpp for the two `static_assert`s that check this
 *      contract against the real CUDA/HIP backend tags whenever the
 *      corresponding `OpenPFC_ENABLE_CUDA`/`OpenPFC_ENABLE_HIP` build
 *      option is on.
 */

#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include <heffte.h>

namespace pfc::fft {

/** Excludes the CPU (FFTW) backend, which is not a device allocator. */
template <typename BackendTag>
concept NotCPUBackend = !std::is_same_v<BackendTag, heffte::backend::fftw>;

/**
 * @brief Satisfied by a HeFFTe GPU backend tag whose per-precision buffer
 *        containers are constructible from a size and expose `size()`/
 *        `data()` -- the type availability GPU `FftWorkspaceStorage` relies
 *        on for both `std::complex<double>` and `std::complex<float>`.
 */
template <typename BackendTag>
concept HeapBackend =
    NotCPUBackend<BackendTag> &&
    requires(typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
                 std::complex<double>>
                 buf,
             std::size_t size) {
      // Both precision workspace types must exist.
      typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
          std::complex<double>>;
      typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
          std::complex<float>>;

      // GPU containers must support construction from a size parameter.
      { decltype(buf)(size) } -> std::same_as<decltype(buf)>;

      // GPU containers must provide size inquiry and data pointer access.
      { buf.size() } -> std::convertible_to<std::size_t>;
      { buf.data() } -> std::convertible_to<typename decltype(buf)::value_type *>;
    };

} // namespace pfc::fft
