// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft.hpp
 * @brief Fast Fourier Transform interface for spectral methods
 *
 * @details
 * This file defines the FFT class and related utilities for performing distributed
 * parallel FFT operations using HeFFTe. OpenPFC uses spectral methods where
 * derivatives and other spatial operations are efficiently computed in Fourier
 * space.
 *
 * The FFT class provides:
 * - Distributed-memory parallel 3D real-to-complex and complex-to-real transforms
 * - Integration with HeFFTe backend (supports FFTW, cuFFT, rocFFT)
 * - FFTLayout for managing real/complex data decomposition
 * - Helper functions for k-space operations (wavenumbers, Laplacian operators)
 *
 * Typical usage:
 * @code
 * // Create FFT for decomposed domain
 * pfc::decomposition::Decomposition decomp(world, MPI_COMM_WORLD);
 * pfc::FFT fft(decomp, pfc::fft::FFTBackend::Default);
 *
 * // Transform to Fourier space
 * std::vector<double> real_data = ...;
 * std::vector<std::complex<double>> fourier_data = fft.forward(real_data);
 *
 * // Compute derivatives in k-space, then transform back
 * // ... modify fourier_data ...
 * std::vector<double> result = fft.backward(fourier_data);
 * @endcode
 *
 * This file is part of the Core Infrastructure module, providing the foundation
 * for spectral method computations in phase-field simulations.
 *
 * @see core/decomposition.hpp for domain decomposition
 * @see model.hpp for FFT usage in physics models
 * @see backends/heffte_adapter.hpp for HeFFTe integration
 */

#pragma once

#include "core/decomposition.hpp"
#include "openpfc/backends/heffte_adapter.hpp" // Ensure this is included for the conversion operator
#include "openpfc/core/backend_tags.hpp"
#include "openpfc/core/databuffer.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/fft/kspace.hpp"

#include <heffte.h>
#include <iostream>
#include <memory>
#include <mpi.h>

namespace pfc {
namespace fft {

namespace layout {

using box3di = heffte::box3d<int>;
using Decomposition = pfc::decomposition::Decomposition;
using pfc::types::Int3;

/**
 * @brief Structure to hold the layout of FFT data.
 *
 * This structure contains the decomposition object, the direction of
 * real-to-complex symmetry, and the boxes for real and complex FFT data.
 */
struct FFTLayout {
  const Decomposition m_decomposition; ///< The Decomposition object.
  const int m_r2c_direction = 0;       ///< Real-to-complex symmetry direction.
  const std::vector<heffte::box3d<int>> m_real_boxes;    ///< Real boxes for FFT.
  const std::vector<heffte::box3d<int>> m_complex_boxes; ///< Complex boxes for FFT.
};

/**
 * @brief Creates an FFTLayout object based on the given decomposition and
 * parameters.
 *
 * @param decomposition The Decomposition object defining the domain
 * decomposition.
 * @param r2c_direction The direction of real-to-complex symmetry.
 * @return An FFTLayout object containing the layout information.
 */
const FFTLayout create(const Decomposition &decomposition, int r2c_direction);

inline const auto &get_real_box(const FFTLayout &layout, int i) {
  return layout.m_real_boxes.at(i);
}

inline const auto &get_complex_box(const FFTLayout &layout, int i) {
  return layout.m_complex_boxes.at(i);
}

inline auto get_r2c_direction(const FFTLayout &layout) {
  return layout.m_r2c_direction;
}

} // namespace layout

using pfc::types::Int3;
using pfc::types::Real3;

using Decomposition = pfc::decomposition::Decomposition;
using RealVector = std::vector<double>;
using ComplexVector = std::vector<std::complex<double>>;
using box3di = heffte::box3d<int>; ///< Type alias for 3D integer box.

// Backend-aware DataBuffer type aliases
using RealDataBuffer = core::DataBuffer<backend::CpuTag, double>;
using ComplexDataBuffer = core::DataBuffer<backend::CpuTag, std::complex<double>>;
#if defined(OpenPFC_ENABLE_CUDA)
using RealDataBufferCUDA = core::DataBuffer<backend::CudaTag, double>;
using ComplexDataBufferCUDA =
    core::DataBuffer<backend::CudaTag, std::complex<double>>;
#endif

/**
 * @brief FFT backend selection
 *
 * Specifies which FFT library backend to use for computations.
 * - FFTW: CPU-based FFT (default, always available)
 * - CUDA: GPU-based FFT using cuFFT (requires CUDA and OpenPFC_ENABLE_CUDA)
 */
enum class Backend {
  FFTW, ///< CPU-based FFT using FFTW (default)
  CUDA  ///< GPU-based FFT using cuFFT (requires CUDA support)
};

// Type aliases for different FFT backends
using fft_r2c = heffte::fft3d_r2c<heffte::backend::fftw>; ///< FFTW backend (CPU)
#if defined(OpenPFC_ENABLE_CUDA)
using fft_r2c_cuda =
    heffte::fft3d_r2c<heffte::backend::cufft>; ///< cuFFT backend (GPU)
#endif

struct IFFT {
  virtual ~IFFT() = default;

  /**
   * @brief Performs the forward FFT transformation.
   *
   * @param in Input vector of real values.
   * @param out Output vector of complex values.
   */
  virtual void forward(const RealVector &in, ComplexVector &out) = 0;

  /**
   * @brief Performs the backward (inverse) FFT transformation.
   *
   * @param in Input vector of complex values.
   * @param out Output vector of real values.
   */
  virtual void backward(const ComplexVector &in, RealVector &out) = 0;

  virtual void reset_fft_time() = 0;
  virtual double get_fft_time() const = 0;

  virtual size_t size_inbox() const = 0;
  virtual size_t size_outbox() const = 0;
  virtual size_t size_workspace() const = 0;
};

/**
 * @brief FFT class for distributed-memory parallel Fourier transforms
 *
 * Provides real-to-complex (R2C) and complex-to-real (C2R) 3D FFT operations
 * using HeFFTe backend. This is the core computational engine for spectral
 * methods in OpenPFC, enabling efficient calculation of derivatives and
 * convolutions in Fourier space.
 *
 * ## Key Features
 * - Distributed-memory parallelism via MPI
 * - Real-to-complex symmetry exploitation (half-space representation)
 * - Multiple backend support (FFTW, cuFFT, rocFFT)
 * - Automatic workspace management
 * - Performance timing capabilities
 *
 * ## Memory Layout
 * - Real data: Full 3D grid (N_x × N_y × N_z)
 * - Complex data: Half-space (N_x × N_y × (N_z/2+1)) due to conjugate symmetry
 * - Both use distributed decomposition across MPI ranks
 *
 * ## Normalization Convention
 * - Forward transform: No normalization
 * - Backward transform: Divides by total grid points (1/N)
 * - Round-trip: x → forward → backward → x (exact)
 *
 * ## Usage Pattern
 * @code
 * // Setup
 * auto world = world::create({256, 256, 256});
 * auto decomp = Decomposition(world, MPI_COMM_WORLD);
 * auto fft = fft::create(decomp);
 *
 * // Allocate fields
 * RealVector real_field(fft.size_inbox());
 * ComplexVector fourier_field(fft.size_outbox());
 *
 * // Forward: real space → k-space
 * fft.forward(real_field, fourier_field);
 *
 * // Apply operators in k-space
 * for (size_t k = 0; k < fourier_field.size(); ++k) {
 *     fourier_field[k] *= laplacian_operator[k];
 * }
 *
 * // Backward: k-space → real space (normalized)
 * fft.backward(fourier_field, real_field);
 * @endcode
 *
 * ## Performance Notes
 * - FFT is O(N log N) operation
 * - MPI communication overhead scales with domain decomposition
 * - Use reset_fft_time() and get_fft_time() to measure performance
 * - Powers of 2 for grid dimensions yield fastest transforms
 *
 * @note FFT stores a reference/pointer to Decomposition - ensure lifetime
 * @warning Forward and backward are NOT inverses without normalization
 * @warning Complex array is half the size of real array (conjugate symmetry)
 *
 * @see fft::create() for construction
 * @see Decomposition for domain decomposition
 * @see kspace.hpp for wavenumber and operator helpers
 */
/**
 * @brief FFT class template for distributed-memory parallel Fourier transforms
 *
 * @tparam BackendTag HeFFTe backend tag (heffte::backend::fftw or
 * heffte::backend::cufft)
 *
 * @note Precision (float/double) is determined by the data types passed to
 *       forward() and backward() methods, not by template parameters.
 *       HeFFTe automatically handles precision based on input/output types.
 */
template <typename BackendTag = heffte::backend::fftw> struct FFT_Impl : IFFT {

  // const Decomposition m_decomposition; /**< The Decomposition object. */
  // const box3di m_inbox, m_outbox;      /**< Local inbox and outbox boxes. */

  using fft_type = heffte::fft3d_r2c<BackendTag>;
  const fft_type m_fft;    /**< HeFFTe FFT object. */
  double m_fft_time = 0.0; /**< Recorded FFT computation time. */

  // Backend-aware workspace - precision determined by data types in forward/backward
  // calls Default to double precision workspace (can be overridden per call)
  using workspace_type = typename heffte::fft3d_r2c<
      BackendTag>::template buffer_container<std::complex<double>>;
  workspace_type
      m_wrk; /**< Workspace vector for FFT computations (double precision). */

  /**
   * @brief Constructs an FFT object with the given HeFFTe FFT object
   *
   * @param fft HeFFTe FFT object (already configured)
   */
  FFT_Impl(fft_type fft) : m_fft(std::move(fft)), m_wrk(m_fft.size_workspace()) {}

  /**
   * @brief Perform forward real-to-complex FFT transform
   *
   * Transforms real-space data to Fourier-space (k-space) using a distributed
   * 3D FFT. The output exploits conjugate symmetry (half-space representation).
   *
   * @param in Input vector of real values (size = size_inbox())
   * @param out Output vector of complex values (size = size_outbox())
   *
   * @pre in.size() must equal size_inbox()
   * @pre out.size() must equal size_outbox()
   *
   * @note No normalization applied (use 1/N if needed)
   * @note Output is half-complex due to conjugate symmetry
   * @note MPI collective operation - all ranks must call
   *
   * @warning Modifies internal workspace (not thread-safe)
   *
   * @example
   * ```cpp
   * auto fft = fft::create(decomp);
   * RealVector density(fft.size_inbox(), 0.5);  // Uniform field
   * ComplexVector density_k(fft.size_outbox());
   *
   * fft.forward(density, density_k);
   * // density_k now contains Fourier coefficients
   * // density_k[0] = N * 0.5 (DC component, no normalization)
   * ```
   *
   * Time complexity: O(N log N) locally + MPI communication
   *
   * @see backward() for inverse transform
   * @see size_inbox() for input size
   * @see size_outbox() for output size
   */
  // Forward method using DataBuffer (backend-aware, precision-aware via template)
  template <typename RealBackendTag, typename ComplexBackendTag, typename RealType>
  void forward(const core::DataBuffer<RealBackendTag, RealType> &in,
               core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &out) {
    static_assert(std::is_same_v<RealBackendTag, ComplexBackendTag>,
                  "Input and output must use the same backend");
    m_fft_time -= MPI_Wtime();
    // HeFFTe's forward method is templated on input/output types and handles
    // precision automatically Create workspace with matching precision
    auto wrk = typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
        std::complex<RealType>>(m_fft.size_workspace());
    m_fft.forward(in.data(), out.data(), wrk.data());
    m_fft_time += MPI_Wtime();
  }

  // Forward method using std::vector (implements IFFT interface)
  // For CPU backend: works directly with std::vector
  // For GPU backend: throws error (must use DataBuffer overload)
  void forward(const RealVector &in, ComplexVector &out) override {
    if constexpr (std::is_same_v<BackendTag, heffte::backend::fftw>) {
      // CPU backend: call HeFFTe directly (no conversion needed)
      m_fft_time -= MPI_Wtime();
      m_fft.forward(in.data(), out.data(), m_wrk.data());
      m_fft_time += MPI_Wtime();
    } else {
      // GPU backend: must use DataBuffer
      throw std::runtime_error(
          "GPU FFT requires DataBuffer, not std::vector. Use forward(DataBuffer, "
          "DataBuffer) instead.");
    }
  }

  /**
   * @brief Perform backward complex-to-real inverse FFT transform
   *
   * Transforms Fourier-space (k-space) data back to real space. Applies full
   * normalization (divides by total grid points N) so round-trip transforms
   * are exact: x → forward → backward → x.
   *
   * @param in Input vector of complex values (size = size_outbox())
   * @param out Output vector of real values (size = size_inbox())
   *
   * @pre in.size() must equal size_outbox()
   * @pre out.size() must equal size_inbox()
   *
   * @note Applies full normalization: output = IFFT(input) / N
   * @note Input uses half-complex representation
   * @note MPI collective operation - all ranks must call
   *
   * @warning Modifies internal workspace (not thread-safe)
   * @warning Input must satisfy conjugate symmetry for real output
   *
   * @example
   * ```cpp
   * ComplexVector field_k(fft.size_outbox());
   * RealVector field(fft.size_inbox());
   *
   * // Apply Laplacian in k-space: \u0302f(k) → -k²\u0302f(k)
   * for (size_t idx = 0; idx < field_k.size(); ++idx) {
   *     double k2 = kx[idx]*kx[idx] + ky[idx]*ky[idx] + kz[idx]*kz[idx];
   *     field_k[idx] *= -k2;
   * }
   *
   * // Transform back (normalized)
   * fft.backward(field_k, field);
   * // field now contains ∇²f in real space
   * ```
   *
   * Time complexity: O(N log N) locally + MPI communication
   *
   * @see forward() for forward transform
   * @see size_inbox() for output size
   * @see size_outbox() for input size
   */
  // Backward method using DataBuffer (backend-aware, precision-aware via template)
  template <typename ComplexBackendTag, typename RealBackendTag, typename RealType>
  void
  backward(const core::DataBuffer<ComplexBackendTag, std::complex<RealType>> &in,
           core::DataBuffer<RealBackendTag, RealType> &out) {
    static_assert(std::is_same_v<ComplexBackendTag, RealBackendTag>,
                  "Input and output must use the same backend");
    m_fft_time -= MPI_Wtime();
    // HeFFTe's backward method is templated on input/output types and handles
    // precision automatically Create workspace with matching precision
    auto wrk = typename heffte::fft3d_r2c<BackendTag>::template buffer_container<
        std::complex<RealType>>(m_fft.size_workspace());
    m_fft.backward(in.data(), out.data(), wrk.data(), heffte::scale::full);
    m_fft_time += MPI_Wtime();
  }

  // Backward method using std::vector (implements IFFT interface)
  // For CPU backend: works directly with std::vector
  // For GPU backend: throws error (must use DataBuffer overload)
  void backward(const ComplexVector &in, RealVector &out) override {
    if constexpr (std::is_same_v<BackendTag, heffte::backend::fftw>) {
      // CPU backend: call HeFFTe directly (no conversion needed)
      m_fft_time -= MPI_Wtime();
      m_fft.backward(in.data(), out.data(), m_wrk.data(), heffte::scale::full);
      m_fft_time += MPI_Wtime();
    } else {
      // GPU backend: must use DataBuffer
      throw std::runtime_error(
          "GPU FFT requires DataBuffer, not std::vector. Use backward(DataBuffer, "
          "DataBuffer) instead.");
    }
  }

  /**
   * @brief Resets the recorded FFT computation time to zero.
   */
  void reset_fft_time() { m_fft_time = 0.0; }

  /**
   * @brief Returns the recorded FFT computation time.
   *
   * @return The FFT computation time in seconds.
   */
  double get_fft_time() const { return m_fft_time; }

  /**
   * @brief Returns the associated Decomposition object.
   *
   * @return Reference to the Decomposition object.
   */
  // const Decomposition &get_decomposition() { return m_decomposition; }

  /**
   * @brief Returns the size of the inbox used for FFT computations.
   *
   * @return Size of the inbox.
   */
  size_t size_inbox() const { return m_fft.size_inbox(); }

  /**
   * @brief Returns the size of the outbox used for FFT computations.
   *
   * @return Size of the outbox.
   */
  size_t size_outbox() const { return m_fft.size_outbox(); }

  /**
   * @brief Returns the size of the workspace used for FFT computations.
   *
   * @return Size of the workspace.
   */
  size_t size_workspace() const { return m_fft.size_workspace(); }
};

// Type aliases for backward compatibility (defaults to FFTW backend)
// Precision is handled by data types, not template parameters
using FFT = FFT_Impl<heffte::backend::fftw>;

// Helper functions
template <typename BackendTag>
inline const auto &get_fft_object(const FFT_Impl<BackendTag> &fft) noexcept {
  return fft.m_fft;
}

template <typename BackendTag>
inline const auto get_inbox(const FFT_Impl<BackendTag> &fft) noexcept {
  return get_fft_object(fft).inbox();
}

template <typename BackendTag>
inline const auto get_outbox(const FFT_Impl<BackendTag> &fft) noexcept {
  return get_fft_object(fft).outbox();
}

using heffte::plan_options;
using layout::FFTLayout;

/**
 * @brief Creates an FFT object based on the given FFTLayout and rank ID.
 *
 * @param fft_layout The FFTLayout object defining the FFT configuration.
 * @param rank_id The rank ID of the current process in the MPI communicator.
 * @param options Plan options for configuring the FFT behavior.
 * @return An FFT object containing the FFT configuration and data.
 *
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods.
 */
FFT create(const FFTLayout &fft_layout, int rank_id, plan_options options);

/**
 * @brief Creates an FFT object based on the given decomposition and rank ID.
 *
 * @param decomposition The Decomposition object defining the domain
 * decomposition.
 * @param rank_id The rank ID of the current process in the MPI communicator.
 * @return An FFT object containing the FFT configuration and data.
 *
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods.
 */
FFT create(const Decomposition &decomposition, int rank_id);

/**
 * @brief Creates an FFT object based on the given decomposition.
 *
 * @param decomposition The Decomposition object defining the domain
 * decomposition.
 * @return An FFT object containing the FFT configuration and data.
 * @throws std::logic_error, if decomposition size and rank size do not match.
 *
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods.
 */
FFT create(const Decomposition &decomposition);

/**
 * @brief Creates an FFT object with runtime backend selection
 *
 * @param fft_layout The FFTLayout object defining the FFT configuration.
 * @param rank_id The rank ID of the current process in the MPI communicator.
 * @param options Plan options for configuring the FFT behavior.
 * @param backend The FFT backend to use (FFTW, CUDA, etc.)
 * @return A unique_ptr to IFFT interface for the selected backend
 * @throws std::runtime_error if backend is not supported or not compiled in
 *
 * @note This function provides runtime polymorphism via the IFFT interface.
 *       For compile-time selection with zero overhead, use create() directly.
 */
std::unique_ptr<IFFT> create_with_backend(const FFTLayout &fft_layout, 
                                           int rank_id, 
                                           plan_options options, 
                                           Backend backend);

/**
 * @brief Creates an FFT object with runtime backend selection
 *
 * @param decomposition The Decomposition object defining the domain decomposition.
 * @param rank_id The rank ID of the current process in the MPI communicator.
 * @param backend The FFT backend to use (FFTW, CUDA, etc.)
 * @return A unique_ptr to IFFT interface for the selected backend
 * @throws std::runtime_error if backend is not supported or not compiled in
 */
std::unique_ptr<IFFT> create_with_backend(const Decomposition &decomposition, 
                                           int rank_id, 
                                           Backend backend);

} // namespace fft

using FFT = fft::FFT;                     ///< Type alias for FFT class.
using FFTLayout = fft::layout::FFTLayout; ///< Type alias for FFTLayout class.

} // namespace pfc
