// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_cuda.hpp
 * @brief GPU FFT factory functions using cuFFT backend
 *
 * @details
 * This file provides factory functions to create FFT objects using the cuFFT
 * backend for GPU-accelerated FFT operations. These functions are only available
 * when CUDA spectral support is enabled at compile time.
 *
 * @note Only available when OpenPFC_ENABLE_CUDA_SPECTRAL is defined.
 * @see fft.hpp for the main FFT interface
 */

#pragma once

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/detail/fft_heffte_backend.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#include <mpi.h>

namespace pfc::fft {

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)

// CUDA DataBuffer type aliases (moved from kernel/fft/fft.hpp)
using RealDataBufferCUDA = core::DataBuffer<backend::CudaTag, double>;
using ComplexDataBufferCUDA =
    core::DataBuffer<backend::CudaTag, std::complex<double>>;

// cuFFT backend type alias
using fft_r2c_cuda = heffte::fft3d_r2c<heffte::backend::cufft>;

/**
 * @brief Creates an FFT object using cuFFT backend for GPU acceleration
 *
 * This function creates an FFT object that uses HeFFTe's cuFFT backend,
 * enabling GPU-accelerated FFT operations. All FFT computations will be
 * performed on the GPU.
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @param rank_id The rank ID of the current process in the MPI communicator
 * @return FFT object configured to use cuFFT backend
 *
 * @throws std::runtime_error if CUDA is not available or FFT creation fails
 *
 * @note Only available when OpenPFC_ENABLE_CUDA_SPECTRAL is defined
 * @note Requires CUDA-capable GPU and GPU-aware MPI for multi-GPU setups
 *
 * @example
 * @code{.cpp}
 * #ifdef OpenPFC_ENABLE_CUDA_SPECTRAL
 * #include <openpfc/kernel/mpi/mpi.hpp>
 *     pfc::Domain domain = pfc::domain::create({128, 128, 128});
 *     auto decomp = decomposition::create(domain, mpi::get_size());
 *     auto gpu_fft = fft::create_cuda(decomp, mpi::get_rank());
 * #endif
 * @endcode
 */
// GPU FFT type alias — implements `IDeviceFFT<CudaSpace>`.
using FFT_CUDA = FFT_Impl<heffte::backend::cufft, IDeviceFFT<pfc::CudaSpace>>;

/**
 * @brief Creates an FFT object using cuFFT backend for GPU acceleration
 *
 * This function creates an FFT object that uses HeFFTe's cuFFT backend,
 * enabling GPU-accelerated FFT operations. All FFT computations will be
 * performed on the GPU.
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @param rank_id The rank ID of the current process in the MPI communicator
 * @return FFT_CUDA object configured to use cuFFT backend
 *
 * @throws std::runtime_error if CUDA is not available or FFT creation fails
 *
 * @note Only available when OpenPFC_ENABLE_CUDA_SPECTRAL is defined
 * @note Requires CUDA-capable GPU and GPU-aware MPI for multi-GPU setups
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods
 */
[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id,
                                   MPI_Comm comm = MPI_COMM_WORLD,
                                   int r2c_direction = 0);

/**
 * @brief Creates an FFT object using cuFFT backend (auto-detect rank)
 *
 * Convenience function that automatically detects the MPI rank from @p comm.
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @return FFT_CUDA object configured to use cuFFT backend
 *
 * @throws std::logic_error if MPI communicator size doesn't match decomposition size
 * @throws std::runtime_error if CUDA is not available or FFT creation fails
 *
 * @note Only available when OpenPFC_ENABLE_CUDA_SPECTRAL is defined
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods
 */
[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition,
                                   MPI_Comm comm = MPI_COMM_WORLD,
                                   int r2c_direction = 0);

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL

} // namespace pfc::fft
