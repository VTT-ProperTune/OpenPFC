// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_cuda.hpp
 * @brief GPU FFT factory functions using cuFFT backend
 *
 * @details
 * This file provides factory functions to create FFT objects using the cuFFT
 * backend for GPU-accelerated FFT operations. These functions are only available
 * when CUDA is enabled at compile time.
 *
 * @note Only available when OpenPFC_ENABLE_CUDA is defined.
 * @see fft.hpp for the main FFT interface
 */

#pragma once

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>

#include <mpi.h>

namespace pfc::fft {

#if defined(OpenPFC_ENABLE_CUDA)

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
 * @note Only available when OpenPFC_ENABLE_CUDA is defined
 * @note Requires CUDA-capable GPU and GPU-aware MPI for multi-GPU setups
 *
 * @example
 * @code{.cpp}
 * #ifdef OpenPFC_ENABLE_CUDA
 * #include <openpfc/kernel/mpi/mpi.hpp>
 *     auto world = world::create({128, 128, 128});
 *     auto decomp = decomposition::create(world, mpi::get_size());
 *     auto gpu_fft = fft::create_cuda(decomp, mpi::get_rank());
 * #endif
 * @endcode
 */
// GPU FFT type alias
using FFT_CUDA = FFT_Impl<heffte::backend::cufft>;

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
 * @note Only available when OpenPFC_ENABLE_CUDA is defined
 * @note Requires CUDA-capable GPU and GPU-aware MPI for multi-GPU setups
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods
 */
FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id);

/**
 * @brief Creates an FFT object using cuFFT backend (auto-detect rank)
 *
 * Convenience function that automatically detects the MPI rank from MPI_COMM_WORLD.
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @return FFT_CUDA object configured to use cuFFT backend
 *
 * @throws std::logic_error if MPI communicator size doesn't match decomposition size
 * @throws std::runtime_error if CUDA is not available or FFT creation fails
 *
 * @note Only available when OpenPFC_ENABLE_CUDA is defined
 * @note Precision (float/double) is determined by data types passed to
 * forward/backward methods
 */
FFT_CUDA create_cuda(const Decomposition &decomposition);

#endif // OpenPFC_ENABLE_CUDA

} // namespace pfc::fft
