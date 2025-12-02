// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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

#include "openpfc/core/decomposition.hpp"
#include "openpfc/fft.hpp"

#include <heffte.h>
#include <mpi.h>

namespace pfc {
namespace fft {

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
 *     auto decomp = Decomposition(world, MPI_COMM_WORLD);
 *     int rank_id = 0;  // Get MPI rank
 *     auto gpu_fft = fft::create_cuda(decomp, rank_id);
 * #endif
 * @endcode
 */
#if defined(OpenPFC_ENABLE_CUDA)
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
 */
FFT_CUDA create_cuda(const Decomposition &decomposition);
#endif // OpenPFC_ENABLE_CUDA

} // namespace fft
} // namespace pfc
