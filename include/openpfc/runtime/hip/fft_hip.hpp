// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_hip.hpp
 * @brief GPU FFT factory functions using rocFFT backend
 *
 * @details
 * This file provides factory functions to create FFT objects using the rocFFT
 * backend for GPU-accelerated FFT operations on AMD GPUs. These functions are
 * only available when HIP (ROCm) is enabled at compile time.
 *
 * @note Only available when OpenPFC_ENABLE_HIP is defined.
 * @see fft.hpp for the main FFT interface
 */

#pragma once

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/detail/fft_heffte_backend.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>

#include <mpi.h>

namespace pfc::fft {

/**
 * @brief Creates an FFT object using rocFFT backend for GPU acceleration
 *
 * This function creates an FFT object that uses HeFFTe's rocFFT backend,
 * enabling GPU-accelerated FFT operations on AMD GPUs. All FFT computations
 * will be performed on the GPU.
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @param rank_id The rank ID of the current process in the MPI communicator
 * @return FFT object configured to use rocFFT backend
 *
 * @throws std::runtime_error if HIP is not available or FFT creation fails
 *
 * @note Only available when OpenPFC_ENABLE_HIP is defined
 * @note Requires AMD GPU with ROCm and GPU-aware MPI for multi-GPU setups
 */
#if defined(OpenPFC_ENABLE_HIP)
// GPU FFT type alias for rocFFT backend
using FFT_HIP = FFT_Impl<heffte::backend::rocfft>;

/**
 * @brief Creates an FFT object using rocFFT backend for GPU acceleration
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @param rank_id The rank ID of the current process in the MPI communicator
 * @return FFT_HIP object configured to use rocFFT backend
 */
[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition, int rank_id,
                                 MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Creates an FFT object using rocFFT backend (auto-detect rank)
 *
 * Convenience function that automatically detects the MPI rank from @p comm.
 *
 * @param decomposition The Decomposition object defining the domain decomposition
 * @return FFT_HIP object configured to use rocFFT backend
 *
 * @throws std::logic_error if MPI communicator size doesn't match decomposition size
 */
[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition,
                                 MPI_Comm comm = MPI_COMM_WORLD);
#endif // OpenPFC_ENABLE_HIP

} // namespace pfc::fft
