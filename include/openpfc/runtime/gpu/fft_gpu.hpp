// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_gpu.hpp
 * @brief Device FFT factories (`create_cuda` / `create_hip`) for M5.
 *
 * @details
 * Single source for GPU FFT type aliases and factories. Vendor headers
 * `runtime/cuda/fft_cuda.hpp` and `runtime/hip/fft_hip.hpp` are thin includes.
 * Factories are compiled only when the matching HeFFTe GPU backend is on
 * (`OpenPFC_ENABLE_CUDA_SPECTRAL` / `OpenPFC_ENABLE_HIP_SPECTRAL`).
 *
 * @see kernel/fft/fft_interface.hpp
 * @see kernel/fft/detail/fft_heffte_backend.hpp
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

using RealDataBufferCUDA = core::DataBuffer<backend::CUDATag, double>;
using ComplexDataBufferCUDA =
    core::DataBuffer<backend::CUDATag, std::complex<double>>;
using fft_r2c_cuda = heffte::fft3d_r2c<heffte::backend::cufft>;
using FFT_CUDA = FFT_Impl<heffte::backend::cufft, IDeviceFFT<pfc::CUDASpace>>;

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id,
                                   MPI_Comm comm, int r2c_direction,
                                   const heffte::plan_options &options);

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition, int rank_id,
                                   MPI_Comm comm = MPI_COMM_WORLD,
                                   int r2c_direction = 0);

[[nodiscard]] FFT_CUDA create_cuda(const Decomposition &decomposition,
                                   MPI_Comm comm = MPI_COMM_WORLD);

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)

using RealDataBufferHIP = core::DataBuffer<backend::HIPTag, double>;
using ComplexDataBufferHIP = core::DataBuffer<backend::HIPTag, std::complex<double>>;
using fft_r2c_hip = heffte::fft3d_r2c<heffte::backend::rocfft>;
using FFT_HIP = FFT_Impl<heffte::backend::rocfft, IDeviceFFT<pfc::HIPSpace>>;

[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition, int rank_id,
                                 MPI_Comm comm, int r2c_direction,
                                 const heffte::plan_options &options);

[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition, int rank_id,
                                 MPI_Comm comm = MPI_COMM_WORLD,
                                 int r2c_direction = 0);

[[nodiscard]] FFT_HIP create_hip(const Decomposition &decomposition,
                                 MPI_Comm comm = MPI_COMM_WORLD);

#endif // OpenPFC_ENABLE_HIP_SPECTRAL

} // namespace pfc::fft
