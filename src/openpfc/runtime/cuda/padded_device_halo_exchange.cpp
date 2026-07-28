// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file padded_device_halo_exchange.cpp
 * @brief Implementation file for CUDA device halo exchange with Field API.
 *
 * This file contains explicit template instantiations for the Field-based
 * halo exchange API in padded_device_halo_exchange.hpp. Most implementations
 * are header-only templates; this file provides explicit instantiations for
 * common types to reduce compile time and improve code organization.
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/runtime/cuda/padded_device_halo_exchange.hpp>

namespace pfc::cuda {

// Explicit template instantiations for common Field types
template void exchange_halo<double, pfc::HostSpace>(
    pfc::data::Field<double, pfc::HostSpace>& field,
    const decomposition::Decomposition& decomp,
    int rank,
    int halo_width,
    MPI_Comm comm,
    cudaStream_t stream);

template void exchange_halo<double, CudaSpace>(
    pfc::data::Field<double, CudaSpace>& field,
    const decomposition::Decomposition& decomp,
    int rank,
    int halo_width,
    MPI_Comm comm,
    cudaStream_t stream);

template halo_buffer pack_halo_data<double, pfc::HostSpace>(
    const pfc::data::Field<double, pfc::HostSpace>& field);

template halo_buffer pack_halo_data<double, CudaSpace>(
    const pfc::data::Field<double, CudaSpace>& field);

template void unpack_halo_data<double, pfc::HostSpace>(
    pfc::data::Field<double, pfc::HostSpace>& field,
    const halo_buffer& buf);

template void unpack_halo_data<double, CudaSpace>(
    pfc::data::Field<double, CudaSpace>& field,
    const halo_buffer& buf);

// Explicit template instantiations for deprecated PaddedBrick wrappers
template void exchange_halo<double>(
    pfc::field::PaddedBrick<double>& brick,
    const decomposition::Decomposition& decomp,
    int rank,
    int halo_width,
    MPI_Comm comm,
    cudaStream_t stream);

template halo_buffer pack_halo_data<double>(
    const pfc::field::PaddedBrick<double>& brick);

template void unpack_halo_data<double>(
    pfc::field::PaddedBrick<double>& brick,
    const halo_buffer& buf);

} // namespace pfc::cuda

#endif // OpenPFC_ENABLE_CUDA