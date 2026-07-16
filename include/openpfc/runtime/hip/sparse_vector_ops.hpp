// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file sparse_vector_ops.hpp
 * @brief HIP gather/scatter for SparseVector (runtime extension)
 *
 * Include this header when using gather() or scatter() with SparseVector<HipTag,
 * T>. Kernel-only code should not include this; kernel remains backend-agnostic when
 * HIP is disabled.
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#include <openpfc/runtime/hip/sparse_vector_ops_hip.hpp>
#include <stdexcept>

namespace pfc {
namespace core {

/** @brief Gather for SparseVector<HipTag, double> (HIP device). */
inline void gather(SparseVector<backend::HipTag, double> &sparse_vector,
                   const double *source, size_t source_size) {
  if (sparse_vector.empty()) {
    return;
  }
  gather_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                   sparse_vector.data().data(), source, source_size);
}

/** @brief Scatter for SparseVector<HipTag, double> (HIP device). */
inline void scatter(const SparseVector<backend::HipTag, double> &sparse_vector,
                    double *dest, size_t dest_size) {
  if (sparse_vector.empty()) {
    return;
  }
  scatter_hip_impl(sparse_vector.size(), sparse_vector.indices().data(),
                    sparse_vector.data().data(), dest, dest_size);
}

} // namespace core
} // namespace pfc

#endif
