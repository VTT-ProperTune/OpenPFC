// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file elementwise_ops_gpu.hpp
 * @brief Generic device elementwise ops for CUDA and HIP (M3).
 *
 * Complex×real multiply, two-term diagonal combine
 * (`out = w0 * x0 + w1 * x1`), and axpy-style fill
 * (`out[i] = alpha * x[i] + beta`). Promoted from the mislabeled
 * Tungsten-specific kernels; used by M7's ETD skeleton.
 *
 * Compiled from `src/openpfc/runtime/gpu/elementwise_ops.cu` / `.hip`.
 *
 * @see apps/tungsten/include/tungsten/common/tungsten_ops.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <cstddef>

namespace pfc {

#if defined(OpenPFC_ENABLE_CUDA)
void multiply_complex_real_cuda_impl(const std::complex<double> *a, const double *b,
                                     std::complex<double> *out, std::size_t n);
void multiply_complex_real_cuda_impl(const std::complex<float> *a, const float *b,
                                     std::complex<float> *out, std::size_t n);
void combine_two_term_cuda_impl(const std::complex<double> *x0,
                                const std::complex<double> *x1, const double *w0,
                                const double *w1, std::complex<double> *out,
                                std::size_t n);
void combine_two_term_cuda_impl(const std::complex<float> *x0,
                                const std::complex<float> *x1, const float *w0,
                                const float *w1, std::complex<float> *out,
                                std::size_t n);
void axpy_fill_cuda_impl(double *out, const double *x, double alpha, double beta,
                         std::size_t n);
void axpy_fill_cuda_impl(float *out, const float *x, float alpha, float beta,
                         std::size_t n);
#endif

#if defined(OpenPFC_ENABLE_HIP)
void multiply_complex_real_hip_impl(const std::complex<double> *a, const double *b,
                                    std::complex<double> *out, std::size_t n);
void multiply_complex_real_hip_impl(const std::complex<float> *a, const float *b,
                                    std::complex<float> *out, std::size_t n);
void combine_two_term_hip_impl(const std::complex<double> *x0,
                               const std::complex<double> *x1, const double *w0,
                               const double *w1, std::complex<double> *out,
                               std::size_t n);
void combine_two_term_hip_impl(const std::complex<float> *x0,
                               const std::complex<float> *x1, const float *w0,
                               const float *w1, std::complex<float> *out,
                               std::size_t n);
void axpy_fill_hip_impl(double *out, const double *x, double alpha, double beta,
                        std::size_t n);
void axpy_fill_hip_impl(float *out, const float *x, float alpha, float beta,
                        std::size_t n);
#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
